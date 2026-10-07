"""Tests for the execution adapter layer — paper passthrough, testnet safety."""

import pytest

from execution.base import Fill, OrderIntent, make_client_order_id
from execution.paper import PaperAdapter


def _intent(**kw):
    base = dict(
        client_order_id="qb-test", symbol="AAAUSDT", action="buy", side="long",
        qty=2.0, ref_price=100.0, limit_price=100.015, candle_ts=1_700_000_000_000,
    )
    base.update(kw)
    return OrderIntent(**base)


# ── client order ids ─────────────────────────────────────────────────────────

def test_client_order_id_deterministic_and_binance_safe():
    a = make_client_order_id("BTCUSDT", "buy", 1_700_000_000_000)
    b = make_client_order_id("BTCUSDT", "buy", 1_700_000_000_000)
    assert a == b                       # same candle -> same id (dedupe key)
    assert a.startswith("qb-")
    assert len(a) <= 36
    assert all(c.isalnum() or c in "-_" for c in a)


def test_client_order_id_varies_by_dimension():
    base = make_client_order_id("BTCUSDT", "buy", 1000)
    assert make_client_order_id("ETHUSDT", "buy", 1000) != base
    assert make_client_order_id("BTCUSDT", "short", 1000) != base
    assert make_client_order_id("BTCUSDT", "buy", 2000) != base


# ── paper adapter ────────────────────────────────────────────────────────────

def test_paper_fills_full_qty_at_limit_price():
    fill = PaperAdapter().execute(_intent())
    assert fill.status == "FILLED"
    assert fill.executed
    assert fill.qty == pytest.approx(2.0)
    assert fill.price == pytest.approx(100.015)


def test_paper_rejects_degenerate_intent():
    assert not PaperAdapter().execute(_intent(qty=0.0)).executed
    assert not PaperAdapter().execute(_intent(limit_price=0.0)).executed


def test_paper_reconcile_reports_never_executed():
    fill = PaperAdapter().reconcile("qb-anything")
    assert fill is not None
    assert fill.status == "REJECTED"
    assert fill.qty == 0.0


# ── adapter selection ────────────────────────────────────────────────────────

def test_get_adapter_defaults_to_paper():
    import execution
    execution._adapter = None
    adapter = execution.get_execution_adapter()
    assert adapter.name == "paper"
    execution._adapter = None


# ── testnet adapter (all HTTP mocked — no network) ───────────────────────────

@pytest.fixture()
def testnet(monkeypatch):
    import execution.binance_testnet as bt
    monkeypatch.setattr(bt, "BINANCE_TESTNET_API_KEY", "test-key")
    monkeypatch.setattr(bt, "BINANCE_TESTNET_API_SECRET", "test-secret")
    adapter = bt.BinanceTestnetAdapter()
    adapter._limiter.acquire = lambda w: None      # no sleeping in tests
    adapter._filters["AAAUSDT"] = {"stepSize": 0.01, "minNotional": 10.0, "minQty": 0.01}
    return adapter


def test_quantize_rounds_down_to_step():
    from execution.binance_testnet import BinanceTestnetAdapter as A
    assert A.quantize_qty(2.379, 0.01) == pytest.approx(2.37)
    assert A.quantize_qty(2.379, 0.0) == pytest.approx(2.379)   # no filter -> untouched
    assert A.quantize_qty(0.005, 0.01) == 0.0


def test_testnet_rejects_below_min_notional_locally(testnet):
    # 0.05 qty * $100 = $5 < $10 min notional -> rejected with NO http call
    calls = []
    testnet._request = lambda *a, **k: calls.append(a) or (500, {})
    fill = testnet.execute(_intent(qty=0.05))
    assert fill.status == "REJECTED"
    assert "notional" in fill.note
    assert calls == []


def test_testnet_full_fill_flow(testnet):
    def fake_request(method, path, params=None, **kw):
        assert path == "/order" and method == "POST"
        assert params["newClientOrderId"] == "qb-test"
        assert params["type"] == "MARKET"
        return 200, {
            "orderId": 555, "status": "FILLED",
            "executedQty": "2.0", "cummulativeQuoteQty": "200.10",
        }
    testnet._request = fake_request
    fill = testnet.execute(_intent())
    assert fill.status == "FILLED"
    assert fill.qty == pytest.approx(2.0)
    assert fill.price == pytest.approx(100.05)     # 200.10 / 2.0 weighted avg
    assert fill.exchange_order_id == "555"


def test_testnet_partial_fill_reports_actual_qty(testnet, monkeypatch):
    monkeypatch.setattr("execution.binance_testnet.ORDER_POLL_TIMEOUT_SECS", 0.0)
    def fake_request(method, path, params=None, **kw):
        if path == "/order" and method == "POST":
            return 200, {"orderId": 7, "status": "PARTIALLY_FILLED",
                         "executedQty": "0.8", "cummulativeQuoteQty": "80.0"}
        if path == "/order" and method == "GET":
            return 200, {"orderId": 7, "status": "PARTIALLY_FILLED",
                         "executedQty": "0.8", "cummulativeQuoteQty": "80.0"}
        raise AssertionError(f"unexpected {method} {path}")
    testnet._request = fake_request
    fill = testnet.execute(_intent())
    assert fill.status == "PARTIALLY_FILLED"
    assert fill.qty == pytest.approx(0.8)          # book what actually happened
    assert fill.price == pytest.approx(100.0)


def test_testnet_timeout_queries_before_retry_and_adopts(testnet):
    """A transport timeout must NEVER blind-resubmit: query by id, adopt result."""
    import requests as _rq
    posts, gets = [], []
    def fake_request(method, path, params=None, **kw):
        if method == "POST":
            posts.append(params)
            raise _rq.ConnectTimeout("boom")
        gets.append(params)
        return 200, {"orderId": 9, "status": "FILLED",
                     "executedQty": "2.0", "cummulativeQuoteQty": "200.0"}
    testnet._request = fake_request
    fill = testnet.execute(_intent())
    assert fill.status == "FILLED"
    assert len(posts) == 1                          # exactly one submission
    assert gets and gets[0]["origClientOrderId"] == "qb-test"


def test_testnet_duplicate_id_adopts_existing_order(testnet):
    def fake_request(method, path, params=None, **kw):
        if method == "POST":
            return 400, {"code": -2010, "msg": "Duplicate order sent."}
        return 200, {"orderId": 11, "status": "FILLED",
                     "executedQty": "2.0", "cummulativeQuoteQty": "199.0"}
    testnet._request = fake_request
    fill = testnet.execute(_intent())
    assert fill.status == "FILLED"
    assert fill.exchange_order_id == "11"


def test_testnet_exchange_rejection_maps_to_rejected(testnet):
    testnet._request = lambda *a, **k: (400, {"code": -2019, "msg": "Margin is insufficient."})
    fill = testnet.execute(_intent())
    assert fill.status == "REJECTED"
    assert "insufficient" in fill.note.lower()


def test_testnet_unknown_symbol_rejects(testnet):
    testnet._request = lambda *a, **k: (400, {"code": -1121, "msg": "Invalid symbol."})
    fill = testnet.execute(_intent(symbol="NOPEUSDT"))
    assert fill.status == "REJECTED"
    assert "unavailable" in fill.note


# ── rate limiter ─────────────────────────────────────────────────────────────

def test_token_bucket_blocks_until_refill():
    from execution.binance_testnet import TokenBucketLimiter
    clock = [0.0]
    sleeps = []
    def fake_sleep(s):
        sleeps.append(s)
        clock[0] += s
    limiter = TokenBucketLimiter(60, clock=lambda: clock[0], sleeper=fake_sleep)  # 1/sec
    limiter.acquire(60)          # drain the bucket
    limiter.acquire(10)          # must wait ~10s of refill
    assert sum(sleeps) == pytest.approx(10.0, abs=0.2)
