"""
Money path, entries: every refusal branch of bot._evaluate_and_trade before
and inside the commit frame, driven through the real coroutine with a
qualifying short signal. Each test asserts what the refusal leaves behind:
no position, untouched cash, and the journaled skip row (reason + features)
or the log line when the branch is not journaled.
"""

import asyncio
import json
import time

import numpy as np
import pytest

import bot
from config import STARTING_CASH, SYMBOLS
from conftest import make_candles

SYM = SYMBOLS[0]
REAL_REFRESH_MARGIN = bot._refresh_margin_health
PEERS = [s for s in SYMBOLS[1:] if s != "BTCUSDT" and s != "ETHUSDT"]


def _short_ens(symbol, price, **over):
    ens = {"signal": "short", "symbol": symbol, "price": price, "regime": "ranging",
           "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
           "ml_prob": 0.05, "breakdown": [{"signal": "sell", "strategy": "QF", "alloc": 0.7}],
           "time": "2026-10-11T00:00:00Z", "meta": {"short_score": 0.7}}
    ens.update(over)
    return ens


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    monkeypatch.setattr(bot, "V3_FIXES", True)
    monkeypatch.setattr(bot, "_symbol_last_trade", {})
    monkeypatch.setattr(bot, "_entry_block_journaled", {})
    monkeypatch.setattr(bot, "_entry_in_flight", {})
    monkeypatch.setattr(bot, "_candles_cache", {})
    monkeypatch.setattr(bot, "_margin_health_cache", {})
    monkeypatch.setattr(bot, "_shutdown_event", asyncio.Event())

    async def _no_margin(*a, **k):
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", _no_margin)
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: 0.5)
    return clean_db


@pytest.fixture()
def evaluate(fresh, monkeypatch):
    """evaluate(live=<live ml_prob>, candles=..., **ensemble overrides) -> the
    symbols the ensemble was asked about (empty if the coroutine returned first)."""
    def run(live=0.05, candles=None, **over):
        candles = candles or make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
        calls = []

        def ens(*a, **k):
            calls.append(a[0])
            return _short_ens(SYM, candles[-1]["close"], **over)
        monkeypatch.setattr(bot.brain, "get_ensemble_signal", ens)

        async def go():
            await bot._evaluate_and_trade(SYM, candles, live, asyncio.get_running_loop())
        asyncio.run(go())
        return calls
    return run


def _rows(db, **where):
    sql = "SELECT * FROM trades" + (" WHERE " + " AND ".join(f"{k}=?" for k in where) if where else "")
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY id", tuple(where.values()))]


def _nothing_happened(db):
    assert db.get_all_positions() == []
    assert db.get_cash() == pytest.approx(STARTING_CASH)


# ── before the ensemble is asked ─────────────────────────────────────────────

def test_shutdown_skips_the_symbol_without_stamping_the_throttle(evaluate, fresh):
    bot._shutdown_event.set()
    assert evaluate() == []
    assert SYM not in bot._symbol_last_trade
    _nothing_happened(fresh)
    assert _rows(fresh) == []


def test_evaluation_throttle_blocks_a_second_look_within_the_window(evaluate, fresh):
    stamp = time.time()
    bot._symbol_last_trade[SYM] = stamp
    assert evaluate() == []
    assert bot._symbol_last_trade[SYM] == stamp
    _nothing_happened(fresh)


def test_hold_signal_stamps_the_throttle_and_trades_nothing(evaluate, fresh):
    assert evaluate(signal="hold") == [SYM]
    assert SYM in bot._symbol_last_trade
    _nothing_happened(fresh)
    assert _rows(fresh) == []


def test_stale_feed_is_skipped_before_any_sizing_work(evaluate, fresh, caplog, monkeypatch):
    sized = []
    monkeypatch.setattr(bot, "compute_position_fraction", lambda s: sized.append(1) or 0.5)
    old = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4, start_time_ms=1_700_000_000_000)
    with caplog.at_level("INFO"):
        evaluate(candles=old)
    _nothing_happened(fresh)
    assert sized == [] and _rows(fresh) == []
    assert any("SHORT skipped — stale_data" in r.getMessage() for r in caplog.records)


# ── ML-probability guards (refused, not journaled) ───────────────────────────

@pytest.mark.parametrize("bad", [None, "abc", 1.5, -0.1])
def test_ensemble_without_a_valid_ml_prob_is_refused(evaluate, fresh, caplog, bad):
    with caplog.at_level("WARNING"):
        evaluate(ml_prob=bad)                         # the ensemble's own field
    _nothing_happened(fresh)
    assert _rows(fresh) == []
    assert any("ensemble missing/invalid ml_prob" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("live, text", [
    ("abc", "live ml_prob unusable"),
    (1.5, "live ml_prob out of range"),
    (0.5, "live ML confidence 0.5000 < 0.8000"),     # short conviction = 1 - 0.5
])
def test_live_ml_prob_is_the_authority_for_entries(evaluate, fresh, caplog, live, text):
    with caplog.at_level("INFO"):
        evaluate(live=live)                           # the ensemble itself carries a valid 0.05
    _nothing_happened(fresh)
    assert _rows(fresh) == []
    assert any(text in r.getMessage() for r in caplog.records)


# ── structural refusals ──────────────────────────────────────────────────────

def test_existing_short_on_the_symbol_is_never_reopened(evaluate, fresh, caplog):
    fresh.open_short(SYM, 2.0, 100.0, "QF", 101.0, 97.0, 40.0)
    fresh.set_cash(STARTING_CASH - 40.0)
    with caplog.at_level("WARNING"):
        evaluate()
    pos = fresh.get_short_position(SYM)
    assert pos["shares"] == 2.0 and pos["margin_reserved"] == 40.0
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 40.0)
    assert _rows(fresh) == []
    assert any("a short is already open" in r.getMessage() for r in caplog.records)


def test_a_dead_flat_last_candle_is_not_traded(evaluate, fresh):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    candles[-1] = dict(candles[-1], high=candles[-1]["close"] + 0.001, low=candles[-1]["close"] - 0.001)
    evaluate(candles=candles)
    _nothing_happened(fresh)
    assert _rows(fresh) == []


# ── commit-frame gates (journaled as skipped rows) ───────────────────────────

def test_live_margin_level_below_floor_halts_new_entries(evaluate, fresh, monkeypatch):
    """Real _refresh_margin_health + evaluate_margin_health on a stubbed SAPI
    snapshot: margin level 1.05 < 1.12 halts; the halt is persisted for the
    dashboard and journaled with the entry's features."""
    monkeypatch.setattr(bot, "_refresh_margin_health", REAL_REFRESH_MARGIN)

    async def snapshot(session):
        return {"marginLevel": "1.05", "userAssets": [{"asset": "USDT", "free": "250.5"},
                                                      {"asset": "usdt", "free": "bad"},
                                                      {"asset": "BTC", "free": "1"}]}
    monkeypatch.setattr(bot, "fetch_margin_account_snapshot", snapshot)
    evaluate()
    _nothing_happened(fresh)
    row = _rows(fresh, status="skipped")[-1]
    assert row["reason"] == "margin_halt"
    health = json.loads(fresh.get_portfolio_stat("margin_health_json"))
    assert health["halt_new_entries"] is True
    assert health["halt_reason"] == "margin_level=1.0500<1.12"
    assert health["source"] == "binance_sapi" and health["available_margin_usdt"] == 250.5
    assert health["utilization_pct"] == 0.0


def test_highly_correlated_open_peers_block_the_entry(evaluate, fresh):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    for peer in PEERS[:2]:                            # two open shorts moving exactly like SYM
        fresh.open_short(peer, 1.0, 100.0, "QF", 101.0, 97.0, 20.0)
        bot._candles_cache[peer] = candles
    fresh.set_cash(STARTING_CASH - 40.0)
    evaluate(candles=candles)
    assert fresh.get_short_position(SYM) is None
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 40.0)
    row = _rows(fresh, status="skipped")[-1]
    assert row["reason"] == "corr_block"
    assert json.loads(row["entry_features"])["worst_rho"] == pytest.approx(1.0, abs=1e-4)


def test_open_long_notional_counts_toward_the_exposure_cap(evaluate, fresh):
    """A long worth 80% of equity (marked at its cost) leaves no room under the
    82% notional cap for the new short; the long's P&L enters equity too."""
    fresh.open_position(PEERS[0], 80.0, 100.0, "QF", 95.0, 110.0)
    fresh.set_cash(STARTING_CASH - 8_000.0)
    evaluate()
    assert fresh.get_short_position(SYM) is None
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 8_000.0)
    assert _rows(fresh, status="skipped")[-1]["reason"] == "exposure_block"


@pytest.mark.parametrize("v3, blocked", [(True, True), (False, False)])
def test_in_flight_entries_count_toward_exposure_only_with_v3(evaluate, fresh, monkeypatch, v3, blocked):
    monkeypatch.setattr(bot, "V3_FIXES", v3)
    bot._entry_in_flight[PEERS[0]] = {"action": "buy", "arm": None, "qty": 80.0, "price": 100.0}
    evaluate()
    assert (fresh.get_short_position(SYM) is None) is blocked
    if blocked:
        assert _rows(fresh, status="skipped")[-1]["reason"] == "exposure_block"


def test_breaker_tripping_during_sizing_is_caught_at_the_commit_frame(evaluate, fresh, monkeypatch):
    def trip(state):
        bot.brain.circuit_open = True
        return 0.5
    monkeypatch.setattr(bot, "compute_position_fraction", trip)
    evaluate()
    _nothing_happened(fresh)
    assert [r["reason"] for r in _rows(fresh, status="skipped")] == ["circuit_breaker"]


def test_short_opened_while_sizing_blocks_the_duplicate(evaluate, fresh, monkeypatch, caplog):
    def race(state):
        fresh.open_short(SYM, 3.0, 100.0, "QF", 101.0, 97.0, 60.0)    # another task won the race
        return 0.5
    monkeypatch.setattr(bot, "compute_position_fraction", race)
    with caplog.at_level("WARNING"):
        evaluate()
    pos = fresh.get_short_position(SYM)
    assert pos["shares"] == 3.0 and pos["margin_reserved"] == 60.0     # the winner, untouched
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)
    assert _rows(fresh, status="filled") == []
    assert any("blocked at commit frame" in r.getMessage() for r in caplog.records)


def test_slots_filled_while_sizing_are_rechecked(evaluate, fresh, monkeypatch):
    def fill_slots(state):
        for peer in PEERS[:5]:
            fresh.open_short(peer, 1.0, 100.0, "QF", 101.0, 97.0, 20.0)
        return 0.5
    monkeypatch.setattr(bot, "compute_position_fraction", fill_slots)
    evaluate()
    assert fresh.get_short_position(SYM) is None
    assert _rows(fresh, status="skipped")[-1]["reason"] == "slots_full"


# ── sizing outcomes through the coroutine ────────────────────────────────────

def test_sac_crash_falls_back_to_two_percent_sizing(evaluate, fresh, monkeypatch, caplog):
    def crash(state):
        raise RuntimeError("weights corrupt")
    monkeypatch.setattr(bot, "compute_position_fraction", crash)
    with caplog.at_level("WARNING"):
        evaluate()
    pos = fresh.get_short_position(SYM)
    assert pos is not None
    row = _rows(fresh, status="filled", action="short")[0]
    feats = json.loads(row["entry_features"])
    assert feats["sac_fraction"] is None
    cm = bot._authoritative_conviction_mult(_short_ens(SYM, 1.0), 0.05)
    expected = bot._entry_proposed_equity_frac(bot._SAC_FALLBACK_PCT,
                                               cm * bot.dynamic_conviction_size_mult(0.05, "short"))
    assert feats["trade_pct"] == pytest.approx(expected, abs=1e-6)
    assert row["trade_value"] == pytest.approx(STARTING_CASH * expected, rel=1e-3)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("SAC inference error" in m for m in msgs) and any("falling back to 2%" in m for m in msgs)


def test_sac_veto_through_the_coroutine_books_nothing(evaluate, fresh, monkeypatch):
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: -0.5)
    evaluate()
    _nothing_happened(fresh)
    assert _rows(fresh, status="skipped")[-1]["reason"] == "sac_veto"
    assert bot._entry_in_flight == {}


# ── exits through the coroutine ──────────────────────────────────────────────

def test_cover_signal_closes_the_short_and_returns_margin_plus_pnl(evaluate, fresh):
    fresh.open_short(SYM, 2.0, 100.0, "QF", 103.0, 90.0, 40.0)
    fresh.set_cash(STARTING_CASH - 40.0)
    candles = make_candles(n=120, start_price=98.0, drift=-0.01, amplitude=0.4)
    evaluate(candles=candles, signal="cover")
    assert fresh.get_short_position(SYM) is None
    row = _rows(fresh, status="filled", action="cover")[0]
    assert row["exec_price"] == pytest.approx(candles[-1]["close"] * (1 + bot.SLIPPAGE_PCT))
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 40.0 + 40.0 + row["net_pnl"], abs=0.01)
    assert row["net_pnl"] > 0


def test_cover_signal_without_a_position_is_journaled_no_position(evaluate, fresh):
    evaluate(signal="cover")
    _nothing_happened(fresh)
    row = _rows(fresh)[-1]
    assert (row["action"], row["status"], row["reason"]) == ("cover", "skipped", "no_position")


def test_long_entries_stop_at_the_open_position_cap(evaluate, fresh, monkeypatch):
    monkeypatch.setattr(bot, "MAX_OPEN_POSITIONS", 2)
    for peer in PEERS[:2]:
        fresh.open_position(peer, 1.0, 100.0, "QF", 95.0, 110.0)
    fresh.set_cash(STARTING_CASH - 200.0)
    evaluate(live=0.97, signal="buy", ml_prob=0.97)
    assert {p["symbol"] for p in fresh.get_all_positions()} == set(PEERS[:2])
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 200.0) and _rows(fresh) == []
