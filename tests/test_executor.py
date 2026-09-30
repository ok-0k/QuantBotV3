"""
Characterization tests for bot._execute_trade — the money-pulling layer.

These pin the CURRENT paper-fill behaviour: gate ordering, sizing, slippage,
cash mutation, fee-deducted PnL, and trade-record contents. The Phase-3
execution-adapter refactor must keep every one of these green (except where a
new protection intentionally changes behaviour — those get updated in the same
commit with an explanation).
"""

import numpy as np
import pytest

import bot
from config import (
    FEE_GATE_ROUND_TRIP,
    SHORT_MARGIN_PCT,
    SLIPPAGE_PCT,
    STARTING_CASH,
)
from conftest import make_candles


ZERO_STATE = np.zeros(13, dtype=np.float32)


def _entry_ensemble(symbol="AAAUSDT", action="buy", price=100.0, ml_prob=0.97,
                    candles=None):
    return {
        "signal": action,
        "symbol": symbol,
        "price": price,
        "regime": "ranging" if action == "buy" else "trend_down",
        "on_fire": False,
        "position_size_mult": 1.0,
        "buy_weight": 0.5,
        "sell_weight": 0.5,
        "ml_prob": ml_prob,
        "breakdown": [],
        "time": "2026-07-11T00:00:00Z",
    }


def _exit_ensemble(symbol="AAAUSDT", action="sell", price=110.0):
    return {
        "signal": action,
        "symbol": symbol,
        "price": price,
        "regime": "ranging",
        "on_fire": False,
        "position_size_mult": 1.0,
        "buy_weight": 0.0,
        "sell_weight": 0.0,
        "breakdown": [],
        "time": "2026-07-11T00:00:00Z",
    }


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    """Clean DB + neutral brain edge profiles so sizing is deterministic."""
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    return clean_db


def _last_trade_row(db):
    with db.get_db() as conn:
        row = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


# ── Entry gates ──────────────────────────────────────────────────────────────

def test_buy_blocked_below_ml_confidence(fresh):
    candles = make_candles(n=120)
    ens = _entry_ensemble(ml_prob=0.60)   # < MIN_ML_CONFIDENCE (0.80)
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)
    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"
    assert "ml_gate_blocked" in row["reason"]


def test_buy_blocked_missing_ml_prob(fresh):
    candles = make_candles(n=120)
    ens = _entry_ensemble()
    del ens["ml_prob"]
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    row = _last_trade_row(fresh)
    assert row["reason"] == "ml_gate_missing_prob"


def test_short_gate_uses_one_minus_ml_prob(fresh):
    candles = make_candles(n=120)
    # ml_prob 0.9 -> short conviction 0.1 -> blocked
    ens = _entry_ensemble(action="short", ml_prob=0.9)
    bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"
    assert "ml_gate_blocked" in row["reason"]


def test_sac_veto_aborts_buy_without_cash_change(fresh):
    candles = make_candles(n=120)
    ens = _entry_ensemble(ml_prob=0.97)
    bot._execute_trade(ens, "TEST", candles, -0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)
    # Veto path returns trade_value=0 -> logged as insufficient_funds skip
    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"


def test_exits_bypass_ml_gate(fresh):
    """sell/cover must never be blocked by the ML confidence gate."""
    candles = make_candles(n=120, start_price=100.0)
    fresh.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 120.0)
    fresh.set_cash(STARTING_CASH - 200.0)
    ens = _exit_ensemble(price=110.0)
    ens["ml_prob"] = None   # would fail the entry gate; exits must not care
    bot._execute_trade(ens, "TEST", candles, None, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []   # position closed


# ── Successful BUY: sizing, slippage, cash conservation ─────────────────────

def test_buy_full_pipeline_numbers(fresh):
    candles = make_candles(n=120, start_price=100.0, drift=0.0, amplitude=0.4)
    raw_price = candles[-1]["close"]
    ens = _entry_ensemble(price=raw_price, ml_prob=0.97)
    sac = 0.5

    bot._execute_trade(ens, "TEST", candles, sac, ZERO_STATE, STARTING_CASH)

    positions = fresh.get_all_positions()
    assert len(positions) == 1
    p = positions[0]

    # Slippage: buys fill above the reference price
    expected_exec = raw_price * (1 + SLIPPAGE_PCT)
    assert p["avg_cost"] == pytest.approx(expected_exec, rel=1e-9)

    # Sizing: dyn mult at ml=0.97 -> 1.0 + ((0.97-0.80)/0.19)*1.5 = 2.342105...
    # cm = 1.0 * dyn = 2.342105; raw = sac*cm = 1.171 -> clamped to ceiling 0.35
    # edge size_mult = 1.0 (fresh) -> trade_value = 3500
    expected_value = STARTING_CASH * 0.35
    assert p["shares"] * p["avg_cost"] == pytest.approx(expected_value, rel=1e-9)

    # Cash conservation: cash + position cost = starting cash
    assert fresh.get_cash() + expected_value == pytest.approx(STARTING_CASH)

    # Stops bracket the entry
    assert p["stop_price"] < expected_exec < p["tp_price"]

    row = _last_trade_row(fresh)
    assert row["status"] == "filled"
    assert row["action"] == "buy"
    assert row["trade_value"] == pytest.approx(expected_value, rel=1e-6)


def test_buy_insufficient_funds_skips(fresh):
    candles = make_candles(n=120)
    fresh.set_cash(100.0)   # trade_value will exceed available cash
    ens = _entry_ensemble(ml_prob=0.97)
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, 10_000.0)
    assert fresh.get_all_positions() == []
    row = _last_trade_row(fresh)
    assert row["reason"] == "insufficient_funds"
    assert fresh.get_cash() == pytest.approx(100.0)


def test_buy_edge_too_small_blocked(fresh):
    # Flat tape (ATR floors at 0.2% of price) + ~$10.8 notional:
    # tp_gross ~= 10.8 * 2.7% = $0.29 minus fees -> below the $0.35 gate.
    # ml_prob=0.80 keeps the dynamic size multiplier at 1.0 so the scaled
    # SAC floor cannot push the trade value back above the gate.
    candles = make_candles(n=120, start_price=100.0, drift=0.0, amplitude=0.0)
    ens = _entry_ensemble(ml_prob=0.80)
    bot._execute_trade(ens, "TEST", candles, 0.036, ZERO_STATE, 300.0)
    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"
    assert row["reason"] == "edge_too_small"
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)


# ── SELL: fee-deducted PnL and cash refund ───────────────────────────────────

def test_sell_pnl_and_cash_flow(fresh):
    candles = make_candles(n=120, start_price=110.0, drift=0.0, amplitude=0.4)
    shares, avg_cost = 2.0, 100.0
    cost = shares * avg_cost
    fresh.open_position("AAAUSDT", shares, avg_cost, "TEST", 95.0, 130.0)
    cash_before = STARTING_CASH - cost
    fresh.set_cash(cash_before)

    raw_exit = 110.0
    ens = _exit_ensemble(price=raw_exit)
    bot._execute_trade(ens, "TEST", candles, None, ZERO_STATE, STARTING_CASH)

    exec_price = raw_exit * (1 - SLIPPAGE_PCT)      # sells fill below reference
    proceeds = shares * exec_price
    gross = proceeds - cost
    fee = ((cost + proceeds) / 2.0) * FEE_GATE_ROUND_TRIP
    net = gross - fee

    assert fresh.get_all_positions() == []
    # Refund = original cost basis + net pnl (fees actually deducted from cash)
    assert fresh.get_cash() == pytest.approx(cash_before + cost + net, rel=1e-9)

    row = _last_trade_row(fresh)
    assert row["status"] == "filled"
    assert row["gross_pnl"] == pytest.approx(round(gross, 2))
    assert row["net_pnl"] == pytest.approx(round(net, 2))
    assert row["fee_total"] == pytest.approx(round(fee, 6))
    # An RL experience row is persisted for every close
    assert len(fresh.get_rl_experience()) == 1


def test_sell_without_position_skips(fresh):
    candles = make_candles(n=120)
    bot._execute_trade(_exit_ensemble(), "TEST", candles, None, ZERO_STATE, STARTING_CASH)
    row = _last_trade_row(fresh)
    assert row["reason"] == "no_position"
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)


# ── SHORT + COVER: margin flow ───────────────────────────────────────────────

def test_short_reserves_margin_and_cover_returns_it(fresh):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    raw_price = candles[-1]["close"]
    ens = _entry_ensemble(action="short", price=raw_price, ml_prob=0.03)

    bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)

    positions = fresh.get_all_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p["side"] == "short"

    exec_entry = raw_price * (1 - SLIPPAGE_PCT)     # short entries fill below
    assert p["avg_cost"] == pytest.approx(exec_entry, rel=1e-9)

    trade_value = p["shares"] * exec_entry
    margin = p["margin_reserved"]
    assert margin == pytest.approx(trade_value * SHORT_MARGIN_PCT, rel=1e-9)
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - margin, rel=1e-9)
    # Short stops sit above entry, targets below
    assert p["tp_price"] < exec_entry < p["stop_price"]

    # ── Cover at a profit ────────────────────────────────────────────────
    cash_before_cover = fresh.get_cash()
    shares = p["shares"]
    entry_cost = shares * p["avg_cost"]
    raw_cover = raw_price * 0.95
    cov = _exit_ensemble(action="cover", price=raw_cover)
    bot._execute_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)

    exec_cover = raw_cover * (1 + SLIPPAGE_PCT)     # covers (buys) fill above
    cover_cost = shares * exec_cover
    gross = entry_cost - cover_cost
    fee = ((entry_cost + cover_cost) / 2.0) * FEE_GATE_ROUND_TRIP
    net = gross - fee

    assert fresh.get_all_positions() == []
    assert fresh.get_cash() == pytest.approx(cash_before_cover + margin + net, rel=1e-9)
    row = _last_trade_row(fresh)
    assert row["action"] == "cover"
    assert row["net_pnl"] == pytest.approx(round(net, 2))


def test_short_twice_same_symbol_rejected_no_capital_leak(fresh):
    """C3: re-shorting an already-open short must not overwrite margin/stop
    tracking or double-debit cash. Previously open_short()'s INSERT OR REPLACE
    silently destroyed the first short's tracked margin_reserved (deducted
    from cash but never refundable again) while a second margin_reserved was
    also deducted -- a permanent, invisible capital leak."""
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    raw_price = candles[-1]["close"]
    ens = _entry_ensemble(action="short", price=raw_price, ml_prob=0.03)

    bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    p1 = fresh.get_all_positions()[0]
    cash_after_first = fresh.get_cash()
    assert p1["margin_reserved"] > 0

    # A second short signal on the SAME symbol, one candle later -> a new
    # candle_ts -> a new client_order_id -> bypasses the duplicate-order
    # journal guard, so this genuinely exercises the existing-position guard
    # rather than order-journal idempotency.
    next_candles = make_candles(n=121, start_price=100.0, drift=-0.05, amplitude=0.4)
    assert next_candles[-1]["time"] != candles[-1]["time"]
    ens2 = _entry_ensemble(action="short", price=next_candles[-1]["close"], ml_prob=0.03)
    bot._execute_trade(ens2, "QF", next_candles, 0.5, ZERO_STATE, STARTING_CASH)

    # Rejected: no second margin debit, original position completely untouched.
    assert fresh.get_cash() == pytest.approx(cash_after_first)
    positions = fresh.get_all_positions()
    assert len(positions) == 1
    p2 = positions[0]
    assert p2["avg_cost"] == pytest.approx(p1["avg_cost"])
    assert p2["margin_reserved"] == pytest.approx(p1["margin_reserved"])
    assert p2["shares"] == pytest.approx(p1["shares"])

    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"
    assert row["reason"] == "short_already_open"


# ── V4 protections at the executor ───────────────────────────────────────────
# (Replaces the pre-V4 characterization test that documented duplicate buys
#  averaging in and doubling exposure — that behaviour is now intentionally
#  blocked by the order journal's deterministic client-order-id claim.)

def test_buy_twice_same_candle_is_blocked_as_duplicate(fresh):
    candles = make_candles(n=120, start_price=100.0)
    ens = _entry_ensemble(ml_prob=0.97)
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)
    cash_after_first = fresh.get_cash()
    first_value = STARTING_CASH - cash_after_first
    assert first_value > 0

    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)

    # Second identical signal: no extra cash deployed, position unchanged
    assert fresh.get_cash() == pytest.approx(cash_after_first)
    p = fresh.get_all_positions()[0]
    assert p["shares"] * p["avg_cost"] == pytest.approx(first_value, rel=1e-6)
    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"
    assert row["reason"] == "duplicate_order_blocked"


def test_buy_on_next_candle_still_allowed(fresh):
    candles = make_candles(n=120, start_price=100.0)
    ens = _entry_ensemble(ml_prob=0.97)
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)
    deployed_first = STARTING_CASH - fresh.get_cash()

    next_candles = make_candles(n=121, start_price=100.0)   # newer last candle
    assert next_candles[-1]["time"] != candles[-1]["time"]
    bot._execute_trade(ens, "TEST", next_candles, 0.5, ZERO_STATE, STARTING_CASH)

    # New candle -> new order id -> averaging-in proceeds as before
    assert (STARTING_CASH - fresh.get_cash()) > deployed_first * 1.5


def test_entry_blocked_on_stale_candles(fresh):
    stale_candles = make_candles(n=120, start_time_ms=1_700_000_000_000)  # years old
    ens = _entry_ensemble(ml_prob=0.97)
    bot._execute_trade(ens, "TEST", stale_candles, 0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)
    row = _last_trade_row(fresh)
    assert row["status"] == "skipped"
    assert "stale_data" in row["reason"]


def test_exit_never_blocked_by_stale_candles(fresh):
    stale_candles = make_candles(n=120, start_price=110.0,
                                 start_time_ms=1_700_000_000_000)
    fresh.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0)
    fresh.set_cash(STARTING_CASH - 200.0)
    bot._execute_trade(_exit_ensemble(price=110.0), "TEST", stale_candles,
                       None, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []   # closed despite dead feed


def test_order_cap_clamps_oversized_trade(fresh, monkeypatch):
    monkeypatch.setattr(bot, "MAX_ORDER_EQUITY_FRAC", 0.10)
    candles = make_candles(n=120, start_price=100.0)
    ens = _entry_ensemble(ml_prob=0.97)
    # Sizing would produce 35% of equity; the cap must clamp it to 10%.
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)
    p = fresh.get_all_positions()[0]
    assert p["shares"] * p["avg_cost"] == pytest.approx(STARTING_CASH * 0.10, rel=1e-9)


def test_filled_entry_writes_journal_row(fresh):
    candles = make_candles(n=120, start_price=100.0)
    ens = _entry_ensemble(ml_prob=0.97)
    bot._execute_trade(ens, "TEST", candles, 0.5, ZERO_STATE, STARTING_CASH)
    orders = fresh.get_open_orders()
    assert orders == []                       # FILLED is terminal
    with fresh.get_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM orders").fetchall()]
    assert len(rows) == 1
    assert rows[0]["state"] == "FILLED"
    assert rows[0]["mode"] == "paper"
    assert rows[0]["filled_qty"] == pytest.approx(
        fresh.get_all_positions()[0]["shares"])


def test_failed_exit_keeps_position_open(fresh, monkeypatch):
    """If the adapter cannot execute an exit, the position must survive."""
    from execution.base import Fill

    class FailingAdapter:
        name = "paper"
        def execute(self, intent):
            return Fill(status="FAILED", note="synthetic outage")

    monkeypatch.setattr(bot, "_exec_adapter", FailingAdapter())
    candles = make_candles(n=120, start_price=110.0)
    fresh.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0)
    fresh.set_cash(STARTING_CASH - 200.0)
    bot._execute_trade(_exit_ensemble(price=110.0), "TEST", candles,
                       None, ZERO_STATE, STARTING_CASH)
    assert len(fresh.get_all_positions()) == 1          # still open
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 200.0)  # no refund
    row = _last_trade_row(fresh)
    assert row["reason"] == "execution_failed"


def test_partial_exit_books_actual_fill_and_keeps_remainder(fresh, monkeypatch):
    """Testnet-style partial fill: book the filled qty, keep the rest open."""
    from execution.base import Fill

    class PartialAdapter:
        name = "paper"
        def execute(self, intent):
            return Fill(status="PARTIALLY_FILLED", qty=intent.qty * 0.5,
                        price=intent.limit_price, note="partial")

    monkeypatch.setattr(bot, "_exec_adapter", PartialAdapter())
    candles = make_candles(n=120, start_price=110.0)
    fresh.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0)
    fresh.set_cash(STARTING_CASH - 200.0)

    bot._execute_trade(_exit_ensemble(price=110.0), "TEST", candles,
                       None, ZERO_STATE, STARTING_CASH)

    positions = fresh.get_all_positions()
    assert len(positions) == 1
    assert positions[0]["shares"] == pytest.approx(1.0)   # half remains

    exec_price = 110.0 * (1 - SLIPPAGE_PCT)
    filled = 1.0
    cost = filled * 100.0
    proceeds = filled * exec_price
    gross = proceeds - cost
    fee = ((cost + proceeds) / 2.0) * FEE_GATE_ROUND_TRIP
    net = gross - fee
    # Cash refunded only for the closed half
    assert fresh.get_cash() == pytest.approx(
        (STARTING_CASH - 200.0) + cost + net, rel=1e-9)
