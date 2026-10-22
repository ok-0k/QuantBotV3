"""
Money path, exits: the long-side and fallback branches of the 1-minute tick
check (bot._tick_exit_check) and the 30-second monitor (bot._monitor_position)
that the short-only live book never reaches today, plus the guards that keep a
corrupt row or a race from moving money. Each test asserts the stored levels,
whether the position is still open, and the fill price and journal fields of
any exit.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
from config import FEE_GATE_ROUND_TRIP, SLIPPAGE_PCT, STARTING_CASH, SYMBOLS
from conftest import make_candles

SYM = SYMBOLS[2]
ENTRY = 100.0


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    bot._tick_minute_stop.clear()
    monkeypatch.setattr(bot, "V3_FIXES", True)
    monkeypatch.setattr(bot, "_candles_cache", {})
    monkeypatch.setattr(bot, "_no_candle_warned", {})
    return clean_db


def _long(db, stop, tp, shares=1.0):
    db.open_position(SYM, shares, ENTRY, "QF", stop, tp)
    db.set_cash(STARTING_CASH - shares * ENTRY)
    return db.get_position(SYM)


def _is_open(db):
    return any(p["symbol"] == SYM for p in db.get_all_positions())


def _history():
    """Quiet 1m history at $100 in the cache (ATR ~0.06); returns the ATR."""
    candles = make_candles(n=120, start_price=ENTRY, drift=0.0, amplitude=0.04)
    bot._candles_cache[SYM] = candles
    return bot._wilder_atr(candles)


def _tick(high, low, close, minute=1):
    last = bot._candles_cache.get(SYM) or [{"time": int(time.time() * 1000) - 60_000}]
    candle = {"open": close, "high": high, "low": low, "close": close, "volume": 1.0,
              "time": last[-1]["time"] + minute * 60_000}
    asyncio.run(_run(bot._tick_exit_check, SYM, candle))
    return candle


async def _run(fn, *args):
    await fn(*args, asyncio.get_running_loop())


def _exits(db):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE action IN ('sell','cover') AND status='filled' ORDER BY id")]


def _long_net(exit_raw, shares=1.0):
    px = exit_raw * (1 - SLIPPAGE_PCT)
    gross = shares * (px - ENTRY)
    return px, gross - (shares * ENTRY + shares * px) / 2.0 * FEE_GATE_ROUND_TRIP


# ── tick path: long-side level management ────────────────────────────────────

def test_long_trailing_stop_ratchets_while_a_fee_losing_target_is_held(fresh, caplog):
    atr = _history()
    _long(fresh, stop=99.5, tp=100.1)                    # target too close to clear fees
    with caplog.at_level("INFO"):
        _tick(high=100.2, low=100.15, close=100.18)
    pos = fresh.get_position(SYM)
    assert pos["stop_price"] == pytest.approx(100.2 - atr)
    assert pos["tp_price"] == 100.1 and _exits(fresh) == []
    msgs = [r.getMessage() for r in caplog.records]
    assert any("TRAIL STOP " + SYM in m for m in msgs)
    assert any(m.startswith("FEE GATE blocked take_profit_tick") for m in msgs)


def test_long_break_even_arms_only_on_a_confirmed_close(fresh, caplog):
    atr = _history()
    _long(fresh, stop=99.5, tp=103.0)
    with caplog.at_level("INFO"):
        _tick(high=ENTRY + 5 * atr, low=ENTRY + 0.9 * atr, close=ENTRY + 1.0 * atr)   # wick only
    assert fresh.get_position(SYM)["stop_price"] == 99.5
    assert any("BE GUARD (long)" in r.getMessage() for r in caplog.records)
    _tick(high=ENTRY + 5 * atr, low=ENTRY + 1.9 * atr, close=ENTRY + 2.0 * atr, minute=2)
    assert fresh.get_position(SYM)["stop_price"] == pytest.approx(ENTRY)
    assert _exits(fresh) == []


@pytest.mark.parametrize("side", ["long", "short"])
def test_without_atr_the_break_even_close_check_uses_one_percent(fresh, caplog, side):
    bot._candles_cache[SYM] = []                         # no history: ATR unavailable
    if side == "long":
        _long(fresh, stop=95.0, tp=110.0)
        close = ENTRY * 1.005
    else:
        fresh.open_short(SYM, 1.0, ENTRY, "QF", 105.0, 90.0, 20.0)
        close = ENTRY * 0.995
    with caplog.at_level("INFO"):
        _tick(high=max(close, ENTRY), low=min(close, ENTRY), close=close)
    pos = fresh.get_position(SYM)
    assert pos["stop_price"] == (95.0 if side == "long" else 105.0)
    assert any(f"BE GUARD ({side}/no-ATR)" in r.getMessage() for r in caplog.records)


# ── tick path: long-side exits ───────────────────────────────────────────────

def test_long_stop_gapped_through_fills_at_the_market_not_the_stop(fresh):
    _history()
    _long(fresh, stop=99.5, tp=103.0)
    _tick(high=99.4, low=99.0, close=99.2)
    assert fresh.get_all_positions() == []
    row = _exits(fresh)[0]
    px, net = _long_net(99.2)
    assert row["exec_price"] == pytest.approx(px)        # min(stop, mark), then exit slippage
    assert row["exit_reason"] == "STOP_LOSS"
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - ENTRY + ENTRY + net)


def test_long_stop_moved_within_the_minute_triggers_on_the_mark_only(fresh):
    candles = make_candles(n=120, start_price=ENTRY, drift=0.0, amplitude=0.04)
    bot._candles_cache[SYM] = candles
    pos = _long(fresh, stop=99.8, tp=103.0)
    t = candles[-1]["time"] + 60_000
    bot._tick_minute_stop[SYM] = ((pos["opened_ts"], t), 99.5)    # the minute began with the stop at 99.5
    _tick(high=99.95, low=99.6, close=99.9)              # wick under the new stop, mark above it
    assert _is_open(fresh)
    _tick(high=99.95, low=99.6, close=99.75)             # same minute, mark now under it
    row = _exits(fresh)[0]
    assert row["exec_price"] == pytest.approx(99.75 * (1 - SLIPPAGE_PCT))


@pytest.mark.parametrize("v3, booked", [(True, "stop"), (False, "target")])
def test_candle_touching_both_levels_books_the_stop_with_v3(fresh, monkeypatch, v3, booked):
    monkeypatch.setattr(bot, "V3_FIXES", v3)
    _history()
    _long(fresh, stop=99.5, tp=100.8)
    _tick(high=100.9, low=99.4, close=100.0)
    row = _exits(fresh)[0]
    raw = 99.5 if booked == "stop" else 100.8
    assert row["exec_price"] == pytest.approx(raw * (1 - SLIPPAGE_PCT))
    assert row["exit_reason"] == ("STOP_LOSS" if booked == "stop" else "TAKE_PROFIT")


def test_long_take_profit_fills_at_the_resting_limit(fresh):
    _history()
    _long(fresh, stop=99.5, tp=100.8)
    _tick(high=100.9, low=100.5, close=100.87)           # closes within 1 ATR of the high
    row = _exits(fresh)[0]
    px, net = _long_net(100.8)
    assert row["exec_price"] == pytest.approx(px) and row["net_pnl"] == pytest.approx(round(net, 2))
    assert row["exit_reason"] == "TAKE_PROFIT"


@pytest.mark.xfail(strict=True, reason=(
    "BUG: with V3_FIXES a candle that trades through a fee-clearing take-profit and then closes "
    ">= 1 ATR off its extreme is booked as a TRAILING_STOP at the close, not a TAKE_PROFIT at tp. "
    "The trail (extreme -/+ ATR) is derived from that same candle, so under the bot's own fill model "
    "(take-profit = resting limit, fills AT tp) the target had already filled; the stop-before-target "
    "ordering then fires the just-created trail on the mark. Short e.g.: tp 99.0, low 98.9, close 99.2 "
    "-> cover at 99.2 (TRAILING_STOP) instead of 99.0. Legacy (V3 off) books the target."))
@pytest.mark.parametrize("side", ["short", "long"])
def test_touched_take_profit_is_not_pre_empted_by_a_trail_from_the_same_candle(fresh, side):
    atr = _history()
    if side == "short":
        fresh.open_short(SYM, 1.0, ENTRY, "QF", 100.55, 99.0, 20.0)
        tp, candle = 99.0, dict(high=99.25, low=98.9, close=99.2)
    else:
        _long(fresh, stop=99.5, tp=100.8)
        tp, candle = 100.8, dict(high=100.9, low=100.5, close=100.6)
    assert abs(candle["close"] - (candle["low"] if side == "short" else candle["high"])) > atr
    _tick(**candle)
    row = _exits(fresh)[0]
    assert row["exit_reason"] == "TAKE_PROFIT"
    slip = (1 + SLIPPAGE_PCT) if side == "short" else (1 - SLIPPAGE_PCT)
    assert row["exec_price"] == pytest.approx(tp * slip)


def test_legacy_survival_floor_fires_when_no_level_is_touched(fresh, monkeypatch):
    monkeypatch.setattr(bot, "V3_FIXES", False)
    _history()
    _long(fresh, stop=95.0, tp=110.0, shares=10.0)
    _tick(high=98.4, low=98.2, close=98.3)               # open P&L -$17 < -$15, stop untouched
    row = _exits(fresh)[0]
    assert row["exec_price"] == pytest.approx(98.3 * (1 - SLIPPAGE_PCT))
    assert row["exit_reason"] == "HARD_STOP"
    assert fresh.get_all_positions() == []


# ── tick path: guards ────────────────────────────────────────────────────────

def test_rows_without_exposure_are_never_exited(fresh):
    """A short row with no reserved margin, or a long row with no shares
    (corrupt / half-written), is skipped even with its stop breached."""
    _history()
    fresh.open_short(SYM, 1.0, ENTRY, "QF", 100.5, 97.0, 0.0)
    _tick(high=101.0, low=100.6, close=100.8)
    assert fresh.get_position(SYM)["shares"] == 1.0
    fresh.delete_position(SYM)
    fresh.open_position(SYM, 0.0, ENTRY, "QF", 99.5, 103.0)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET margin_reserved=5.0 WHERE symbol=?", (SYM,))
    _tick(high=99.4, low=99.0, close=99.2, minute=2)
    assert _is_open(fresh)
    assert _exits(fresh) == [] and fresh.get_cash() == pytest.approx(STARTING_CASH)


def test_position_closed_before_the_lock_is_not_exited_twice(fresh, monkeypatch):
    _history()
    _long(fresh, stop=99.5, tp=103.0)
    real, calls = bot.get_all_positions, []

    def racing():
        calls.append(1)
        return real() if len(calls) == 1 else []         # closed by another task after the first look
    monkeypatch.setattr(bot, "get_all_positions", racing)
    sent = []
    monkeypatch.setattr(bot._exec_adapter, "execute", lambda i: sent.append(i))
    _tick(high=99.4, low=99.0, close=99.2)
    assert sent == [] and _exits(fresh) == []
    assert SYM not in bot._exit_in_flight


# ── monitor path ─────────────────────────────────────────────────────────────

def _monitor(pos):
    asyncio.run(_run(bot._monitor_position, pos))


def _mark(price):
    candles = make_candles(n=120, start_price=ENTRY, drift=0.0, amplitude=0.04)
    candles[-1] = dict(candles[-1], close=price, high=max(price, candles[-1]["high"]),
                       low=min(price, candles[-1]["low"]))
    bot._candles_cache[SYM] = candles


def test_zombie_shell_row_is_deleted_before_any_maths(fresh, caplog):
    fresh.open_position(SYM, 1.0, ENTRY, "QF", 99.5, 103.0)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET shares=0 WHERE symbol=?", (SYM,))
    shell = fresh.get_position(SYM)
    with caplog.at_level("WARNING"):
        _monitor(shell)
    with fresh.get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM positions WHERE symbol=?", (SYM,)).fetchone()[0] == 0
    assert fresh.get_cash() == pytest.approx(STARTING_CASH) and _exits(fresh) == []
    assert any("ZOMBIE SHELL detected for " + SYM in r.getMessage() for r in caplog.records)


def test_missing_candles_warn_once_per_ten_minutes_and_touch_nothing(fresh, caplog):
    pos = _long(fresh, stop=99.5, tp=103.0)
    with caplog.at_level("WARNING"):
        _monitor(pos)
        _monitor(pos)
    assert sum("has no candle data yet" in r.getMessage() for r in caplog.records) == 1
    assert fresh.get_position(SYM)["candle_count"] == 0             # not even counted
    assert SYM in bot._no_candle_warned


@pytest.mark.parametrize("price, reason", [(99.4, "STOP_LOSS"), (100.9, "TAKE_PROFIT")])
def test_monitor_closes_a_long_at_the_mark(fresh, price, reason):
    pos = _long(fresh, stop=99.5, tp=100.8)
    _mark(price)
    _monitor(pos)
    row = _exits(fresh)[0]
    px, net = _long_net(price)
    assert (row["action"], row["exit_reason"]) == ("sell", reason)
    assert row["exec_price"] == pytest.approx(px)
    assert fresh.get_cash() == pytest.approx(STARTING_CASH + net, abs=1e-6)


def test_hard_max_hold_exit_bypasses_the_fee_gate(fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_HARD_MAX_HOLD_EXIT", True)
    pos = _long(fresh, stop=99.0, tp=103.0)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET candle_count=? WHERE symbol=?", (bot.MAX_HOLD_CANDLES - 1, SYM))
    _mark(100.05)                                        # tiny winner: net after fees is negative
    _monitor(fresh.get_position(SYM))
    row = _exits(fresh)[0]
    assert row["exit_reason"] == "MAX_HOLD" and row["net_pnl"] < 0


def test_unparseable_open_time_never_triggers_the_time_exit(fresh):
    pos = _long(fresh, stop=99.0, tp=103.0)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET opened_ts='not-a-date' WHERE symbol=?", (SYM,))
    _mark(99.8)                                          # losing, inside its stop
    _monitor(fresh.get_position(SYM))
    assert _is_open(fresh) and _exits(fresh) == []


def test_monitor_exit_that_raises_leaves_the_position_for_the_next_cycle(fresh, monkeypatch, caplog):
    pos = _long(fresh, stop=99.5, tp=103.0)
    _mark(99.0)

    def boom(*a, **k):
        raise RuntimeError("sqlite busy")
    monkeypatch.setattr(bot, "_prepare_trade", boom)
    with caplog.at_level("ERROR"):
        _monitor(pos)
    assert fresh.get_position(SYM)["shares"] == 1.0
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - ENTRY)
    assert any("trade execution failed for " + SYM in r.getMessage() for r in caplog.records)


def test_four_hour_time_exit_applies_to_longs_too(fresh):
    pos = _long(fresh, stop=98.0, tp=103.0)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET opened_ts=? WHERE symbol=?",
                     ((datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(), SYM))
    _mark(99.7)
    _monitor(fresh.get_position(SYM))
    assert _exits(fresh)[0]["exit_reason"] == "TIME_HOLD_EXIT"
