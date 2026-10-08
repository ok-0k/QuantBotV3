"""
The 30 s exit_monitor path (previously untested): one real monitor cycle per
test — stop, take-profit gate (hold vs book), 4h time exit, survival stop,
and the equity snapshot each cycle writes.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
from config import SLIPPAGE_PCT, STARTING_CASH, SYMBOLS
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
SYM = SYMBOLS[0]          # experiment arm A: legacy decay, a no-op on a fresh position


@pytest.fixture()
def fresh(clean_db):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    return clean_db


def _open_short(db):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    ens = {"signal": "short", "symbol": SYM, "price": candles[-1]["close"], "regime": "ranging",
           "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
           "ml_prob": 0.05, "breakdown": [], "time": "2026-10-08T00:00:00Z", "meta": {}}
    bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    return candles, db.get_short_position(SYM)


def _set(db, **cols):
    with db.get_db() as conn:
        for k, v in cols.items():
            conn.execute(f"UPDATE positions SET {k}=? WHERE symbol=?", (v, SYM))


def _one_cycle(monkeypatch, candles, price):
    """Run exit_monitor for exactly one cycle with the mark at `price`."""
    last = dict(candles[-1], close=price, high=max(price, candles[-1]["high"]), low=min(price, candles[-1]["low"]))
    bot._candles_cache[SYM] = candles[:-1] + [last]
    recorded = []

    async def run():
        ev = asyncio.Event()
        monkeypatch.setattr(bot, "_shutdown_event", ev)
        monkeypatch.setattr(bot, "CHECK_EVERY_SECS", 0.01)
        monkeypatch.setattr(bot, "record_equity", lambda eq: (recorded.append(eq), ev.set()))
        await asyncio.wait_for(bot.exit_monitor(asyncio.get_running_loop()), 10)

    asyncio.run(run())
    return recorded


def _covers(db):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE action='cover' AND status='filled' ORDER BY id")]


def test_monitor_stop_fires_at_market_and_is_labelled(fresh, monkeypatch):
    candles, pos = _open_short(fresh)
    # $1,500 test notional: keep the stop inside the -$15 survival floor (live
    # positions are ~$160, where that floor is ~9% away and never interferes).
    stop = pos["avg_cost"] * 1.004
    _set(fresh, stop_price=stop)
    price = stop * 1.001
    recorded = _one_cycle(monkeypatch, candles, price)
    assert fresh.get_short_position(SYM) is None
    row = _covers(fresh)[0]
    assert row["exit_reason"] == "TIGHTENED_STOP"               # moved in from the recorded initial stop
    assert row["exec_price"] == pytest.approx(price * (1 + SLIPPAGE_PCT), rel=1e-9)
    assert len(recorded) == 1                                   # one equity snapshot per cycle


def test_monitor_take_profit_held_when_net_would_be_negative(fresh, monkeypatch):
    candles, pos = _open_short(fresh)
    tp = pos["avg_cost"] * (1 - 0.0021)                         # clears the fee, not fee + slippage
    _set(fresh, tp_price=tp)
    _one_cycle(monkeypatch, candles, tp)
    assert fresh.get_short_position(SYM) is not None and _covers(fresh) == []


def test_monitor_take_profit_books_when_net_positive(fresh, monkeypatch):
    candles, pos = _open_short(fresh)
    tp = pos["avg_cost"] * (1 - 0.006)
    _set(fresh, tp_price=tp)
    _one_cycle(monkeypatch, candles, tp)
    row = _covers(fresh)[0]
    assert row["exit_reason"] == "TAKE_PROFIT" and row["net_pnl"] > 0


def test_monitor_four_hour_time_exit_closes_a_loser(fresh, monkeypatch):
    candles, pos = _open_short(fresh)
    avg = pos["avg_cost"]
    _set(fresh, stop_price=avg * 1.05, tp_price=avg * 0.95,
         opened_ts=(datetime.now(timezone.utc) - timedelta(hours=5)).isoformat())
    _one_cycle(monkeypatch, candles, avg * 1.002)                # losing, nowhere near the stop
    assert _covers(fresh)[0]["exit_reason"] == "TIME_HOLD_EXIT"


def test_monitor_time_exit_leaves_a_winner_alone(fresh, monkeypatch):
    candles, pos = _open_short(fresh)
    avg = pos["avg_cost"]
    _set(fresh, stop_price=avg * 1.05, tp_price=avg * 0.95,
         opened_ts=(datetime.now(timezone.utc) - timedelta(hours=5)).isoformat())
    _one_cycle(monkeypatch, candles, avg * 0.998)                # in profit -> keep running
    assert fresh.get_short_position(SYM) is not None


def test_monitor_survival_stop_beats_everything(fresh, monkeypatch):
    candles, pos = _open_short(fresh)
    avg = pos["avg_cost"]
    notional = pos["shares"] * avg
    _set(fresh, stop_price=avg * 1.05)                           # stop far away
    price = avg * (1 + 20.0 / notional)                          # open loss ~ -$20 < -$15 floor
    _one_cycle(monkeypatch, candles, price)
    row = _covers(fresh)[0]
    assert row["exit_reason"] == "HARD_STOP"
    assert row["exec_price"] == pytest.approx(price * (1 + SLIPPAGE_PCT), rel=1e-9)
