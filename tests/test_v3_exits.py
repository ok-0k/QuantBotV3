"""
Exit paths: what the trade journal records on every exit (on by default),
and the V3_FIXES exit behaviour (off by default, deferred until decay_v2
ends): stops/targets only ever tighten, break-even and trailing stops survive
time decay, tick-path order survival -> stop -> take-profit, a stop moved
within the minute triggers on the mark only, and exit rewards credit the
regime the position was entered in.

Each V3 behaviour is also asserted with the flag off where the legacy result
differs, so the flag demonstrably gates it.
"""

import asyncio
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
from config import (DECAY_HALFLIFE_CANDLES, DECAY_SL_TIGHTEN_STRENGTH, SLIPPAGE_PCT,
                    STARTING_CASH, SYMBOLS)
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
A_SYM = SYMBOLS[0]        # experiment arm A (legacy decay)
B_SYM = SYMBOLS[1]        # experiment arm B (designed decay)
ENTRY, SL0, TP0 = 100.0, 100.55, 97.4


@pytest.fixture()
def fresh(clean_db):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    return clean_db


def _ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _place_short(db, sym, arm="B", age_min=1, count=0, regime="ranging", stop=SL0, tp=TP0):
    """A 1-share short at $100 (survival floor -$15 is a 15% move away)."""
    feats = {"experiment": "decay_v2", "arm": arm, "initial_stop": SL0,
             "initial_tp": TP0, "regime": regime}
    db.open_short(sym, 1.0, ENTRY, "QF", stop, tp, 20.0, entry_features=feats)
    with db.get_db() as conn:
        conn.execute("UPDATE positions SET opened_ts=?, candle_count=? WHERE symbol=?",
                     (_ago(age_min), count, sym))
    return db.get_short_position(sym)


def _history(sym):
    """Quiet 1m history at $100 (ATR ~0.06) in the cache; returns (candles, atr)."""
    candles = make_candles(n=120, start_price=ENTRY, drift=0.0, amplitude=0.04)
    bot._candles_cache[sym] = candles
    return candles, bot._wilder_atr(candles)


def _tick(sym, candles, high, low, close, minute=1):
    candle = {"open": close, "high": high, "low": low, "close": close,
              "volume": 1.0, "time": candles[-1]["time"] + minute * 60_000}

    async def run():
        await bot._tick_exit_check(sym, candle, asyncio.get_running_loop())
    asyncio.run(run())


def _stop(db, sym):
    return db.get_short_position(sym)["stop_price"]


def _covers(db):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE action='cover' AND status='filled' ORDER BY id")]


def _be_candle(atr):
    """Deep, confirmed favourable move: arms break-even, touches nothing else."""
    return dict(high=ENTRY - 0.5 * atr, low=ENTRY - 5.0 * atr, close=ENTRY - 2.0 * atr)


def _designed_target_sl(age_min):
    lam = 1 - math.exp(-age_min / DECAY_HALFLIFE_CANDLES)
    return SL0 + lam * DECAY_SL_TIGHTEN_STRENGTH * (ENTRY - SL0)


def _spy_stop_writes(monkeypatch):
    """Records (new_stop, row_written) for every update_stop_price call from bot."""
    writes = []
    real = bot.update_stop_price

    def spy(sym, new_stop, only_tighten=False):
        ok = real(sym, new_stop, only_tighten=only_tighten)
        writes.append((new_stop, ok))
        return ok
    monkeypatch.setattr(bot, "update_stop_price", spy)
    return writes


# ── db: optional directional guard ───────────────────────────────────────────

def test_update_stop_price_only_tighten_guard(fresh):
    _place_short(fresh, B_SYM)
    assert fresh.update_stop_price(B_SYM, 100.7, only_tighten=True) is False   # loosen: refused
    assert _stop(fresh, B_SYM) == SL0
    assert fresh.update_stop_price(B_SYM, 100.2, only_tighten=True) is True    # tighten: written
    assert _stop(fresh, B_SYM) == 100.2
    assert fresh.update_stop_price(B_SYM, 100.9) is True                       # default: unguarded
    assert _stop(fresh, B_SYM) == 100.9


@pytest.mark.parametrize("side", ["long", None])
def test_update_stop_price_guard_treats_non_short_as_long(fresh, side):
    _place_short(fresh, B_SYM)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET side=?, stop_price=95.0, tp_price=110.0 WHERE symbol=?",
                     (side, B_SYM))
    get = lambda col: fresh.get_position(B_SYM)[col]
    assert fresh.update_stop_price(B_SYM, 94.0, only_tighten=True) is False
    assert fresh.update_stop_price(B_SYM, 96.0, only_tighten=True) is True and get("stop_price") == 96.0
    assert fresh.update_tp_price(B_SYM, 111.0, only_tighten=True) is False
    assert fresh.update_tp_price(B_SYM, 108.0, only_tighten=True) is True and get("tp_price") == 108.0


def test_update_tp_price_only_tightens_toward_entry_and_fills_empty(fresh):
    _place_short(fresh, B_SYM)
    assert fresh.update_tp_price(B_SYM, 97.0, only_tighten=True) is False      # further away
    assert fresh.update_tp_price(B_SYM, 98.0, only_tighten=True) is True       # toward entry
    assert fresh.get_short_position(B_SYM)["tp_price"] == 98.0
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET stop_price=NULL, tp_price=0 WHERE symbol=?", (B_SYM,))
    assert fresh.update_stop_price(B_SYM, 101.0, only_tighten=True) is True    # none stored
    assert fresh.update_tp_price(B_SYM, 96.0, only_tighten=True) is True
