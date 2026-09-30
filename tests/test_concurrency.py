"""
Concurrency tests for the C4 lock-scope fix — the exchange adapter call must
run with _strict_execution_lock released, so a slow order for one symbol
cannot block another symbol's stop-loss/entry from acquiring the lock.

These exercise the exact prepare -> release -> adapter -> reacquire -> commit
pattern every async call site in bot.py uses (_tick_exit_check,
_evaluate_and_trade, exit_monitor), using bot's real lock objects and real
_prepare_trade/_commit_trade, with a synthetic ensemble dict (same technique
test_executor.py already uses for _execute_trade) rather than driving
brain's full signal pipeline, which is a separate concern already covered by
test_brain.py.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

import bot
from config import STARTING_CASH
from conftest import make_candles
from execution.base import Fill

ZERO_STATE = np.zeros(13, dtype=np.float32)

# Distinct from the symbols test_executor.py uses, so this file's lock
# objects (bot._trade_locks is a process-wide dict shared across the whole
# test session) never overlap with another test's in-flight state.
_SYM_A = "ZZZCONCURA"
_SYM_B = "ZZZCONCURB"


@pytest.fixture()
def fresh(clean_db):
    """Clean DB + neutral brain state so sizing/gates are deterministic."""
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    return clean_db


def _entry_ensemble(symbol: str, price: float, ml_prob: float = 0.97) -> dict:
    return {
        "signal": "buy", "symbol": symbol, "price": price,
        "regime": "ranging", "on_fire": False, "position_size_mult": 1.0,
        "buy_weight": 0.5, "sell_weight": 0.5, "ml_prob": ml_prob,
        "breakdown": [], "time": "2026-07-11T00:00:00Z",
    }


class _SlowAdapter:
    """Simulates a slow exchange round-trip (e.g. testnet order polling)."""
    name = "paper"

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0

    def execute(self, intent):
        self.calls += 1
        time.sleep(self.delay)
        return Fill(status="FILLED", qty=intent.qty, price=intent.limit_price, note="slow fill")


async def _run_trade_with_lock_dance(ensemble: dict, candles: list[dict], symbol: str):
    """Mirrors the exact lock pattern every async call site in bot.py uses:
    _trade_lock(symbol) outermost; _strict_execution_lock held only for the
    prepare and commit phases, released around the exchange call."""
    loop = asyncio.get_event_loop()
    async with bot._trade_lock(symbol):
        async with bot._strict_execution_lock:
            prepared = await loop.run_in_executor(
                None, bot._prepare_trade, ensemble, "TEST", candles, 0.5,
                ZERO_STATE, STARTING_CASH, None,
            )
        if prepared is None:
            return None
        fill = await loop.run_in_executor(None, bot._exec_adapter.execute, prepared.order_intent)
        async with bot._strict_execution_lock:
            await loop.run_in_executor(None, bot._commit_trade, prepared, fill)
        return fill


def test_c4_slow_adapter_call_does_not_hold_global_lock(fresh, monkeypatch):
    """The core C4 property: while symbol A's (slow) adapter call is in
    flight, symbol B must be able to acquire _strict_execution_lock promptly
    -- not wait for A's entire round-trip. Before the fix, this lock was
    held across the whole call and B would have to wait ~0.3s too."""
    slow = _SlowAdapter(delay=0.3)
    monkeypatch.setattr(bot, "_exec_adapter", slow)

    candles = make_candles(n=120, start_price=100.0)
    ens_a = _entry_ensemble(_SYM_A, price=candles[-1]["close"])

    async def contender_b():
        # Give A's task a moment to reach the (now-unlocked) adapter call.
        await asyncio.sleep(0.05)
        start = time.monotonic()
        async with bot._strict_execution_lock:
            pass
        return time.monotonic() - start

    async def main():
        return await asyncio.gather(
            _run_trade_with_lock_dance(ens_a, candles, _SYM_A),
            contender_b(),
        )

    fill_a, wait_b = asyncio.run(main())
    assert fill_a is not None and fill_a.executed
    assert slow.calls == 1
    assert wait_b < 0.15, (
        f"_strict_execution_lock was held during the exchange call "
        f"(a different symbol waited {wait_b:.3f}s for it)"
    )


def test_c4_same_symbol_still_serialized_across_network_call(fresh, monkeypatch):
    """Narrowing the global lock must not accidentally let two operations on
    the SAME symbol race each other during the network call -- _trade_lock
    must still serialize repeat access to one symbol end-to-end."""
    slow = _SlowAdapter(delay=0.2)
    monkeypatch.setattr(bot, "_exec_adapter", slow)

    candles = make_candles(n=120, start_price=100.0)
    ens_a = _entry_ensemble(_SYM_B, price=candles[-1]["close"])

    async def second_same_symbol_attempt():
        await asyncio.sleep(0.05)
        start = time.monotonic()
        async with bot._trade_lock(_SYM_B):
            pass
        return time.monotonic() - start

    async def main():
        return await asyncio.gather(
            _run_trade_with_lock_dance(ens_a, candles, _SYM_B),
            second_same_symbol_attempt(),
        )

    _, wait_same_symbol = asyncio.run(main())
    # Must wait for roughly the full adapter delay (unlike the
    # different-symbol case above), proving same-symbol exclusion survived.
    assert wait_same_symbol > 0.15, (
        f"_trade_lock(symbol) did not serialize same-symbol access across "
        f"the network call (only waited {wait_same_symbol:.3f}s)"
    )
