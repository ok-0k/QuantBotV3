"""
Concurrency tests for the C4 lock-scope fix, driving bot.py's real
coroutines (_tick_exit_check, exit_monitor, _evaluate_and_trade) against a
slow exchange adapter:

  * _strict_execution_lock is released while an order is at the exchange, so
    one symbol's slow order never blocks another symbol's exit;
  * _trade_lock(symbol) still spans prepare -> exchange call -> booking, so
    operations on the SAME symbol stay serialised.

Every async call site that sends an order has its own test, so moving any one
exchange call back under the global lock fails the suite. Assertions are on
the order of events (order sent / filled / booked), never on wall-clock
thresholds: SLOW_ORDER_SECS only has to outlast one fast exit on the Pi.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

import bot
from config import STARTING_CASH, SYMBOLS
from conftest import make_candles
from execution.base import Fill

ZERO_STATE = np.zeros(13, dtype=np.float32)
SYM_A, SYM_B = SYMBOLS[2], SYMBOLS[4]   # both experiment arm A: legacy decay, a no-op on fresh positions
SLOW_ORDER_SECS = 1.0                   # A's exchange round-trip; B's whole exit takes tens of ms


class _SlowExchange:
    """Paper fills; orders for the symbols in `slow` first block their
    executor thread with time.sleep (a slow testnet round-trip). Logs
    ("send" | "fill" | "booked", symbol) in the order they happen, from
    whichever thread they happen on."""

    name = "paper"

    def __init__(self, slow: dict[str, float]):
        self.slow = slow
        self._mu = threading.Lock()
        self._t0 = time.monotonic()
        self._log: list[tuple[str, str, float]] = []

    def mark(self, what: str, symbol: str) -> None:
        with self._mu:
            self._log.append((what, symbol, time.monotonic() - self._t0))

    @property
    def events(self) -> list[tuple[str, str]]:
        with self._mu:
            return [(what, sym) for what, sym, _ in self._log]

    def __str__(self) -> str:                    # the timeline, for assertion messages
        with self._mu:
            return ", ".join(f"{w} {s} @{t * 1000:.0f}ms" for w, s, t in self._log)

    def execute(self, intent):
        self.mark("send", intent.symbol)
        time.sleep(self.slow.get(intent.symbol, 0.0))
        self.mark("fill", intent.symbol)
        return Fill(status="FILLED", qty=intent.qty, price=intent.limit_price, note="slow paper fill")


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    """Clean DB + neutral brain, and per-test copies of bot's asyncio state:
    an asyncio.Lock binds to the first event loop that contends it, and every
    test here runs (and contends) its own loop."""
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    monkeypatch.setattr(bot, "_strict_execution_lock", asyncio.Lock())
    monkeypatch.setattr(bot, "_cash_lock", asyncio.Lock())
    monkeypatch.setattr(bot, "_trade_locks", {})
    monkeypatch.setattr(bot, "_candles_cache", {})
    monkeypatch.setattr(bot, "_shutdown_event", asyncio.Event())

    async def _no_margin_refresh(*_a, **_k):     # no aiohttp session, no Binance call
        return {}

    monkeypatch.setattr(bot, "_refresh_margin_health", _no_margin_refresh)
    return clean_db


def _ensemble(symbol: str, candles: list[dict], signal: str = "short") -> dict:
    return {"signal": signal, "symbol": symbol, "price": candles[-1]["close"], "regime": "ranging",
            "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
            "ml_prob": 0.05, "breakdown": [], "time": "2026-10-08T00:00:00Z", "meta": {}}


def _candles() -> list[dict]:
    return make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)


def _open_short(db, sym: str, candles: list[dict]) -> dict:
    bot._execute_trade(_ensemble(sym, candles), "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    return db.get_short_position(sym)


def _short_with_stop(db, sym: str, *, close_through_stop: bool) -> dict:
    """Open a paper short on `sym` with its stop 0.4% above entry (inside the
    -$15 survival floor at the $1,500 test notional; take-profit far away) and
    install a live candle whose high trades through that stop -- its close
    too, if asked -- as the newest cached bar, the way the websocket handler
    does before it schedules the tick check. Returns that candle."""
    candles = _candles()
    pos = _open_short(db, sym, candles)
    stop = pos["avg_cost"] * 1.004
    with db.get_db() as conn:
        conn.execute("UPDATE positions SET stop_price=?, tp_price=? WHERE symbol=?",
                     (stop, pos["avg_cost"] * 0.95, sym))
    through = stop * 1.001
    live = dict(candles[-1], high=max(through, candles[-1]["high"]))
    if close_through_stop:
        live["close"] = through
    bot._candles_cache[sym] = candles[:-1] + [live]
    return live


def _slow_exchange(monkeypatch, slow: dict[str, float]) -> _SlowExchange:
    """Install a _SlowExchange and log every booking (_commit_trade) into its
    timeline. Call after the test's positions are open."""
    ex = _SlowExchange(slow)
    monkeypatch.setattr(bot, "_exec_adapter", ex)
    commit = bot._commit_trade

    def _commit_and_log(prepared, fill):
        commit(prepared, fill)
        ex.mark("booked", prepared.symbol)

    monkeypatch.setattr(bot, "_commit_trade", _commit_and_log)
    return ex


def _in_order(ex: _SlowExchange, *events: tuple[str, str]) -> bool:
    """Every event happened, in this order."""
    seen = ex.events
    if not all(e in seen for e in events):
        return False
    idx = [seen.index(e) for e in events]
    return idx == sorted(idx)


async def _tick_when_sent(ex: _SlowExchange, sent: str, sym: str, candle: dict) -> None:
    """Deliver `sym`'s websocket tick once `sent`'s order is at the exchange."""
    deadline = time.monotonic() + 10
    while ("send", sent) not in ex.events:
        assert time.monotonic() < deadline, f"{sent}'s order never reached the exchange: {ex}"
        await asyncio.sleep(0.002)
    await bot._tick_exit_check(sym, candle, asyncio.get_running_loop())


async def _one_monitor_cycle(monkeypatch) -> None:
    """Exactly one real exit_monitor cycle (as tests/test_exit_monitor.py runs it)."""
    done = asyncio.Event()
    monkeypatch.setattr(bot, "_shutdown_event", done)
    monkeypatch.setattr(bot, "CHECK_EVERY_SECS", 0.01)
    monkeypatch.setattr(bot, "record_equity", lambda eq: done.set())
    await bot.exit_monitor(asyncio.get_running_loop())


def _run(coro) -> None:
    asyncio.run(asyncio.wait_for(coro, timeout=30))


def _covers(db, sym: str) -> list[dict]:
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT status, reason FROM trades WHERE symbol=? AND action='cover' ORDER BY id", (sym,))]


def test_slow_tick_exit_does_not_block_another_symbols_tick_exit(fresh, monkeypatch):
    """The core C4 property: two ticks breach two stops at once. While A's
    cover hangs at the exchange, B's cover is sent, filled and booked. With
    the global lock held across the exchange call (pre-C4), B could not even
    be prepared until A's order had returned."""
    ticks = {s: _short_with_stop(fresh, s, close_through_stop=True) for s in (SYM_A, SYM_B)}
    ex = _slow_exchange(monkeypatch, slow={SYM_A: SLOW_ORDER_SECS})

    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            bot._tick_exit_check(SYM_A, ticks[SYM_A], loop),
            bot._tick_exit_check(SYM_B, ticks[SYM_B], loop),
        )

    _run(main())
    assert fresh.get_short_position(SYM_A) is None and fresh.get_short_position(SYM_B) is None
    assert _in_order(ex, ("send", SYM_A), ("send", SYM_B), ("booked", SYM_B), ("fill", SYM_A)), (
        f"B's exit waited for A's exchange call: {ex}")


def test_same_symbol_stays_serialised_while_its_order_is_at_the_exchange(fresh, monkeypatch):
    """_trade_lock(A) spans prepare -> exchange call -> booking: while A's
    cover is at the exchange, a second tick for A is dropped and the 30 s
    monitor's exit for A waits, then finds nothing left to close. One order
    is sent, one cover is booked."""
    tick = _short_with_stop(fresh, SYM_A, close_through_stop=True)
    ex = _slow_exchange(monkeypatch, slow={SYM_A: SLOW_ORDER_SECS})
    at_second_tick: list[tuple[str, str]] = []

    async def same_symbol_while_in_flight():
        await _tick_when_sent(ex, SYM_A, SYM_A, tick)        # a second websocket tick for A
        at_second_tick.extend(ex.events)
        await _one_monitor_cycle(monkeypatch)                  # the monitor sees A through its stop too

    async def main():
        await asyncio.gather(
            bot._tick_exit_check(SYM_A, tick, asyncio.get_running_loop()),
            same_symbol_while_in_flight(),
        )

    _run(main())
    assert ("fill", SYM_A) not in at_second_tick, f"second tick came too late to test anything: {ex}"
    assert ex.events.count(("send", SYM_A)) == 1, f"two orders in flight for one symbol: {ex}"
    assert _covers(fresh, SYM_A) == [
        {"status": "filled", "reason": None},
        {"status": "skipped", "reason": "no_position"},      # the monitor's turn came after the booking
    ]


def test_slow_monitor_exit_does_not_block_another_symbols_tick_exit(fresh, monkeypatch):
    """Same property at exit_monitor's call site: the monitor's cover for A
    hangs at the exchange while a tick's wick stops B out."""
    _short_with_stop(fresh, SYM_A, close_through_stop=True)              # the monitor exits A
    tick_b = _short_with_stop(fresh, SYM_B, close_through_stop=False)    # only B's live wick crosses
    ex = _slow_exchange(monkeypatch, slow={SYM_A: SLOW_ORDER_SECS})

    async def main():
        await asyncio.gather(
            _one_monitor_cycle(monkeypatch),
            _tick_when_sent(ex, SYM_A, SYM_B, tick_b),
        )

    _run(main())
    assert fresh.get_short_position(SYM_A) is None and fresh.get_short_position(SYM_B) is None
    assert _in_order(ex, ("send", SYM_A), ("send", SYM_B), ("booked", SYM_B), ("fill", SYM_A)), (
        f"B's exit waited for the monitor's exchange call: {ex}")


@pytest.mark.parametrize("signal", ["short", "cover"])
def test_slow_signal_order_does_not_block_another_symbols_tick_exit(fresh, monkeypatch, signal):
    """Same property at _evaluate_and_trade's two call sites -- a short entry
    and a strategy-signalled cover: A's order hangs at the exchange while a
    tick stops B out. Only the inputs are stubbed (signal, SAC size, cooldown,
    margin health); every gate and lock is the real code path."""
    tick_b = _short_with_stop(fresh, SYM_B, close_through_stop=True)
    candles_a = _candles()
    if signal == "cover":
        _open_short(fresh, SYM_A, candles_a)
    bot._candles_cache[SYM_A] = candles_a
    monkeypatch.setattr(type(bot.brain), "get_ensemble_signal",
                        lambda self, symbol, candles, **_: _ensemble(symbol, candles, signal))
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: 0.5)
    monkeypatch.setattr(bot, "_symbol_last_trade", {})
    monkeypatch.setattr(bot, "_margin_health_cache", {})
    ex = _slow_exchange(monkeypatch, slow={SYM_A: SLOW_ORDER_SECS})

    async def main():
        await asyncio.gather(
            bot._evaluate_and_trade(SYM_A, candles_a, 0.05, asyncio.get_running_loop()),
            _tick_when_sent(ex, SYM_A, SYM_B, tick_b),
        )

    _run(main())
    assert (fresh.get_short_position(SYM_A) is not None) == (signal == "short")
    assert fresh.get_short_position(SYM_B) is None
    assert _in_order(ex, ("send", SYM_A), ("send", SYM_B), ("booked", SYM_B), ("fill", SYM_A)), (
        f"B's exit waited for A's {signal} order: {ex}")
