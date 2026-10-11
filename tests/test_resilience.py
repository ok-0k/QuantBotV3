"""
Runtime resilience: the supervisor fails fast when a core loop dies, a second
bot instance cannot start, the websocket handler creates a missing candle cache,
heartbeats are written, and model ages are published.
"""

import asyncio
import json
import os
import time

import pytest

import bot
from config import DATA_DIR, SYMBOLS


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    monkeypatch.setattr(bot, "_shutdown_event", asyncio.Event())
    monkeypatch.setattr(bot, "_stop_requested", False)
    return clean_db


def test_supervise_fails_fast_when_a_core_loop_dies(fresh, caplog):
    async def run():
        async def healthy():
            while not bot._shutdown_event.is_set():
                await asyncio.sleep(0.01)

        async def dies():
            await asyncio.sleep(0.05)
            raise RuntimeError("sqlite exploded")

        return await asyncio.wait_for(bot._supervise({"healthy": healthy(), "exit_monitor": dies()}), 10)

    with caplog.at_level("CRITICAL"):
        code = asyncio.run(run())
    assert code == 1
    assert bot._shutdown_event.is_set()                       # siblings told to stop
    assert any("exit_monitor stopped unexpectedly" in r.message for r in caplog.records)


def test_supervise_returns_zero_on_graceful_shutdown(fresh):
    async def run():
        async def loop_():
            while not bot._shutdown_event.is_set():
                await asyncio.sleep(0.01)

        async def stopper():
            await asyncio.sleep(0.05)
            bot._handle_signal("SIGTERM")
            await loop_()

        return await asyncio.wait_for(bot._supervise({"a": loop_(), "b": stopper()}), 10)

    assert asyncio.run(run()) == 0


def test_second_instance_cannot_start(fresh):
    first = bot._acquire_instance_lock()
    try:
        with pytest.raises(SystemExit):
            bot._acquire_instance_lock()
    finally:
        first.close()
    bot._acquire_instance_lock().close()                     # free again once released


def _kline(sym, t, close, closed=False):
    return json.dumps({"data": {"s": sym, "k": {
        "t": t, "o": close, "h": close, "l": close, "c": close, "v": 1.0, "x": closed}}})


def test_ws_handler_creates_missing_cache_and_writes_throttled_heartbeat(fresh, monkeypatch):
    sym = SYMBOLS[5]
    monkeypatch.setattr(bot, "_candles_cache", {})            # startup fetch "failed"
    monkeypatch.setattr(bot, "_ws_heartbeat_last", 0.0)
    writes = []
    real = bot.set_portfolio_stat
    monkeypatch.setattr(bot, "set_portfolio_stat",
                        lambda k, v: (writes.append(k), real(k, v))[1])

    async def run():
        loop = asyncio.get_running_loop()
        await bot._handle_ws_message(_kline(sym, 60_000, 1.0), loop)
        await bot._handle_ws_message(_kline(sym, 60_000, 1.1), loop)    # same bar, updated
        await bot._handle_ws_message(_kline(sym, 120_000, 1.2), loop)   # next bar
        await asyncio.sleep(0.05)                                       # let tick tasks finish
    asyncio.run(run())
    cache = bot._candles_cache[sym]
    assert [c["time"] for c in cache] == [60_000, 120_000] and cache[0]["close"] == 1.1
    assert writes.count("ws_last_msg") == 1                   # throttled to once per 10 s
    assert fresh.get_portfolio_stat("ws_last_msg")


def test_model_ages_published(fresh):
    import config
    for path in (config.XGB_MODEL, config.SAC_WEIGHTS):
        path.write_text("x")
        old = time.time() - 3 * 86400
        os.utime(path, (old, old))
    bot._report_model_ages()
    assert float(fresh.get_portfolio_stat("model_age_xgb_days")) == pytest.approx(3.0, abs=0.1)
    assert float(fresh.get_portfolio_stat("model_age_sac_days")) == pytest.approx(3.0, abs=0.1)
    for path in (config.XGB_MODEL, config.SAC_WEIGHTS):
        path.unlink()
    assert str(DATA_DIR).startswith("/tmp")                   # sandboxed by conftest


def test_crash_loop_backoff_sleeps_only_on_repeat(fresh, monkeypatch):
    slept = []
    monkeypatch.setattr(bot.time, "sleep", lambda s: slept.append(s))
    marker = DATA_DIR / "last_unexpected_exit"
    marker.unlink(missing_ok=True)
    bot._crash_loop_backoff()                 # first unexpected exit: no pause
    assert slept == []
    bot._crash_loop_backoff()                 # again within 15 min: pause
    assert slept == [bot._CRASH_LOOP_SLEEP_SECS]
    marker.write_text(str(time.time() - 3600))
    bot._crash_loop_backoff()                 # an hour later: no pause
    assert slept == [bot._CRASH_LOOP_SLEEP_SECS]


def test_exit_monitor_isolates_a_failing_position_and_writes_heartbeat(fresh, monkeypatch):
    """One position raising inside its checks must not stop the other
    positions' exits or the cycle's equity snapshot / heartbeat."""
    import numpy as np
    from config import STARTING_CASH
    from conftest import make_candles
    sym_bad, sym_ok = SYMBOLS[0], SYMBOLS[2]
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    for sym in (sym_bad, sym_ok):
        ens = {"signal": "short", "symbol": sym, "price": candles[-1]["close"], "regime": "ranging",
               "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
               "ml_prob": 0.05, "breakdown": [], "time": "2026-10-11T00:00:00Z", "meta": {}}
        bot._execute_trade(ens, "QF", candles, 0.5, np.zeros(13, dtype=np.float32), STARTING_CASH)
    ok = fresh.get_short_position(sym_ok)
    stop = ok["avg_cost"] * 1.004
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET stop_price=? WHERE symbol=?", (stop, sym_ok))
    price = stop * 1.001
    last = dict(candles[-1], close=price, high=price, low=min(price, candles[-1]["low"]))
    bot._candles_cache.clear()
    bot._candles_cache.update({sym_bad: candles, sym_ok: candles[:-1] + [last]})

    real_decay = bot._apply_ml_time_decay

    def decay(sym, pos, c):
        if sym == sym_bad:
            raise RuntimeError("database is locked")
        return real_decay(sym, pos, c)
    monkeypatch.setattr(bot, "_apply_ml_time_decay", decay)

    async def run():
        async def no_margin(*a, **k):
            return {}
        monkeypatch.setattr(bot, "_refresh_margin_health", no_margin)
        monkeypatch.setattr(bot, "CHECK_EVERY_SECS", 0.01)
        real_record = bot.record_equity
        monkeypatch.setattr(bot, "record_equity", lambda eq: (real_record(eq), bot._shutdown_event.set()))
        await asyncio.wait_for(bot.exit_monitor(asyncio.get_running_loop()), 10)
    asyncio.run(run())
    assert fresh.get_short_position(sym_ok) is None              # the healthy position still exited
    assert fresh.get_short_position(sym_bad) is not None         # the failing one is retried next cycle
    assert fresh.get_portfolio_stat("exit_monitor_last_cycle")


def _entry(monkeypatch, main):
    pauses = []
    monkeypatch.setattr(bot, "main", main)
    monkeypatch.setattr(bot, "_crash_loop_backoff", lambda: pauses.append(1))
    return pauses


def test_run_pauses_when_main_raises(fresh, monkeypatch):
    async def boom():
        raise RuntimeError("startup failed")         # e.g. initial_train OOM on every boot
    pauses = _entry(monkeypatch, boom)
    assert bot._run() == 1 and pauses == [1]


def test_run_pauses_on_subsystem_failure_but_not_after_a_stop(fresh, monkeypatch):
    async def died():
        return 1
    pauses = _entry(monkeypatch, died)
    assert bot._run() == 1 and pauses == [1]
    bot._handle_signal("SIGTERM")                    # stop arrived during cleanup
    assert bot._run() == 1 and pauses == [1]         # no pause delaying systemctl stop


def test_run_clean_exit_and_lock_refusal_never_pause(fresh, monkeypatch):
    async def ok():
        return 0
    pauses = _entry(monkeypatch, ok)
    assert bot._run() == 0

    async def locked():
        raise SystemExit(1)
    monkeypatch.setattr(bot, "main", locked)
    with pytest.raises(SystemExit):
        bot._run()
    assert pauses == []
