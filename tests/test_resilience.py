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
