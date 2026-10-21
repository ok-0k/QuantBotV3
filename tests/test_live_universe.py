"""
The live universe without the 4 pairs delisted from Binance spot: the
websocket streams, the startup/backfill history fetch and the tick handler
skip them, and an open position on a symbol outside SYMBOLS (no price feed,
so no exits) is reported at boot.
"""

import asyncio
import inspect
import json

import pytest

import bot
import insights
from config import INTERVAL, SYMBOLS
from research.universe import RESEARCH_SYMBOLS

DELISTED = {"MATICUSDT", "MKRUSDT", "OCEANUSDT", "AGIXUSDT"}


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    monkeypatch.setattr(bot, "_shutdown_event", asyncio.Event())
    return clean_db


def test_symbols_are_the_research_universe_minus_the_delisted_pairs():
    assert insights.DELISTED_SYMBOLS == DELISTED
    assert SYMBOLS == [s for s in RESEARCH_SYMBOLS if s not in DELISTED]
    assert len(SYMBOLS) == len(set(SYMBOLS)) == 46


def test_ws_streams_and_history_fetch_skip_the_delisted_pairs(monkeypatch):
    streams = bot._ws_url().split("?streams=", 1)[1].split("/")
    assert streams == [f"{s.lower()}@kline_{INTERVAL}" for s in SYMBOLS]
    fetched = []

    async def fake_fetch(session, sym):
        fetched.append(sym)
        return []

    monkeypatch.setattr(bot, "fetch_candle_history", fake_fetch)
    assert asyncio.run(bot._load_history_for_all_symbols()) == {}
    assert fetched == SYMBOLS and not DELISTED & set(fetched)


def test_ws_handler_ignores_a_delisted_symbol(fresh, monkeypatch):
    monkeypatch.setattr(bot, "_candles_cache", {})
    msg = json.dumps({"data": {"s": "MATICUSDT", "k": {
        "t": 60_000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0, "x": True}}})

    async def run():
        await bot._handle_ws_message(msg, asyncio.get_running_loop())
    asyncio.run(run())
    assert "MATICUSDT" not in bot._candles_cache


def test_boot_reports_open_positions_outside_symbols(fresh, caplog):
    fresh.open_short("MATICUSDT", 10.0, 0.5, "QF", 0.52, 0.45, 1.0)
    fresh.open_short(SYMBOLS[0], 0.001, 60_000.0, "QF", 60_500.0, 58_000.0, 12.0)
    with caplog.at_level("ERROR"):
        assert bot._report_unfed_positions() == ["MATICUSDT"]
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "MATICUSDT" in errors[0] and "no price feed" in errors[0]
    assert "_report_unfed_positions()" in inspect.getsource(bot.main)      # wired into boot


def test_boot_report_is_silent_without_such_positions(fresh, caplog):
    fresh.open_short(SYMBOLS[0], 0.001, 60_000.0, "QF", 60_500.0, 58_000.0, 12.0)
    with caplog.at_level("ERROR"):
        assert bot._report_unfed_positions() == []
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
