"""
Entry gates: the drawdown circuit breaker blocks shorts (it used to gate only
'buy', which this short-only bot never emits), blocked entries are journaled,
the reset script, and the V3 in-flight slot reservation.
"""

import asyncio
import json

import numpy as np
import pytest

import bot
from config import STARTING_CASH, SYMBOLS
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
A_SYMS = [s for i, s in enumerate(SYMBOLS) if i % 2 == 0]


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    monkeypatch.setattr(bot, "_symbol_last_trade", {})
    monkeypatch.setattr(bot, "_entry_block_journaled", {})
    monkeypatch.setattr(bot, "_entry_in_flight", {})
    return clean_db


def _short_ens(symbol, price):
    return {"signal": "short", "symbol": symbol, "price": price, "regime": "ranging",
            "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
            "ml_prob": 0.05, "breakdown": [{"signal": "sell", "strategy": "QF", "alloc": 0.7}],
            "time": "2026-10-11T00:00:00Z", "meta": {"short_score": 0.7}}


@pytest.fixture()
def harness(fresh, monkeypatch):
    """Drive the real _evaluate_and_trade with a qualifying short signal."""
    async def _no_margin(*a, **k):
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", _no_margin)
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: 0.5)

    def evaluate(symbol):
        candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
        monkeypatch.setattr(bot.brain, "get_ensemble_signal",
                            lambda *a, **k: _short_ens(symbol, candles[-1]["close"]))

        async def run():
            await bot._evaluate_and_trade(symbol, candles, 0.05, asyncio.get_running_loop())
        asyncio.run(run())
    return evaluate


def _rows(db, status, action="short"):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE status=? AND action=? ORDER BY id", (status, action))]


# ── circuit breaker ──────────────────────────────────────────────────────────

def test_harness_opens_a_short_when_breaker_closed(harness, fresh):
    harness(A_SYMS[0])
    assert fresh.get_short_position(A_SYMS[0]) is not None


def test_breaker_blocks_shorts_in_evaluate_and_trade(harness, fresh):
    bot.brain.circuit_open = True
    harness(A_SYMS[0])
    assert fresh.get_short_position(A_SYMS[0]) is None
    skipped = _rows(fresh, "skipped")
    assert [r["reason"] for r in skipped] == ["circuit_breaker"]
    assert json.loads(skipped[0]["entry_features"])["ml_tier"] == 95


def test_breaker_backstop_in_prepare_trade(fresh):
    bot.brain.circuit_open = True
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    bot._execute_trade(_short_ens("AAAUSDT", candles[-1]["close"]), "QF", candles, 0.5,
                       ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert _rows(fresh, "skipped")[-1]["reason"] == "circuit_breaker"


def test_breaker_never_blocks_covers(fresh):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    px = candles[-1]["close"]
    bot._execute_trade(_short_ens("AAAUSDT", px), "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    bot.brain.circuit_open = True
    cov = dict(_short_ens("AAAUSDT", px * 0.99), signal="cover")
    bot._execute_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert len(_rows(fresh, "filled", "cover")) == 1


def test_breaker_trip_warns_once(fresh, caplog, monkeypatch):
    monkeypatch.setattr(bot, "_cb_last_open", None)
    bot.brain.circuit_open = True
    with caplog.at_level("WARNING"):
        bot._note_circuit_breaker_state()
        bot._note_circuit_breaker_state()
    assert sum("CIRCUIT BREAKER OPEN" in r.message for r in caplog.records) == 1


# ── journaling blocked entries ───────────────────────────────────────────────

def test_slots_full_block_journaled_once_per_window(harness, fresh, monkeypatch, decay_v2_on):
    for s in A_SYMS[1:4]:                                   # fill arm A's 3 slots
        fresh.open_short(s, 1.0, 100.0, "QF", 101.0, 97.0, 20.0)
    harness(A_SYMS[0])
    bot._symbol_last_trade.clear()                          # re-evaluate inside 300 s
    harness(A_SYMS[0])
    rows = _rows(fresh, "skipped")
    assert [r["reason"] for r in rows] == ["slots_full"]
    assert json.loads(rows[0]["entry_features"])["arm"] == "A"


# ── reset script ─────────────────────────────────────────────────────────────

def test_reset_script_dry_run_then_confirm(fresh, monkeypatch):
    import importlib.util
    from pathlib import Path

    import config
    path = Path(__file__).resolve().parent.parent / "scripts" / "reset_circuit_breaker.py"
    spec = importlib.util.spec_from_file_location("reset_cb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh.save_brain_key(mod.PEAK_KEY, 10000.0)
    fresh.record_equity(7800.0)
    assert mod.main([]) == 0
    assert fresh.load_brain_key(mod.PEAK_KEY) == 10000.0          # dry run writes nothing
    assert mod.main(["--confirm"]) == 0
    assert fresh.load_brain_key(mod.PEAK_KEY) == 7800.0
    assert list((config.DATA_DIR / "backups").glob("circuit-breaker-pre-*.json"))


# ── V3 in-flight slot reservation ────────────────────────────────────────────

def test_v3_in_flight_entries_count_against_the_arm_cap(fresh, monkeypatch, decay_v2_on):
    for s in A_SYMS[1:3]:
        fresh.open_short(s, 1.0, 100.0, "QF", 101.0, 97.0, 20.0)
    bot._entry_in_flight[A_SYMS[3]] = {"action": "short", "arm": "A", "qty": 1.0, "price": 100.0}
    monkeypatch.setattr(bot, "V3_FIXES", False)
    assert bot._short_slots_full(A_SYMS[0]) is False              # legacy: in-flight ignored
    monkeypatch.setattr(bot, "V3_FIXES", True)
    assert bot._short_slots_full(A_SYMS[0]) is True               # 2 booked + 1 in flight = 3


def test_v3_reservation_released_when_the_exchange_call_raises(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "V3_FIXES", True)

    def boom(intent):
        assert intent.symbol in bot._entry_in_flight              # reserved during the call
        raise RuntimeError("exchange down")
    monkeypatch.setattr(bot._exec_adapter, "execute", boom)
    with pytest.raises(RuntimeError):
        harness(A_SYMS[0])
    assert bot._entry_in_flight == {}
