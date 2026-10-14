"""
Evidence gate (config.ENTRY_EVIDENCE_GATE, off by default): with it on, a new
entry needs a strategy family in VALIDATED_STRATEGIES; refused signals are
journaled as skipped rows (reason "no_validated_edge", throttled per symbol)
and exits are never gated. With it off, behaviour is unchanged.
"""

import asyncio
import json
import os

import numpy as np
import pytest

import bot
from config import STARTING_CASH, SYMBOLS
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
SYM = SYMBOLS[0]


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    monkeypatch.setattr(bot, "_symbol_last_trade", {})
    monkeypatch.setattr(bot, "_entry_block_journaled", {})
    monkeypatch.setattr(bot, "_entry_in_flight", {})
    return clean_db


def _short_ens(symbol, price, strategy="QF_RA_S7_M87"):
    return {"signal": "short", "symbol": symbol, "price": price, "regime": "ranging",
            "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
            "ml_prob": 0.05, "breakdown": [{"signal": "sell", "strategy": strategy, "alloc": 0.7}],
            "time": "2026-10-14T00:00:00Z", "meta": {"short_score": 0.7}}


@pytest.fixture()
def harness(fresh, monkeypatch):
    """Drive the real _evaluate_and_trade with a qualifying short signal."""
    async def _no_margin(*a, **k):
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", _no_margin)
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: 0.5)

    def evaluate(symbol, strategy="QF_RA_S7_M87"):
        candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
        monkeypatch.setattr(bot.brain, "get_ensemble_signal",
                            lambda *a, **k: _short_ens(symbol, candles[-1]["close"], strategy))

        async def run():
            await bot._evaluate_and_trade(symbol, candles, 0.05, asyncio.get_running_loop())
        asyncio.run(run())
    return evaluate


def _rows(db):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY id")]


@pytest.mark.skipif(bool(os.getenv("ENTRY_EVIDENCE_GATE")), reason="env overrides the default")
def test_gate_off_by_default():
    import config
    assert config.ENTRY_EVIDENCE_GATE is False
    assert config.VALIDATED_STRATEGIES == frozenset()


def test_gate_off_behaviour_unchanged(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", False)
    harness(SYM)
    rows = _rows(fresh)
    assert fresh.get_short_position(SYM) is not None
    assert [(r["action"], r["status"]) for r in rows] == [("short", "filled")]


def test_gate_on_refuses_unvalidated_entry_and_journals_it(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", True)
    harness(SYM)
    assert fresh.get_short_position(SYM) is None
    assert fresh.get_all_positions() == []
    rows = _rows(fresh)
    assert [(r["action"], r["status"], r["reason"]) for r in rows] == [("short", "skipped", "no_validated_edge")]
    feats = json.loads(rows[0]["entry_features"])
    assert feats["strategy"] == "QF_RA_S7_M87" and feats["ml_tier"] == 95


def test_gate_on_journal_is_throttled_per_symbol(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", True)
    harness(SYM)
    bot._symbol_last_trade.clear()                 # re-evaluate inside the 300 s window
    harness(SYM)
    assert [r["reason"] for r in _rows(fresh)] == ["no_validated_edge"]


def test_gate_on_allows_a_validated_strategy(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", True)
    monkeypatch.setattr(bot, "VALIDATED_STRATEGIES", frozenset({"QF_RA_S7_M87"}))
    harness(SYM)
    assert fresh.get_short_position(SYM) is not None
    assert [r["status"] for r in _rows(fresh)] == ["filled"]


def test_gate_on_refuses_an_untested_variant_of_a_validated_strategy(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", True)
    monkeypatch.setattr(bot, "VALIDATED_STRATEGIES", frozenset({"QF_RA_S7_M87"}))
    harness(SYM, strategy="QF_RA_S7_M90")
    assert fresh.get_short_position(SYM) is None
    assert [r["reason"] for r in _rows(fresh)] == ["no_validated_edge"]


def test_gate_on_never_blocks_covers(fresh, monkeypatch):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    px = candles[-1]["close"]
    bot._execute_trade(_short_ens(SYM, px), "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", True)
    cov = dict(_short_ens(SYM, px * 0.99), signal="cover")
    bot._execute_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert [(r["action"], r["status"]) for r in _rows(fresh)] == [("short", "filled"), ("cover", "filled")]


@pytest.mark.parametrize("name, validated, ok", [
    ("QF_RA_S7_M87", {"QF_RA_S7_M87"}, True),
    ("QF_RA_S7_M87", {"QF"}, False),                 # no family / prefix matching
    ("QF_RA_S7_M87", {"QF_RA_S7"}, False),
    ("QF", {"QF_RA_S7_M87"}, False),
    ("QF_RA_S7_M87", set(), False),
    (None, {"QF_RA_S7_M87"}, False),
])
def test_validated_exact_name_matching(monkeypatch, name, validated, ok):
    monkeypatch.setattr(bot, "VALIDATED_STRATEGIES", frozenset(validated))
    bd = [{"signal": "sell", "strategy": name, "alloc": 0.7}] if name else []
    assert bot._has_validated_edge({"breakdown": bd}, "short") is ok


# ── visibility: boot report and dashboard tile ───────────────────────────────

@pytest.mark.parametrize("on, validated", [(True, set()), (True, {"QF_RA_S7_M87"}), (False, set())])
def test_boot_report_publishes_gate_state(fresh, monkeypatch, caplog, on, validated):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", on)
    monkeypatch.setattr(bot, "VALIDATED_STRATEGIES", frozenset(validated))
    with caplog.at_level("INFO"):
        bot._report_evidence_gate()
    assert fresh.get_portfolio_stat("evidence_gate") == ("1" if on else "0")
    assert fresh.get_portfolio_stat("evidence_gate_validated") == str(len(validated))
    warned = [r for r in caplog.records if "Evidence gate ON, 0 validated" in r.message]
    assert len(warned) == (1 if on and not validated else 0)
    assert any("Evidence gate ON" in r.message for r in caplog.records) is on


def test_dashboard_health_reads_the_bots_gate_state(fresh):
    import dashboard
    assert dashboard._build_telemetry()["health"]["evidence_gate"] is None    # bot never reported
    fresh.set_portfolio_stat("evidence_gate", "1")
    fresh.set_portfolio_stat("evidence_gate_validated", 0)
    h = dashboard._build_telemetry()["health"]
    assert h["evidence_gate"] is True and h["evidence_gate_validated"] == 0
