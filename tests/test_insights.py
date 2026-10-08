"""
insights.py (read-only journal analytics) and the dashboard's /api/insights
endpoint + Insights sections.
"""

import asyncio
import json

import numpy as np
import pytest

import bot
import insights
from config import STARTING_CASH, SYMBOLS
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
A_SYM, B_SYM = SYMBOLS[0], SYMBOLS[1]          # alternate experiment arms


@pytest.fixture()
def fresh(clean_db):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    return clean_db


def _round_trip(symbol, move, exit_reason):
    """Open a short and cover it `move` (fraction) away, journaled with exit_reason."""
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    px = candles[-1]["close"]
    ens = {"signal": "short", "symbol": symbol, "price": px, "regime": "ranging", "on_fire": False,
           "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6, "ml_prob": 0.05,
           "breakdown": [], "time": "2026-10-08T00:00:00Z", "meta": {}}
    bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    cov = dict(ens, signal="cover", price=px * (1 + move), exit_reason=exit_reason, exit_detail="t")
    bot._execute_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)


def test_empty_journal(fresh):
    snap = insights.snapshot()
    assert snap["week"]["closed"] == 0 and snap["week"]["exit_mix"] == []
    assert snap["lifetime"]["rows_kept"] == 0


def test_journal_window_and_exit_mix(fresh):
    _round_trip(A_SYM, -0.02, "TAKE_PROFIT")       # short wins 2%
    _round_trip(B_SYM, +0.01, "STOP_LOSS")         # short loses 1%
    _round_trip(A_SYM, +0.01, "STOP_LOSS")
    w = insights.snapshot()["week"]
    assert w["closed"] == 3
    assert w["win_rate"] == pytest.approx(100 / 3, abs=0.1)
    mix = {m["reason"]: m for m in w["exit_mix"]}
    assert mix["STOP_LOSS"]["n"] == 2 and mix["TAKE_PROFIT"]["n"] == 1
    assert mix["TAKE_PROFIT"]["net"] > 0 > mix["STOP_LOSS"]["net"]
    assert sum(m["share"] for m in w["exit_mix"]) == pytest.approx(100, abs=0.2)
    assert w["exit_mix"][0]["reason"] == "STOP_LOSS"            # sorted by count
    assert w["cost_pct"] == pytest.approx(0.36)


def test_experiment_arms_and_verdict(fresh):
    for _ in range(2):
        _round_trip(A_SYM, +0.01, "STOP_LOSS")
        _round_trip(B_SYM, -0.01, "TAKE_PROFIT")
    e = insights.snapshot()["experiment"]
    assert e["name"] == "decay_v2"
    assert e["arms"]["A"]["closed"] == 2 and e["arms"]["B"]["closed"] == 2
    assert e["arms"]["B"]["avg_pct"] > e["arms"]["A"]["avg_pct"]
    assert e["diff"] == pytest.approx(e["arms"]["B"]["avg_pct"] - e["arms"]["A"]["avg_pct"], abs=1e-3)
    assert e["verdict"] in ("B better", "not yet distinguishable", "A better")


def test_no_experiment_when_disabled(fresh):
    conn = insights.connect()
    try:
        assert insights.experiment(conn, "") is None
    finally:
        conn.close()


def test_insights_connection_is_read_only(fresh):
    conn = insights.connect()
    try:
        with pytest.raises(Exception):
            conn.execute("DELETE FROM trades")
    finally:
        conn.close()


def test_dashboard_endpoint_caches_and_page_has_sections(fresh, monkeypatch):
    import dashboard
    calls = []
    monkeypatch.setattr(dashboard.insights, "snapshot", lambda: calls.append(1) or {"week": {}, "ok": len(calls)})
    monkeypatch.setitem(dashboard._INSIGHTS_CACHE, "data", None)
    first = json.loads(asyncio.run(dashboard.api_insights()).body)
    second = json.loads(asyncio.run(dashboard.api_insights()).body)
    assert len(calls) == 1 and first == second                  # served from the 60 s cache
    for el in ('id="insights-grid"', 'id="experiment-grid"', 'id="exit-mix"', "function renderInsights"):
        assert el in dashboard.DASHBOARD_HTML
