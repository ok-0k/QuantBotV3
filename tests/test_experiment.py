"""
Symbol-split experiment decay_v2: arm assignment, per-arm slot caps, the
as-designed time decay (arm B) vs the untouched legacy path (arm A), and the
read-only report script.
"""

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
from config import (DECAY_HALFLIFE_CANDLES, DECAY_SL_TIGHTEN_STRENGTH,
                    DECAY_TP_PULL_STRENGTH, STARTING_CASH, SYMBOLS)
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
A_SYMS = [s for i, s in enumerate(SYMBOLS) if i % 2 == 0]
B_SYMS = [s for i, s in enumerate(SYMBOLS) if i % 2 == 1]


@pytest.fixture()
def fresh(clean_db):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    return clean_db


def _ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _place_short(db, symbol, entry=100.0, sl0=100.55, tp0=97.4, arm="B", age_min=60, count=0):
    feats = {"experiment": "decay_v2", "arm": arm, "initial_stop": sl0, "initial_tp": tp0}
    db.open_short(symbol, 1.0, entry, "QF", sl0, tp0, 20.0, entry_features=feats)
    with db.get_db() as conn:
        conn.execute("UPDATE positions SET opened_ts=?, candle_count=? WHERE symbol=?",
                     (_ago(age_min), count, symbol))
    return db.get_short_position(symbol)


# ── Arms ─────────────────────────────────────────────────────────────────────

def test_arms_alternate_balanced_and_deterministic():
    arms = [bot._experiment_arm(s) for s in SYMBOLS]
    assert abs(arms.count("A") - arms.count("B")) <= 1
    assert bot._experiment_arm("BTCUSDT") != bot._experiment_arm("ETHUSDT")
    assert [bot._experiment_arm(s) for s in SYMBOLS] == arms


def test_experiment_off_means_no_arm(monkeypatch):
    monkeypatch.setattr(bot, "EXPERIMENT_NAME", "")
    assert bot._experiment_arm("BTCUSDT") is None


def test_entry_snapshot_records_arm(fresh):
    sym = B_SYMS[0]
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    ens = {"signal": "short", "symbol": sym, "price": candles[-1]["close"], "regime": "ranging",
           "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
           "ml_prob": 0.05, "breakdown": [], "time": "2026-10-06T00:00:00Z", "meta": {}}
    bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
    f = json.loads(fresh.get_short_position(sym)["entry_features"])
    assert f["experiment"] == "decay_v2" and f["arm"] == "B"


# ── Per-arm slot caps ────────────────────────────────────────────────────────

def test_short_slots_split_per_arm(fresh, monkeypatch):
    for s in A_SYMS[:3]:
        _place_short(fresh, s, arm="A")
    assert bot._short_slots_full(A_SYMS[3]) is True        # arm A has its 3
    assert bot._short_slots_full(B_SYMS[0]) is False       # arm B untouched
    monkeypatch.setattr(bot, "EXPERIMENT_NAME", "")
    assert bot._short_slots_full(A_SYMS[3]) is False       # global cap 5, 3 open


# ── Decay: arm B as designed ─────────────────────────────────────────────────

def _expected(age_min, entry=100.0, sl0=100.55, tp0=97.4):
    lam = 1 - math.exp(-age_min / DECAY_HALFLIFE_CANDLES)
    return (entry + (tp0 - entry) * (1 - lam * DECAY_TP_PULL_STRENGTH),
            sl0 + lam * DECAY_SL_TIGHTEN_STRENGTH * (entry - sl0))


def test_designed_decay_matches_formula_and_is_idempotent(fresh):
    sym = B_SYMS[0]
    for _ in range(200):          # ~7 minutes of websocket updates
        bot._apply_ml_time_decay(sym, fresh.get_short_position(sym) or _place_short(fresh, sym), [])
    p = fresh.get_short_position(sym)
    tp, sl = _expected(60)
    assert p["tp_price"] == pytest.approx(tp, rel=1e-6)
    assert p["stop_price"] == pytest.approx(sl, rel=1e-6)
    assert 97.4 < p["tp_price"] < 97.6        # barely moved after an hour: no collapse


def test_designed_decay_waits_min_age(fresh):
    sym = B_SYMS[0]
    _place_short(fresh, sym, age_min=10)
    bot._apply_ml_time_decay(sym, fresh.get_short_position(sym), [])
    p = fresh.get_short_position(sym)
    assert (p["tp_price"], p["stop_price"]) == (97.4, 100.55)


def test_designed_decay_never_loosens_a_tighter_stop(fresh):
    sym = B_SYMS[0]
    _place_short(fresh, sym, age_min=120)
    fresh.update_stop_price(sym, 100.10)          # trail / BE already tighter
    bot._apply_ml_time_decay(sym, fresh.get_short_position(sym), [])
    assert fresh.get_short_position(sym)["stop_price"] == 100.10


# ── Decay: arm A and pre-experiment positions keep legacy behaviour ──────────

@pytest.mark.parametrize("arm", ["A", None])
def test_legacy_decay_unchanged_for_arm_a_and_old_positions(fresh, arm):
    sym = A_SYMS[0]
    _place_short(fresh, sym, arm=arm or "A", count=60)
    if arm is None:
        with fresh.get_db() as conn:
            conn.execute("UPDATE positions SET entry_features=NULL WHERE symbol=?", (sym,))
    lam = 1 - math.exp(-60 / DECAY_HALFLIFE_CANDLES)
    for _ in range(2):
        bot._apply_ml_time_decay(sym, fresh.get_short_position(sym), [])
    # legacy compounds on the current level: two calls = factor applied twice
    expected_tp = 100.0 - 2.6 * (1 - lam * DECAY_TP_PULL_STRENGTH) ** 2
    assert fresh.get_short_position(sym)["tp_price"] == pytest.approx(expected_tp, rel=1e-6)


# ── Report script ────────────────────────────────────────────────────────────

def test_experiment_report_runs(fresh, capsys, monkeypatch):
    import importlib.util
    from pathlib import Path
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    for sym in (A_SYMS[0], B_SYMS[0]):
        ens = {"signal": "short", "symbol": sym, "price": candles[-1]["close"],
               "regime": "ranging", "on_fire": False, "position_size_mult": 1.0,
               "buy_weight": 0.0, "sell_weight": 0.6, "ml_prob": 0.05, "breakdown": [],
               "time": "2026-10-06T00:00:00Z", "meta": {}}
        bot._execute_trade(ens, "QF", candles, 0.5, ZERO_STATE, STARTING_CASH)
        cov = dict(ens, signal="cover", price=candles[-1]["close"] * 0.99,
                   exit_reason="TAKE_PROFIT", exit_detail="t")
        bot._execute_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)

    path = Path(__file__).resolve().parent.parent / "scripts" / "experiment_report.py"
    spec = importlib.util.spec_from_file_location("experiment_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["experiment_report.py", "decay_v2"])
    mod.main()
    out = capsys.readouterr().out
    assert "Arm A" in out and "Arm B" in out and "closed=1" in out
