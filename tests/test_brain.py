"""Characterization tests for brain.py — circuit breaker, stops, sizing, regime."""

import numpy as np
import pytest

from brain import Brain
from conftest import make_candles
from config import (
    CB_RECOVERY_PCT,
    MAX_DRAWDOWN_PCT,
    SAC_STATE_DIM,
)


@pytest.fixture()
def brain():
    return Brain()


# ── Circuit breaker (drawdown halt with hysteresis) ──────────────────────────

def test_circuit_breaker_stays_closed_in_normal_drawdown(brain):
    assert brain.check_circuit_breaker(10_000.0) is False
    assert brain.check_circuit_breaker(9_000.0) is False   # -10% < 20% limit


def test_circuit_breaker_trips_at_max_drawdown(brain):
    brain.check_circuit_breaker(10_000.0)
    assert brain.check_circuit_breaker(8_000.0) is True    # exactly -20%
    assert brain.circuit_open


def test_circuit_breaker_hysteresis_recovery(brain):
    brain.check_circuit_breaker(10_000.0)
    brain.check_circuit_breaker(8_000.0)
    # Needs drawdown <= MAX_DD - RECOVERY = 10% to close again
    assert brain.check_circuit_breaker(8_950.0) is True    # -10.5%: still open
    assert brain.check_circuit_breaker(9_000.0) is False   # -10.0%: closes
    assert MAX_DRAWDOWN_PCT == pytest.approx(0.20)
    assert CB_RECOVERY_PCT == pytest.approx(0.10)


def test_circuit_breaker_peak_ratchets_up(brain):
    brain.check_circuit_breaker(10_000.0)
    brain.check_circuit_breaker(12_000.0)                  # new peak
    assert brain.check_circuit_breaker(9_600.0) is True    # -20% from 12k


def test_peak_equity_persists_and_restores_across_fresh_brain(clean_db):
    """C2: the circuit breaker's peak-equity high-water mark must survive a
    restart (crash, deploy, OOM) instead of silently re-arming MAX_DRAWDOWN_PCT
    from whatever equity exists when the process comes back up."""
    b1 = Brain()
    b1.check_circuit_breaker(10_000.0)
    b1.check_circuit_breaker(12_000.0)          # new peak -> persisted immediately
    assert b1._peak_equity == pytest.approx(12_000.0)

    b2 = Brain()                                # simulates a fresh process after restart
    assert b2._peak_equity == pytest.approx(0.0)   # not auto-restored on construction
    restored = b2.restore_peak_equity(default=0.0)
    assert restored == pytest.approx(12_000.0)
    assert b2._peak_equity == pytest.approx(12_000.0)

    # The restored peak must actually drive circuit-breaker behaviour, not
    # just sit there as a dead value: drawdown should be measured from the
    # persisted historical peak, not from equity-at-restart.
    assert b2.check_circuit_breaker(9_600.0) is True   # -20% from restored 12k peak


def test_restore_peak_equity_never_goes_below_default(clean_db):
    """A missing/never-persisted key must fall back to `default` (boot
    equity), not silently leave the peak at 0.0 (which would make every
    restart look like a 100% drawdown and trip the breaker immediately)."""
    b = Brain()
    restored = b.restore_peak_equity(default=8_500.0)
    assert restored == pytest.approx(8_500.0)
    assert b.check_circuit_breaker(8_500.0) is False


# ── get_stop_take ────────────────────────────────────────────────────────────

def test_stop_take_degenerate_fallback_long(brain):
    # <15 candles -> fixed 0.5% band, TP 1.6x the pad
    stop, take = brain.get_stop_take(100.0, [], False, side="long")
    assert stop == pytest.approx(99.5)
    assert take == pytest.approx(100.8)


def test_stop_take_degenerate_fallback_short(brain):
    stop, take = brain.get_stop_take(100.0, [], False, side="short")
    assert stop == pytest.approx(100.5)
    assert take == pytest.approx(99.2)


def test_stop_take_long_ordering_and_min_rr(brain):
    candles = make_candles(n=60, start_price=100.0, drift=0.05, amplitude=0.4)
    stop, take = brain.get_stop_take(100.0, candles, False, side="long", regime="ranging")
    assert stop < 100.0 < take
    # Floor enforced in multiplier space: tp distance >= 2x sl distance
    assert (take - 100.0) >= 2.0 * (100.0 - stop) - 1e-9


def test_stop_take_short_ordering_and_min_rr(brain):
    candles = make_candles(n=60, start_price=100.0, drift=-0.05, amplitude=0.4)
    stop, take = brain.get_stop_take(100.0, candles, False, side="short", regime="trend_down")
    assert take < 100.0 < stop
    assert (100.0 - take) >= 2.0 * (stop - 100.0) - 1e-9


def test_stop_take_yolo_widens_take_profit(brain):
    candles = make_candles(n=60, start_price=100.0, drift=0.05, amplitude=0.4)
    _, take_normal = brain.get_stop_take(100.0, candles, False, side="long")
    _, take_yolo = brain.get_stop_take(100.0, candles, True, side="long")
    assert take_yolo > take_normal


# ── ml_conviction_size_mult ──────────────────────────────────────────────────

def test_conviction_mult_bounds():
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        for ss in (0.0, 0.5, 1.0):
            m = Brain.ml_conviction_size_mult(p, ss)
            assert 0.35 <= m <= 1.45


def test_conviction_mult_monotone_in_ml_distance():
    lo = Brain.ml_conviction_size_mult(0.60, 0.5)
    hi = Brain.ml_conviction_size_mult(0.95, 0.5)
    assert hi > lo


def test_conviction_mult_neutral_prob_is_minimum_base():
    # p=0.5 -> conv=0 -> base=0.42; struct(0.5)=0.81 -> 0.3402 -> clamped 0.35
    assert Brain.ml_conviction_size_mult(0.5, 0.5) == pytest.approx(0.35)


# ── compute_sac_state ────────────────────────────────────────────────────────

def test_sac_state_shape_and_dtype(brain):
    candles = make_candles(n=60)
    vec = brain.compute_sac_state("BTCUSDT", candles, 5_000.0, 100.0, 10_000.0)
    assert vec.shape == (SAC_STATE_DIM,)
    assert vec.dtype == np.float32


def test_sac_state_short_history_partial_features(brain):
    brain.update_ml_prob("BTCUSDT", 0.9)
    vec = brain.compute_sac_state("BTCUSDT", [], 5_000.0, 0.0, 10_000.0)
    assert vec[0] == pytest.approx(0.9)
    assert vec[1] == pytest.approx(0.5)      # cash / equity
    assert np.all(vec[3:] == 0.0)


def test_update_ml_prob_clamps(brain):
    brain.update_ml_prob("X", 1.7)
    assert brain.ml_probs["X"] == 1.0
    brain.update_ml_prob("X", -0.3)
    assert brain.ml_probs["X"] == 0.0


# ── Adaptive edge learner ────────────────────────────────────────────────────

def test_reward_moves_edge_profile_up_on_wins(brain):
    for _ in range(50):
        brain.reward(pnl=50.0, trade_value=1000.0, regime="trend_down", side="short")
    p = brain._edge_profile("short", "trend_down")
    assert p["size_mult"] > 1.0
    assert p["size_mult"] <= 1.35    # EDGE_SIZE_MAX_MULT clamp
    assert p["rr_mult"] <= 1.45      # EDGE_RR_MAX_MULT clamp


def test_reward_moves_edge_profile_down_on_losses(brain):
    for _ in range(50):
        brain.reward(pnl=-50.0, trade_value=1000.0, regime="ranging", side="long")
    p = brain._edge_profile("long", "ranging")
    assert p["size_mult"] < 1.0
    # EDGE_SIZE_MIN_MULT lowered from 0.65 to 0.05: a persistently bad bucket
    # must be able to size down near zero rather than floor at a still-large
    # 65%, so the existing $10 min-notional veto in bot.py can actually fire.
    assert p["size_mult"] >= 0.05


def test_edge_profile_export_import_roundtrip(brain):
    brain.reward(pnl=25.0, trade_value=1000.0, regime="trend_down", side="short")
    exported = brain.export_edge_profiles()
    fresh = Brain()
    assert fresh.import_edge_profiles(exported) == len(exported)
    assert fresh.export_edge_profiles() == exported


# ── Regime classification ────────────────────────────────────────────────────

def test_regime_low_adx_is_ranging(brain):
    assert brain._classify_regime(15.0, 101.0, 100.0, 102.0) == "ranging"


def test_regime_trend_down(brain):
    assert brain._classify_regime(30.0, 99.0, 100.0, 98.0) == "trend_down"


def test_regime_trend_up(brain):
    assert brain._classify_regime(30.0, 101.0, 100.0, 102.0) == "trend_up"


# ── Ensemble signal (short-only engine) ──────────────────────────────────────

def test_ensemble_never_emits_buy(brain):
    """The live Brain is short-only: longs can exit but never enter from here."""
    candles = make_candles(n=120, start_price=100.0, drift=-0.10, amplitude=0.3)
    for ml_prob in (0.05, 0.5, 0.95):
        out = brain.get_ensemble_signal(
            "AAAUSDT", candles, position_side=None, ml_prob=ml_prob,
        )
        assert out["signal"] in ("short", "none")
        assert out["buy_weight"] == 0.0


def test_ensemble_warmup_blocks(brain):
    out = brain.get_ensemble_signal("AAAUSDT", make_candles(n=30), ml_prob=0.05)
    assert out["signal"] == "none"
    assert out["block_reason"] == "warmup"


def test_ensemble_contains_execution_contract_keys(brain):
    out = brain.get_ensemble_signal("AAAUSDT", make_candles(n=120), ml_prob=0.5)
    for key in ("signal", "symbol", "price", "regime", "ml_prob",
                "position_size_mult", "breakdown", "time"):
        assert key in out
