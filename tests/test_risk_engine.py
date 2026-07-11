"""Characterization tests for risk_engine — sizing, R:R, correlation, exposure."""

import pytest

from conftest import make_candles, make_correlated_walk
from risk_engine import (
    apply_dynamic_rr,
    correlation_blocks_entry,
    dynamic_conviction_size_mult,
    exposure_blocked,
    pearson_rho,
    volatility_rr_multipliers,
)
import numpy as np


# ── dynamic_conviction_size_mult ─────────────────────────────────────────────
# Config anchors: floor_conf=0.80 -> 1.00x, ceil_conf=0.99 -> 2.50x, linear.

def test_size_mult_at_floor_confidence():
    assert dynamic_conviction_size_mult(0.80, "long") == pytest.approx(1.00)


def test_size_mult_at_ceiling_confidence():
    assert dynamic_conviction_size_mult(0.995, "long") == pytest.approx(2.50)


def test_size_mult_midpoint_interpolates_linearly():
    # conf 0.895 -> t = 0.5 -> 1.0 + 0.5 * 1.5 = 1.75
    assert dynamic_conviction_size_mult(0.895, "long") == pytest.approx(1.75)


def test_size_mult_short_side_uses_one_minus_p():
    # p=0.105 -> short conviction 0.895 -> same 1.75
    assert dynamic_conviction_size_mult(0.105, "short") == pytest.approx(1.75)


def test_size_mult_low_confidence_clamps_to_min():
    assert dynamic_conviction_size_mult(0.55, "long") == pytest.approx(1.00)


def test_size_mult_garbage_input_returns_min():
    assert dynamic_conviction_size_mult("nan-ish", "long") == pytest.approx(1.00)


# ── volatility_rr_multipliers ────────────────────────────────────────────────
# Anchors: vol<=0.30% -> (0.85, 1.20); vol>=1.20% -> (1.30, 0.85); linear mid.

def test_rr_multipliers_quiet_tape():
    assert volatility_rr_multipliers(0.3, 100.0) == pytest.approx((0.85, 1.20))


def test_rr_multipliers_hectic_tape():
    assert volatility_rr_multipliers(1.2, 100.0) == pytest.approx((1.30, 0.85))


def test_rr_multipliers_mid_regime_interpolates():
    sl_m, tp_m = volatility_rr_multipliers(0.75, 100.0)  # vol=0.75% -> t=0.5
    assert sl_m == pytest.approx(1.075)
    assert tp_m == pytest.approx(1.025)


def test_rr_multipliers_degenerate_inputs_neutral():
    assert volatility_rr_multipliers(0.0, 100.0) == (1.0, 1.0)
    assert volatility_rr_multipliers(1.0, 0.0) == (1.0, 1.0)


# ── apply_dynamic_rr ─────────────────────────────────────────────────────────

def test_apply_rr_long_quiet_tape():
    # entry 100, stop 99 (dist 1), take 102 (dist 2), atr 0.3 -> quiet
    # sl_dist = 1*0.85 = 0.85 ; tp_dist = 2*1.2 = 2.4 (>= 1.5*0.85 ok)
    stop, take = apply_dynamic_rr(entry=100.0, stop=99.0, take=102.0, atr=0.3, side="long")
    assert stop == pytest.approx(99.15)
    assert take == pytest.approx(102.4)


def test_apply_rr_enforces_min_rr_floor():
    # Hectic tape: sl 1*1.3=1.3 ; tp 0.5*0.85=0.425 < 1.5*1.3=1.95 -> floored
    stop, take = apply_dynamic_rr(entry=100.0, stop=99.0, take=100.5, atr=1.2, side="long")
    assert stop == pytest.approx(98.7)
    assert take == pytest.approx(101.95)


def test_apply_rr_short_mirrors_long():
    stop, take = apply_dynamic_rr(entry=100.0, stop=101.0, take=98.0, atr=0.3, side="short")
    assert stop == pytest.approx(100.85)
    assert take == pytest.approx(97.6)


def test_apply_rr_never_inverts_sides():
    for atr in (0.1, 0.5, 1.0, 2.0):
        s, t = apply_dynamic_rr(entry=100.0, stop=99.0, take=102.0, atr=atr, side="long")
        assert s < 100.0 < t
        s, t = apply_dynamic_rr(entry=100.0, stop=101.0, take=98.0, atr=atr, side="short")
        assert t < 100.0 < s


def test_apply_rr_invalid_inputs_pass_through():
    assert apply_dynamic_rr(entry=0.0, stop=99.0, take=102.0, atr=0.3, side="long") == (99.0, 102.0)


# ── exposure_blocked ─────────────────────────────────────────────────────────

def _pos(symbol, shares, avg_cost, side="long", margin=0.0):
    return {"symbol": symbol, "shares": shares, "avg_cost": avg_cost,
            "side": side, "margin_reserved": margin}


def test_exposure_under_cap_allows():
    positions = [_pos("AAAUSDT", 10.0, 480.0)]
    blocked, cur = exposure_blocked(
        positions, {"AAAUSDT": 500.0}, 10_000.0, 0.30, global_cap=0.82,
    )
    assert cur == pytest.approx(0.50)
    assert not blocked  # 0.50 + 0.30 = 0.80 <= 0.82


def test_exposure_over_cap_blocks():
    positions = [_pos("AAAUSDT", 10.0, 480.0)]
    blocked, cur = exposure_blocked(
        positions, {"AAAUSDT": 500.0}, 10_000.0, 0.33, global_cap=0.82,
    )
    assert blocked  # 0.50 + 0.33 = 0.83 > 0.82


def test_exposure_zero_equity_fails_closed():
    assert exposure_blocked([], {}, 0.0, 0.1, global_cap=0.82) == (True, 1.0)


# ── correlation ──────────────────────────────────────────────────────────────

def test_pearson_rho_needs_12_samples():
    a = np.arange(5, dtype=float)
    assert pearson_rho(a, a) == 0.0


def test_pearson_rho_identical_series_is_one():
    walk = make_correlated_walk(seed=7)
    lr = np.diff(np.log([c["close"] for c in walk]))
    assert pearson_rho(lr, lr) == pytest.approx(1.0)


def test_correlation_blocks_identical_peer():
    walk = make_correlated_walk(seed=7)
    cache = {"PEERUSDT": walk}
    blocked, worst = correlation_blocks_entry(
        walk, ["PEERUSDT"], cache, rho_threshold=0.90, max_high_corr_peers=1,
    )
    assert blocked
    assert worst == pytest.approx(1.0)


def test_correlation_allows_unrelated_peer():
    a = make_correlated_walk(seed=7)
    b = make_correlated_walk(seed=99991)
    blocked, worst = correlation_blocks_entry(
        a, ["PEERUSDT"], {"PEERUSDT": b}, rho_threshold=0.90, max_high_corr_peers=1,
    )
    assert not blocked
    assert worst < 0.90


def test_correlation_flat_new_series_never_blocks():
    flat = make_candles(n=130, drift=0.0, amplitude=0.0)
    blocked, worst = correlation_blocks_entry(
        flat, ["PEERUSDT"], {"PEERUSDT": make_correlated_walk(seed=3)},
        rho_threshold=0.90, max_high_corr_peers=1,
    )
    assert not blocked
