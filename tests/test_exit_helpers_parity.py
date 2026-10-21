"""
The pure exit helpers (trail_stop_move, breakeven_stop_move,
designed_decay_moves) were lifted out of bot.py's inline exit code so the
shadow learner can share them. These tests pin them to a verbatim copy of the
pre-refactor inline logic (86ee599) on random inputs, both sides.
"""

import math
import random

import pytest

import bot
from config import (BE_MIN_PROFIT_ATR, BE_TRIGGER_ATR_MULT, DECAY_HALFLIFE_CANDLES,
                    DECAY_MIN_CANDLES, DECAY_SL_TIGHTEN_STRENGTH, DECAY_TP_PULL_STRENGTH)


def _old_trail(is_short, tp, stop, high, low, atr):
    if atr > 0 and not is_short:
        if tp > 0 and high > tp:
            new_stop = high - atr
            if new_stop > stop:
                return new_stop
    elif atr > 0 and is_short:
        if tp > 0 and low < tp:
            new_stop = low + atr
            if new_stop < stop:
                return new_stop
    return None


def _old_be(is_short, avg, stop, high, low, close, atr):
    """(new_stop, deferred) exactly as the old inline ATR branch decided."""
    if not (atr > 0 and avg > 0):
        return None, False
    min_be = BE_MIN_PROFIT_ATR * atr
    if is_short:
        if low <= avg - BE_TRIGGER_ATR_MULT * atr:
            if avg - close >= min_be:
                return (avg if avg < stop else None), False
            return None, True
        return None, False
    if high >= avg + BE_TRIGGER_ATR_MULT * atr:
        if close - avg >= min_be:
            return (avg if avg > stop else None), False
        return None, True
    return None, False


def _old_decay(entry, tp0, sl0, age_min, stop, tp, is_short):
    if entry <= 0 or tp0 <= 0 or sl0 <= 0 or age_min < DECAY_MIN_CANDLES:
        return None, None
    lam = 1.0 - math.exp(-age_min / max(DECAY_HALFLIFE_CANDLES, 1e-6))
    if lam < 0.03:
        return None, None
    target_tp = entry + (tp0 - entry) * (1.0 - lam * DECAY_TP_PULL_STRENGTH)
    target_sl = sl0 + lam * DECAY_SL_TIGHTEN_STRENGTH * (entry - sl0)
    new_tp = new_sl = None
    if is_short:
        if tp < target_tp < entry - 1e-9:
            new_tp = target_tp
        if entry + 1e-9 < target_sl < stop:
            new_sl = target_sl
    else:
        if entry + 1e-9 < target_tp < tp:
            new_tp = target_tp
        if stop < target_sl < entry - 1e-9:
            new_sl = target_sl
    return new_tp, new_sl


def _levels(rng, is_short):
    entry = rng.uniform(0.5, 50_000)
    atr = rng.choice([0.0, -1.0, entry * rng.uniform(1e-4, 0.02)])
    d_stop, d_tp = entry * rng.uniform(1e-3, 0.05), entry * rng.uniform(1e-3, 0.1)
    stop = entry + d_stop if is_short else entry - d_stop
    tp = entry - d_tp if is_short else entry + d_tp
    if rng.random() < 0.1:
        tp = rng.choice([0.0, -1.0])
    move = entry * rng.uniform(-0.12, 0.12)
    high = entry + abs(move) + entry * rng.uniform(0, 0.01)
    low = entry - abs(move) - entry * rng.uniform(0, 0.01)
    close = rng.uniform(low, high)
    return entry, atr, stop, tp, high, low, close


@pytest.mark.parametrize("is_short", [True, False])
def test_trail_and_breakeven_match_the_old_inline_code(is_short):
    rng = random.Random(1234 + is_short)
    hits = {"trail": 0, "be": 0, "deferred": 0}
    for _ in range(20_000):
        entry, atr, stop, tp, high, low, close = _levels(rng, is_short)
        old_t = _old_trail(is_short, tp, stop, high, low, atr)
        assert bot.trail_stop_move(is_short, tp, stop, high, low, atr) == old_t
        old_b = _old_be(is_short, entry, stop, high, low, close, atr)
        assert bot.breakeven_stop_move(is_short, entry, stop, high, low, close, atr) == old_b
        hits["trail"] += old_t is not None
        hits["be"] += old_b[0] is not None
        hits["deferred"] += old_b[1]
    assert all(v > 50 for v in hits.values()), hits          # every branch exercised


@pytest.mark.parametrize("is_short", [True, False])
def test_designed_decay_matches_the_old_inline_code(is_short):
    rng = random.Random(99 + is_short)
    moved = 0
    for _ in range(20_000):
        entry, _, stop, tp, *_ = _levels(rng, is_short)
        tp0 = tp if tp > 0 else entry * (0.97 if is_short else 1.03)
        sl0 = entry * (1.0055 if is_short else 0.9945)
        if rng.random() < 0.5:                                # trail/BE already moved it
            stop = rng.uniform(min(entry, stop), max(entry, stop))
        age = rng.choice([0, 10, 29.9, 30, 31, 60, 240, 1000, 5000]) * rng.uniform(0.9, 1.1)
        old = _old_decay(entry, tp0, sl0, age, stop, tp, is_short)
        assert bot.designed_decay_moves(entry, tp0, sl0, age, stop, tp, is_short) == old
        moved += old != (None, None)
    assert moved > 1000
