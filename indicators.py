"""
indicators.py — Shared technical-indicator math used by both bot.py (the
live event loop, which works with dict-based OHLCV candles) and brain.py
(the vectorized numpy signal engine).

Wilder's ATR previously had two independent implementations, one per
module. They were carefully re-aligned by hand once already (see bot.py's
Fix #9 changelog) after drifting apart and producing inconsistent
trailing/break-even stop distances between the tick-level exit checker and
the signal engine. This module is the single source of truth going forward
so that class of bug can't recur.
"""

from __future__ import annotations

import numpy as np


def wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    """
    Average True Range (Wilder smoothing) from pre-extracted OHLC arrays.

    Requires at least `period + 1` bars — returns 0.0 otherwise. Seeds from
    the mean of the first `period` true ranges, then applies Wilder's
    exponential smoothing: ATR = (prev * (period-1) + TR) / period.
    """
    if len(c) < period + 1:
        return 0.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    seed = tr[:period].mean()
    atr = seed
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return float(atr)


def wilder_atr_from_candles(candles: list[dict], period: int = 14) -> float:
    """
    Average True Range (Wilder smoothing) from a list of OHLCV candle dicts
    (bot.py's live candle-cache format: keys open/high/low/close/volume).

    Extracts arrays and delegates to wilder_atr() so both call styles share
    one computation. Note: this requires the same `period + 1` bars as
    wilder_atr() above — bot.py's original standalone implementation
    tolerated a partial seed window (as few as 2 candles) for its tick-level
    exit checker specifically. That tolerance is not preserved here; the
    narrow practical effect is that _tick_exit_check's trailing-stop /
    break-even ratchet (never the hard stop-loss, take-profit, or survival
    kill-switch, none of which depend on ATR) stays inactive for the first
    `period` candles after a restart, for a symbol whose REST candle-history
    backfill happened to fail at startup. See bot.py's O1 changelog entry.
    """
    if len(candles) < period + 1:
        return 0.0
    h = np.array([bar["high"] for bar in candles], dtype=float)
    l = np.array([bar["low"] for bar in candles], dtype=float)
    c = np.array([bar["close"] for bar in candles], dtype=float)
    return wilder_atr(h, l, c, period=period)
