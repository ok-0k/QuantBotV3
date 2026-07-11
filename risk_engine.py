"""
risk_engine.py — Portfolio-level exposure & correlation (V2) +
dynamic conviction sizing & volatility-aware R:R (V3).

Theory
------
1) **Shotgunning**: many small uncorrelated bets diversify; many **correlated** bets act as
   one large bet on a single factor (e.g. beta to BTC). We cap simultaneous exposure
   and block entries highly correlated with existing sleeves.

2) **Correlation**: Pearson ρ on log-return vectors over a rolling window. High |ρ|
   means the new symbol would not diversify idiosyncratic risk.

3) **Dynamic sizing**: position multiplier scales linearly with ML conviction across
   [DYNAMIC_SIZE_FLOOR_CONF, DYNAMIC_SIZE_CEIL_CONF].

4) **Dynamic R:R**: stop/take ATR multipliers scale with current volatility regime
   (ATR/price), tightening stops in chop and widening them in storms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config import (
    DYNAMIC_RR_ENABLED,
    DYNAMIC_RR_HIGHVOL_SL_MULT,
    DYNAMIC_RR_HIGHVOL_TP_MULT,
    DYNAMIC_RR_LOWVOL_SL_MULT,
    DYNAMIC_RR_LOWVOL_TP_MULT,
    DYNAMIC_RR_MIN_RR_RATIO,
    DYNAMIC_RR_VOL_HIGH_PCT,
    DYNAMIC_RR_VOL_LOW_PCT,
    DYNAMIC_SIZE_CEIL_CONF,
    DYNAMIC_SIZE_ENABLED,
    DYNAMIC_SIZE_FLOOR_CONF,
    DYNAMIC_SIZE_MAX_MULT,
    DYNAMIC_SIZE_MIN_MULT,
)


def dynamic_conviction_size_mult(ml_prob: float, side: str) -> float:
    """
    Linearly scale position multiplier by ML conviction.

      side="long"  -> conviction = ml_prob
      side="short" -> conviction = 1 - ml_prob

    conviction <= floor -> MIN_MULT
    conviction >= ceil  -> MAX_MULT
    in-between          -> linear interpolation

    Returned multiplier is never below DYNAMIC_SIZE_MIN_MULT and never above
    DYNAMIC_SIZE_MAX_MULT — extreme values cannot break the SAC ceiling.
    """
    if not DYNAMIC_SIZE_ENABLED:
        return 1.0
    try:
        p = float(ml_prob)
    except (TypeError, ValueError):
        return DYNAMIC_SIZE_MIN_MULT
    p = max(0.0, min(1.0, p))
    conf = (1.0 - p) if str(side).lower() == "short" else p

    lo = float(DYNAMIC_SIZE_FLOOR_CONF)
    hi = float(DYNAMIC_SIZE_CEIL_CONF)
    if hi <= lo:
        return DYNAMIC_SIZE_MIN_MULT
    if conf <= lo:
        return DYNAMIC_SIZE_MIN_MULT
    if conf >= hi:
        return DYNAMIC_SIZE_MAX_MULT
    t = (conf - lo) / (hi - lo)
    return float(
        DYNAMIC_SIZE_MIN_MULT
        + t * (DYNAMIC_SIZE_MAX_MULT - DYNAMIC_SIZE_MIN_MULT)
    )


def volatility_rr_multipliers(atr: float, price: float) -> tuple[float, float]:
    """
    Map current ATR/price to (sl_mult, tp_mult) for entry stop/take adjustment.

    Quiet tape (vol <= LOW)   -> LOWVOL_SL_MULT  (tighter), LOWVOL_TP_MULT  (longer)
    Hectic tape (vol >= HIGH) -> HIGHVOL_SL_MULT (wider),   HIGHVOL_TP_MULT (shorter)
    Mid regime                -> linear interpolation between the two anchors.

    Returns (1.0, 1.0) when DYNAMIC_RR_ENABLED is False or inputs are degenerate.
    """
    if not DYNAMIC_RR_ENABLED:
        return 1.0, 1.0
    try:
        a = float(atr)
        p = float(price)
    except (TypeError, ValueError):
        return 1.0, 1.0
    if a <= 0.0 or p <= 0.0:
        return 1.0, 1.0
    vol = a / p

    lo = float(DYNAMIC_RR_VOL_LOW_PCT)
    hi = float(DYNAMIC_RR_VOL_HIGH_PCT)
    if hi <= lo:
        return 1.0, 1.0

    if vol <= lo:
        sl_m = float(DYNAMIC_RR_LOWVOL_SL_MULT)
        tp_m = float(DYNAMIC_RR_LOWVOL_TP_MULT)
    elif vol >= hi:
        sl_m = float(DYNAMIC_RR_HIGHVOL_SL_MULT)
        tp_m = float(DYNAMIC_RR_HIGHVOL_TP_MULT)
    else:
        t = (vol - lo) / (hi - lo)
        sl_m = float(
            DYNAMIC_RR_LOWVOL_SL_MULT
            + t * (DYNAMIC_RR_HIGHVOL_SL_MULT - DYNAMIC_RR_LOWVOL_SL_MULT)
        )
        tp_m = float(
            DYNAMIC_RR_LOWVOL_TP_MULT
            + t * (DYNAMIC_RR_HIGHVOL_TP_MULT - DYNAMIC_RR_LOWVOL_TP_MULT)
        )
    return sl_m, tp_m


def apply_dynamic_rr(
    entry: float,
    stop: float,
    take: float,
    atr: float,
    side: str,
) -> tuple[float, float]:
    """
    Apply volatility-aware sl/tp multipliers around `entry` and enforce the
    DYNAMIC_RR_MIN_RR_RATIO floor so a degenerate vol reading cannot collapse
    risk-reward. `stop` and `take` are absolute prices from brain.get_stop_take.

    Returns (new_stop, new_take). On invalid inputs, returns the originals.
    """
    try:
        e = float(entry)
        s = float(stop)
        t = float(take)
    except (TypeError, ValueError):
        return stop, take
    if e <= 0.0 or s <= 0.0 or t <= 0.0:
        return stop, take

    sl_m, tp_m = volatility_rr_multipliers(atr, e)
    is_short = str(side).lower() == "short"

    sl_dist = abs(e - s) * sl_m
    tp_dist = abs(t - e) * tp_m

    min_rr = float(DYNAMIC_RR_MIN_RR_RATIO)
    if sl_dist > 0.0 and tp_dist < min_rr * sl_dist:
        tp_dist = min_rr * sl_dist

    if is_short:
        new_stop = e + sl_dist
        new_take = e - tp_dist
        if new_take <= 0.0:
            return stop, take
    else:
        new_stop = e - sl_dist
        new_take = e + tp_dist
        if new_stop <= 0.0:
            return stop, take

    return float(new_stop), float(new_take)


def _log_returns(closes: Sequence[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=np.float64)
    if c.size < 3:
        return np.array([])
    lr = np.diff(np.log(np.clip(c, 1e-12, None)))
    return lr[np.isfinite(lr)]


def pearson_rho(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 12:
        return 0.0
    a = a[-n:]
    b = b[-n:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def correlation_blocks_entry(
    new_closes: list[dict],
    open_symbols: list[str],
    candle_cache: dict[str, list[dict]],
    *,
    rho_threshold: float,
    max_high_corr_peers: int,
) -> tuple[bool, float]:
    """
    Returns (blocked, worst_rho).

    Block when count of open positions with |ρ| ≥ threshold would exceed policy
    (including the proposed new leg).
    """
    lr_new = _log_returns([float(c["close"]) for c in new_closes[-120:]])
    if lr_new.size < 15:
        return False, 0.0

    high = 0
    worst = 0.0
    for sym in open_symbols:
        oc = candle_cache.get(sym)
        if not oc or len(oc) < 30:
            continue
        lr_o = _log_returns([float(c["close"]) for c in oc[-120:]])
        rho = abs(pearson_rho(lr_new, lr_o))
        worst = max(worst, rho)
        if rho >= rho_threshold:
            high += 1

    if high >= max_high_corr_peers:
        return True, worst
    return False, worst


def exposure_blocked(
    positions: list[dict],
    mark_prices: dict[str, float],
    total_equity: float,
    proposed_trade_frac: float,
    *,
    global_cap: float,
) -> tuple[bool, float]:
    """
    Current deployed notional (|shares|×mark) / equity + proposed trade fraction vs cap.

    Uses gross mark exposure as a proxy for simultaneous factor loading (shotgun limit).
    """
    if total_equity <= 1e-9:
        return True, 1.0
    deployed = 0.0
    for p in positions:
        sym = p["symbol"]
        px = float(mark_prices.get(sym) or p.get("avg_cost") or 0.0)
        sh = float(p.get("shares", 0.0) or 0.0)
        deployed += abs(sh) * px
    cur_frac = deployed / total_equity
    blocked = cur_frac + proposed_trade_frac > global_cap + 1e-9
    return blocked, cur_frac
