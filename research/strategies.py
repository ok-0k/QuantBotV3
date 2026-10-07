"""
Candidate signals -> target weight frames (rows: decision bar, cols: symbol).

Every function only uses data up to and including the decision bar. Sleeve
strategies (trend, breakout, mean reversion) give each tradable symbol an
equal 1/N slice of capital; cross-sectional strategies are dollar-neutral
(+0.5 long book, -0.5 short book).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tradable(p: dict[str, pd.DataFrame], bars_per_day: float,
             min_age_days: int = 30, min_daily_qv: float = 2e6) -> pd.DataFrame:
    """Listed for >= min_age_days (no new-listing noise), trailing 7-day average
    quote volume >= min_daily_qv, and has a price at the decision bar."""
    close, qv = p["close"], p["qv"]
    age_bars = int(min_age_days * bars_per_day)
    listed = close.notna().rolling(age_bars, min_periods=1).sum() >= int(0.95 * age_bars)
    wk = int(7 * bars_per_day)
    adv = qv.rolling(wk, min_periods=wk // 2).mean() * bars_per_day >= min_daily_qv
    return listed & adv & close.notna()


def _per_sleeve(sig: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    n = mask.sum(axis=1).replace(0, np.nan)
    return sig.where(mask, 0.0).fillna(0.0).div(n, axis=0).fillna(0.0)


def equal_weight(p, mask):
    return _per_sleeve(mask.astype(float), mask)


def hold(p, mask, symbol="BTCUSDT"):
    w = pd.DataFrame(0.0, index=mask.index, columns=mask.columns)
    w[symbol] = mask[symbol].astype(float)
    return w


def tsmom(p, mask, lookback: int, long_only: bool = False):
    """Time-series momentum: long if the trailing return is up, short (or flat) if down."""
    past = p["close"] / p["close"].shift(lookback) - 1.0
    sig = np.sign(past)
    if long_only:
        sig = sig.clip(lower=0.0)
    return _per_sleeve(sig, mask)


def xs_rank(p, mask, lookback: int, skip: int = 0, quantile: float = 0.2,
            reverse: bool = False, min_names: int = 10):
    """Cross-sectional: long the top quantile of trailing return (t-lookback ..
    t-skip), short the bottom quantile; reverse=True flips (short-term reversal)."""
    c = p["close"]
    past = (c.shift(skip) / c.shift(lookback) - 1.0).where(mask)
    rk = past.rank(axis=1, pct=True)
    long_, short_ = rk > 1.0 - quantile, rk <= quantile
    if reverse:
        long_, short_ = short_, long_
    w = (long_.astype(float).div(long_.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
         - short_.astype(float).div(short_.sum(axis=1).replace(0, np.nan), axis=0) * 0.5)
    w[past.notna().sum(axis=1) < min_names] = 0.0
    return w.fillna(0.0)


def _state_machine(enter_long, exit_long, enter_short, exit_short, mask) -> pd.DataFrame:
    el, xl, es, xs = (x.fillna(False).to_numpy(bool) for x in (enter_long, exit_long, enter_short, exit_short))
    m = mask.to_numpy(bool)
    T, N = el.shape
    st = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        cur = np.where(~m[t], 0.0, cur)                  # untradable -> flat
        cur = np.where((cur > 0) & xl[t], 0.0, cur)
        cur = np.where((cur < 0) & xs[t], 0.0, cur)
        cur = np.where((cur == 0) & el[t] & m[t], 1.0, cur)
        cur = np.where((cur == 0) & es[t] & m[t], -1.0, cur)
        st[t] = cur
    return pd.DataFrame(st, index=mask.index, columns=mask.columns)


def donchian(p, mask, n: int, long_only: bool = False):
    """Breakout: enter on a close beyond the prior n-bar extreme, exit on a
    close beyond the opposite prior n/2-bar extreme (classic turtle rules)."""
    c = p["close"]
    hi, lo = p["high"].rolling(n).max().shift(1), p["low"].rolling(n).min().shift(1)
    hx, lx = p["high"].rolling(n // 2).max().shift(1), p["low"].rolling(n // 2).min().shift(1)
    never = pd.DataFrame(False, index=c.index, columns=c.columns)
    st = _state_machine(c > hi, c < lx, never if long_only else c < lo, c > hx, mask)
    return _per_sleeve(st, mask)


def zscore_mr(p, mask, window: int, z_in: float = 2.5, z_out: float = 0.5):
    """Short-horizon mean reversion: fade |z| > z_in moves vs the rolling mean,
    exit when |z| < z_out."""
    c = p["close"]
    z = (c - c.rolling(window).mean()) / c.rolling(window).std()
    st = _state_machine(z < -z_in, z.abs() < z_out, z > z_in, z.abs() < z_out, mask)
    return _per_sleeve(st, mask)


def funding_carry(funding: pd.DataFrame, theta_on: float, trail: int = 9,
                  switch_cost: float = 0.0031, capital_eff: float = 1 / 1.2) -> pd.DataFrame:
    """Delta-neutral cash-and-carry on perps: hold long spot + short perp while
    trailing mean funding (per 8h) > theta_on; exit below theta_on / 2.

    Returns a DataFrame of per-settlement sleeve returns (rows: settlement
    time, cols: symbol) already net of switching costs; the portfolio is the
    equal-weight mean across symbols that have funding data. Income for a
    position held after settlement t is the funding paid at t+1.
    switch_cost per open or close = spot leg (0.10% fee + 0.08% slip) +
    perp leg (0.05% taker + 0.08% slip) = 0.31% of notional; notional is
    1/1.2 of sleeve capital (20% perp margin).
    Basis drift between spot and perp is ignored (hedged price P&L ~ 0).
    """
    tr = funding.rolling(trail, min_periods=trail).mean()
    on = _state_machine(tr > theta_on, tr < theta_on / 2,
                        pd.DataFrame(False, index=tr.index, columns=tr.columns),
                        pd.DataFrame(False, index=tr.index, columns=tr.columns),
                        funding.notna())
    income = (on * funding.shift(-1).fillna(0.0)) * capital_eff
    switches = on.diff().abs().fillna(on.abs())
    return (income - switches * switch_cost * capital_eff).where(funding.notna())
