"""
Cost-aware, bar-close portfolio backtester.

Timing: the weight row at bar t is decided at bar t's CLOSE and earns the
return of bar t+1 (close_t -> close_{t+1}). A signal can therefore only use
information up to and including bar t — there is no way to trade on bar t+1.

Costs (defaults match the live bot's paper model, so research and paper
results are comparable):
  - every change in position, including rebalancing away price drift, pays
    fee_side + slip_side per unit of notional traded (0.10% + 0.08%);
  - short notional pays borrow_daily per day (margin interest; the live bot
    does not model this yet, so this is the more conservative assumption).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee_side: float = 0.0010
    slip_side: float = 0.0008
    borrow_daily: float = 0.0003

    @property
    def side(self) -> float:
        return self.fee_side + self.slip_side

    def scaled(self, k: float) -> "Costs":
        return replace(self, fee_side=self.fee_side * k, slip_side=self.slip_side * k,
                       borrow_daily=self.borrow_daily * k)


@dataclass
class Result:
    net: pd.Series            # per-bar net return, labelled by the bar it is earned over
    gross: pd.Series
    cost: pd.Series
    borrow: pd.Series
    turnover: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series

    def window(self, start=None, end=None) -> "Result":
        sl = slice(start, end)
        return Result(*(getattr(self, f)[sl] for f in self.__dataclass_fields__))


def run(weights: pd.DataFrame, close: pd.DataFrame, bars_per_day: float,
        costs: Costs = Costs(), band: float = 0.0) -> Result:
    """Simulate target weights against close prices (same index/columns).

    band: no-trade band. A position whose sign matches its target and whose
    drifted weight is within band x |target| of it is left alone instead of
    being traded back to the exact target. band=0 rebalances every bar
    (constant-weight; for volatile shorts this behaves like an inverse ETF
    and decays). band=0.5 trades on signal changes and large drift only —
    how a discretionary or systematic trader actually holds positions.
    """
    close = close.sort_index()
    W = weights.reindex(index=close.index, columns=close.columns).fillna(0.0).to_numpy(float)
    R = close.pct_change(fill_method=None).to_numpy(float)
    R = np.where(np.isfinite(R), R, 0.0)            # missing next bar: flat (no fill possible)
    T, N = W.shape
    net = np.zeros(T); gross = np.zeros(T); cost = np.zeros(T); borrow = np.zeros(T)
    turn = np.zeros(T); gexp = np.zeros(T); nexp = np.zeros(T)
    held = np.zeros(N)                              # weights carried into bar t, after drift
    borrow_bar = costs.borrow_daily / bars_per_day
    for t in range(T - 1):
        target = W[t]
        if band > 0:
            keep = (np.sign(held) == np.sign(target)) & (target != 0) & \
                   (np.abs(held - target) <= band * np.abs(target))
            target = np.where(keep, held, target)
        traded = np.abs(target - held).sum()
        c = traded * costs.side
        r = R[t + 1]
        g = float(target @ r)
        b = float(np.clip(-target, 0.0, None).sum()) * borrow_bar
        p = g - c - b
        net[t + 1], gross[t + 1], cost[t + 1], borrow[t + 1] = p, g, c, b
        turn[t + 1] = traded
        gexp[t + 1] = np.abs(target).sum()
        nexp[t + 1] = target.sum()
        if 1.0 + p <= 0.0:                          # account wiped out
            net[t + 1] = -1.0
            net[t + 2:] = 0.0
            break
        held = target * (1.0 + r) / (1.0 + p)
    idx = close.index
    mk = lambda a: pd.Series(a, index=idx)
    return Result(mk(net), mk(gross), mk(cost), mk(borrow), mk(turn), mk(gexp), mk(nexp))


def _daily(r: pd.Series) -> pd.Series:
    return (1.0 + r).resample("1D", closed="right", label="right").prod() - 1.0


def metrics(res: Result, bars_per_year: float, bench: pd.Series | None = None) -> dict:
    """Headline stats for a (windowed) result. Bench: per-bar net returns of a
    benchmark over the same bars, for beta / alpha (computed on daily returns)."""
    r = res.net
    n = len(r)
    if n < 2:
        return {"bars": n}
    years = n / bars_per_year
    eq = (1.0 + r).cumprod()
    sd = r.std()
    sharpe = r.mean() / sd * math.sqrt(bars_per_year) if sd > 0 else 0.0   # never traded -> 0
    out = {
        "years": round(years, 2),
        "total_return": eq.iloc[-1] - 1.0,
        "cagr": eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else -1.0,
        "vol": r.std() * math.sqrt(bars_per_year),
        "sharpe": sharpe,
        "sharpe_se": math.sqrt((1.0 + 0.5 * sharpe ** 2) / years),
        "max_dd": float((eq / eq.cummax() - 1.0).min()),
        "turnover_yr": res.turnover.sum() / years,
        "cost_yr": (res.cost.sum() + res.borrow.sum()) / years,
        "gross_return_yr": res.gross.sum() / years,
        "avg_gross_exposure": res.gross_exposure.mean(),
        "avg_net_exposure": res.net_exposure.mean(),
    }
    m = _daily(r).dropna()
    out["pct_pos_months"] = float(((1 + r).resample("ME").prod() - 1 > 0).mean())
    if bench is not None:
        b = _daily(bench).reindex(m.index).fillna(0.0)
        if b.var() > 0 and len(m) > 30:
            beta = float(np.cov(m, b)[0, 1] / b.var())
            resid = m - beta * b
            alpha_d = resid.mean()
            se = resid.std() / math.sqrt(len(resid))
            out.update(beta=beta, alpha_yr=alpha_d * 365, alpha_t=alpha_d / se if se > 0 else float("nan"))
    return out
