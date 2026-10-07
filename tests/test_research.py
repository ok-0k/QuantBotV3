"""
Research backtester correctness: timing (no lookahead), cost / drift / borrow
accounting, resampling, and the candidate signal constructors.
"""

import math

import numpy as np
import pandas as pd
import pytest

from research import backtest as bt
from research import strategies as S
from research.data import resample

IDX_H = pd.date_range("2024-01-01 01:00", periods=24 * 10, freq="1h", tz="UTC")


def _px(values, cols=("A",), index=None):
    index = index if index is not None else pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    arr = np.asarray(values, float)
    return pd.DataFrame(arr if arr.ndim == 2 else arr[:, None], index=index, columns=list(cols))


# ── timing ───────────────────────────────────────────────────────────────────

def test_weight_at_t_earns_return_of_t_plus_1_only():
    close = _px([100, 110, 99, 99])
    w = _px([1, 0, 0, 0])                      # long only at bar 0's close
    r = bt.run(w, close, bars_per_day=1, costs=bt.Costs(0, 0, 0))
    assert list(r.gross.round(10)) == [0, 0.10, 0, 0]


def test_lookahead_is_impossible_with_lagged_signal():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (2000, 5))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0),
                         index=pd.date_range("2020-01-01", periods=2000, freq="1D", tz="UTC"),
                         columns=list("ABCDE"))
    nxt = close.pct_change(fill_method=None).shift(-1)
    cheat = np.sign(nxt).fillna(0) / 5          # peeks at bar t+1
    honest = np.sign(close.pct_change(fill_method=None)).fillna(0) / 5
    zero = bt.Costs(0, 0, 0)
    s_cheat = bt.metrics(bt.run(cheat, close, 1, zero), 365)["sharpe"]
    s_honest = bt.metrics(bt.run(honest, close, 1, zero), 365)["sharpe"]
    assert s_cheat > 20                         # proves t -> t+1 alignment
    assert abs(s_honest) < 1.0                  # random walk: no edge without peeking


# ── costs, drift, borrow ─────────────────────────────────────────────────────

def test_entry_cost_charged_once_on_turnover():
    close = _px([100, 100, 100, 100])
    w = _px([1, 1, 1, 0])
    c = bt.Costs(0.001, 0.0008, 0)
    r = bt.run(w, close, 1, c)
    # entry trade; costs are paid from cash, so holding "weight 1" afterwards
    # re-trims the 0.18% the fee took out of equity (second-order, ~c^2 per bar)
    assert r.turnover.sum() == pytest.approx(1.0, abs=0.005)
    assert r.net.sum() == pytest.approx(-0.0018, abs=1e-5)


def test_rebalancing_drift_is_traded_and_costed():
    close = _px([[100, 100], [110, 100], [110, 100]], cols=("A", "B"))
    w = _px([[0.5, 0.5]] * 3, cols=("A", "B"))
    r = bt.run(w, close, 1, bt.Costs(0.001, 0, 0))
    # after A +10% the book is 0.55/0.50 of 1.0475 equity -> rebalance back to 50/50
    p1 = 0.05 - 0.001                                    # gross minus entry cost
    drifted = np.array([0.55, 0.50]) / (1 + p1)
    assert r.turnover.iloc[2] == pytest.approx(np.abs(np.array([0.5, 0.5]) - drifted).sum())


def test_short_borrow_accrues_per_day():
    close = _px(np.full(25, 100.0), index=pd.date_range("2024-01-01", periods=25, freq="1h", tz="UTC"))
    w = _px(np.full(25, -1.0), index=close.index)
    r = bt.run(w, close, bars_per_day=24, costs=bt.Costs(0, 0, 0.0003))
    assert r.borrow.sum() == pytest.approx(0.0003)


def test_random_signals_lose_the_costs():
    rng = np.random.default_rng(1)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (3000, 8)), axis=0),
                         index=pd.date_range("2020-01-01", periods=3000, freq="1D", tz="UTC"),
                         columns=[f"S{i}" for i in range(8)])
    w = pd.DataFrame(rng.choice([-1, 0, 1], size=close.shape) / 8, index=close.index, columns=close.columns)
    r = bt.run(w, close, 1, bt.Costs())
    assert abs(r.gross.mean()) < 3 * r.gross.std() / math.sqrt(len(r.gross))
    assert r.net.sum() < -0.5 * r.cost.sum()


# ── data helpers ─────────────────────────────────────────────────────────────

def test_resample_daily_ohlc_and_incomplete_days():
    c = pd.DataFrame({"A": np.arange(1.0, len(IDX_H) + 1)}, index=IDX_H)
    panel = {"open": c - 0.5, "high": c + 1, "low": c - 1, "close": c, "qv": c * 0 + 10}
    panel["close"].iloc[30:33] = np.nan                  # 3 missing hours on day 2 (> 10%)
    panel["close"].iloc[60] = np.nan                     # 1 missing hour on day 3 (tolerated)
    d = resample(panel, "1D")
    day1 = pd.Timestamp("2024-01-02", tz="UTC")           # bar covering (01-01 00:00, 01-02 00:00]
    assert d["close"]["A"][day1] == 24.0 and d["high"]["A"][day1] == 25.0
    assert d["low"]["A"][day1] == 0.0 and d["qv"]["A"][day1] == 240.0
    assert np.isnan(d["close"]["A"][pd.Timestamp("2024-01-03", tz="UTC")])   # incomplete day dropped
    assert d["close"]["A"][pd.Timestamp("2024-01-04", tz="UTC")] == 72.0      # one gap hour tolerated
    assert resample(panel, "1D", min_frac=1.0)["close"]["A"].isna().sum() >= 3


def test_tradable_requires_listing_age_and_volume():
    idx = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    close = pd.DataFrame({"OLD": 1.0, "NEW": np.nan, "THIN": 1.0}, index=idx)
    close.loc[idx[40]:, "NEW"] = 1.0
    qv = pd.DataFrame({"OLD": 5e6, "NEW": 5e6, "THIN": 1e5}, index=idx)
    m = S.tradable({"close": close, "qv": qv}, bars_per_day=1)
    assert m["OLD"].iloc[45] and not m["THIN"].iloc[45]
    assert not m["NEW"].iloc[45] and not m["NEW"].iloc[59 - 21]


# ── signals ──────────────────────────────────────────────────────────────────

def test_xs_rank_is_dollar_neutral():
    rng = np.random.default_rng(2)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (60, 20)), axis=0),
                         index=pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC"),
                         columns=[f"S{i}" for i in range(20)])
    mask = close.notna()
    w = S.xs_rank({"close": close}, mask, lookback=7, skip=1)
    live = w.iloc[10:]
    assert np.allclose(live.sum(axis=1), 0) and np.allclose(live.abs().sum(axis=1), 1)


def test_tsmom_sleeves_and_long_only():
    close = _px([[1, 4], [2, 3], [3, 2], [4, 1]], cols=("UP", "DOWN"))
    mask = close.notna()
    w = S.tsmom({"close": close}, mask, lookback=1)
    assert list(w.iloc[3]) == [0.5, -0.5]
    assert list(S.tsmom({"close": close}, mask, 1, long_only=True).iloc[3]) == [0.5, 0.0]


def test_donchian_enters_on_breakout_and_exits_on_reversal():
    up = list(range(100, 130)) + list(range(130, 100, -1))
    close = _px(up)
    p = {"close": close, "high": close + 0.5, "low": close - 0.5}
    w = S.donchian(p, close.notna(), n=10)
    assert w["A"].iloc[15] == 1.0                        # riding the breakout
    assert w["A"].iloc[45] == -1.0                       # flipped short on the downside break


def test_funding_carry_earns_funding_net_of_switch_costs():
    idx = pd.date_range("2024-01-01", periods=40, freq="8h", tz="UTC")
    f = pd.DataFrame({"A": 0.0005}, index=idx)           # 0.05% per 8h
    r = S.funding_carry(f, theta_on=0.0001, trail=3, switch_cost=0.003, capital_eff=1.0)
    assert r["A"].iloc[2] == pytest.approx(0.0005 - 0.003)   # entry bar: income - entry cost
    assert r["A"].iloc[10] == pytest.approx(0.0005)
