"""Synthetic checks for research/explore_perp.py (EXPLORATORY — NOT EVIDENCE)."""

import numpy as np
import pandas as pd

from research import backtest as bt
from research import explore_perp as X
from research import smc


def _panel(T=300, N=4, seed=1, freq="1h"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq=freq, tz="UTC")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0)), index=idx,
                         columns=[f"S{i}" for i in range(N)])
    w = pd.DataFrame(np.sign(rng.normal(size=(T, N))) / N, index=idx, columns=close.columns)
    return close, w


def test_run_perp_equals_backtest_run_without_funding():
    close, w = _panel()
    ta = pd.Series(np.arange(len(close)) % 8 == 1, index=close.index)
    for band, trade_at in ((0.0, None), (0.5, None), (0.5, ta)):
        a = bt.run(w, close, 24, bt.Costs(), band=band, trade_at=trade_at)
        b, ex = X.run_perp(w, close, 24, bt.Costs().side, bt.Costs().borrow_daily, band, trade_at)
        assert np.array_equal(a.net.to_numpy(), b.net.to_numpy())
        assert np.array_equal(a.turnover.to_numpy(), b.turnover.to_numpy())
        assert ex["funding"].abs().sum() == 0


def test_funding_sign_and_bar_labelling():
    close, _ = _panel(T=30, N=2)
    close[:] = 100.0                                     # no price P&L
    w = pd.DataFrame({"S0": 0.5, "S1": -0.5}, index=close.index)
    t = close.index[10]
    fund = {"S0": pd.Series([0.001], index=[t + pd.Timedelta("3ms")]),
            "S1": pd.Series([0.001], index=[t])}
    F = X.funding_on_grid(fund, close.index, close.columns)
    assert F[10].tolist() == [0.001, 0.001] and F.sum() == 0.002
    res, ex = X.run_perp(w, close, 24, 0.0, fund=F)
    assert np.isclose(ex["funding"].iloc[10], -0.5 * 0.001 + 0.5 * 0.001)   # long pays, short receives
    res, ex = X.run_perp(w[["S1"]].reindex(columns=close.columns, fill_value=0.0), close, 24, 0.0, fund=F)
    assert np.isclose(ex["funding"].iloc[10], 0.0005) and np.isclose(res.net.iloc[10], 0.0005)


def test_funding_daily_grid_window():
    idx = pd.date_range("2024-01-02", periods=5, freq="1D", tz="UTC")
    s = pd.Series([1.0, 2.0, 4.0], index=pd.to_datetime(["2024-01-02 08:00", "2024-01-02 16:00",
                                                          "2024-01-03 00:00"], utc=True))
    F = X.funding_on_grid({"A": s}, idx, ["A"])[:, 0]
    assert F.tolist() == [0.0, 7.0, 0.0, 0.0, 0.0]       # (Jan 2, Jan 3] -> label Jan 3


def test_events_count_sign_changes_not_drift():
    close, _ = _panel(T=6, N=1)
    w = pd.DataFrame({"S0": [0.0, 1.0, 1.0, -1.0, -1.0, 0.0]}, index=close.index)
    _, ex = X.run_perp(w, close, 24, 0.0)
    assert ex["events"].tolist() == [0, 0, 1, 0, 1, 0]
    assert ex["entries"].tolist() == [0, 0, 1, 0, 1, 0]


class _C:
    def __init__(self, side, limit, stop):
        self.side, self.limit, self.stop, self.sym = side, limit, stop, "X"


def test_perp_net_r_matches_net_r_when_costs_coincide():
    c = _C(-1, -100.0, -101.0)                           # short at 100, stop 101 (long space)
    oc = smc.Outcome(True, "stop", 10, 0, ((0.5, -98.0, False), (0.5, -100.5, True)), 1)
    same = X.PerpCost(taker=0.001, maker=0.001, slip=0.0008, funding=False)
    ref = smc.net_r(c, oc, smc.Cost(0.001, 0.0008, 0.0))
    assert np.isclose(X.perp_net_r(c, oc, same), ref)
    X._FUND_MS["X"] = (np.array([5], np.int64), np.array([0.0, 0.0002]))
    fund = X.perp_net_r(c, oc, X.PerpCost(0.001, 0.001, 0.0008)) - ref
    assert np.isclose(fund, 0.0002 * 100.0)               # short receives rate x notional (100 R units)
    X._FUND_MS.pop("X")
