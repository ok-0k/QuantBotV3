"""
Carry spot-tradability mask and carry_forward accounting (audit 2026-10-11).

The carry backtest let funding alone decide whether a hedge was on, so a
hedge could outlive its spot pair: MATICUSDT's perp kept printing the default
0.01%/8h after its spot delisting, which never meets the strict exit
(< theta / 2), and that phantom hedge was most of carry_2's OOS result.
"""

import numpy as np
import pandas as pd
import pytest

import research.carry_forward as cf
import research.study as study
from research import strategies as S

IDX = pd.date_range("2024-01-01", periods=40, freq="8h", tz="UTC")


def test_sleeve_cannot_be_on_while_spot_is_not_tradable():
    f = pd.DataFrame({"A": 0.0005, "B": 0.0005}, index=IDX)           # funding keeps printing...
    spot = pd.DataFrame({"A": [True] * 20 + [False] * 20}, index=IDX)  # ...after A's spot ends; B has no spot
    r, on = S.funding_carry(f, theta_on=0.0001, trail=3, switch_cost=0.003, capital_eff=1.0,
                            tradable=spot, return_state=True)
    assert on["A"].iloc[2:20].eq(1).all()
    assert on["A"].iloc[20:].eq(0).all()                 # forced flat once spot stops trading
    assert r["A"].iloc[19] == pytest.approx(0.0005)      # last hedged period still earns
    assert r["A"].iloc[20] == pytest.approx(-0.003)      # the forced exit pays its cost
    assert r["A"].iloc[21:].isna().all()                 # and the sleeve stops counting
    assert on["B"].eq(0).all() and r["B"].isna().all()   # no spot pair at all -> never hedged

    unmasked, on_u = S.funding_carry(f, theta_on=0.0001, trail=3, switch_cost=0.003,
                                     capital_eff=1.0, return_state=True)
    assert on_u["A"].iloc[2:].eq(1).all()                # the old behaviour, for contrast


def test_spot_tradable_buckets_by_window_end_like_funding_buckets():
    idx = pd.date_range("2024-09-09 01:00", "2024-09-11 00:00", freq="1h", tz="UTC")
    close = pd.DataFrame({"MATICUSDT": 1.0, "BTCUSDT": 1.0}, index=idx)
    close.loc[close.index > pd.Timestamp("2024-09-10 02:00", tz="UTC"), "MATICUSDT"] = np.nan
    close.loc[pd.Timestamp("2024-09-10 08:00", tz="UTC"):pd.Timestamp("2024-09-10 16:00", tz="UTC"),
              "BTCUSDT"] = np.nan
    close.loc[pd.Timestamp("2024-09-10 16:00", tz="UTC"), "BTCUSDT"] = 1.0   # bar (15:00, 16:00]
    ok = study.spot_tradable(close)
    t = lambda s: pd.Timestamp(f"2024-09-10 {s}", tz="UTC")
    assert set(ok.index.hour) == {0, 8, 16}
    assert ok.loc[t("08:00"), "MATICUSDT"]               # last close 02:00 is inside (00:00, 08:00]
    assert not ok.loc[t("16:00"), "MATICUSDT"]
    assert not ok.loc[pd.Timestamp("2024-09-11 00:00", tz="UTC"), "MATICUSDT"]
    assert ok.loc[t("08:00"), "BTCUSDT"]                 # closes 01:00..07:00
    assert ok.loc[t("16:00"), "BTCUSDT"]                 # a single close at the window end counts


def test_matic_like_phantom_hedge_no_longer_contributes():
    idx = pd.date_range("2024-08-01", periods=120, freq="8h", tz="UTC")
    delist = 60
    f = pd.DataFrame({"MATICUSDT": [0.0004] * delist + [0.0001] * (120 - delist),   # exchange default
                      "BTCUSDT": 0.00005}, index=idx)                                # never enters
    spot = pd.DataFrame({"MATICUSDT": [True] * delist + [False] * (120 - delist),
                         "BTCUSDT": True}, index=idx)

    phantom, on_p = S.funding_carry(f, theta_on=0.0002, return_state=True)
    assert on_p["MATICUSDT"].iloc[-1] == 1               # the bug: flat 1e-4 never meets "< 1e-4"
    assert phantom["MATICUSDT"].iloc[delist + 1:-1].gt(0).all()

    masked, on_m = S.funding_carry(f, theta_on=0.0002, return_state=True, tradable=spot)
    assert on_m["MATICUSDT"].iloc[delist:].eq(0).all()
    assert masked["MATICUSDT"].iloc[delist + 1:].isna().all()

    old = study._carry_result(f, 0.0002).net
    new = study._carry_result(f, 0.0002, tradable=spot).net
    after = idx[delist + 2:]                             # portfolio returns are labelled by period end
    assert old[after].sum() > 0
    assert new[after].abs().max() == 0.0


def test_carry_forward_deployed_apr_includes_exit_costs():
    f = pd.DataFrame({"A": [0.0005] * 10 + [0.0] * 10}, index=IDX[:20])
    ret, on = S.funding_carry(f, theta_on=0.0001, trail=3, switch_cost=0.003, capital_eff=1.0,
                              return_state=True)
    (trade,) = cf.trades(ret, on)
    sym, opened, closed, net, periods = trade
    assert (opened, closed) == (IDX[2], IDX[12])         # hedged bars 2..11, exit booked at bar 12
    assert periods == 10
    assert net == pytest.approx(7 * 0.0005 - 2 * 0.003)  # income less entry AND exit cost
    apr = cf.deployed_apr([trade])
    assert apr == pytest.approx(net / periods * cf.PERIODS_PER_YEAR)
    assert apr < 0
    # the old measure (mean over hedged periods) dropped the exit cost and would have
    # cleared the 5% APR bar on this losing trade
    old = ret.to_numpy()[on.to_numpy(bool)].mean() * cf.PERIODS_PER_YEAR
    assert old >= cf.MIN_DEPLOYED_APR


def test_carry_forward_open_trade_has_no_exit_cost_and_is_not_closed():
    f = pd.DataFrame({"A": 0.0005}, index=IDX[:10])
    ret, on = S.funding_carry(f, theta_on=0.0001, trail=3, switch_cost=0.003, capital_eff=1.0,
                              return_state=True)
    (trade,) = cf.trades(ret, on)
    assert trade[2] is None and trade[4] == 8
    assert np.isnan(cf.deployed_apr([]))
