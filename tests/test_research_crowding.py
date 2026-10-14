"""
Crowding study (research/PREREG_FUNDING.md): alignment, no-lookahead, ranking,
eligibility, schedule and decision rules, on SYNTHETIC data only. No real
funding / premium data is read here and no real-data result is computed.
"""

import numpy as np
import pandas as pd
import pytest

from research import backtest as bt
from research import crowding as C

SYMS = [f"S{i:02d}" for i in range(12)]
G = pd.date_range("2024-01-01 00:00", periods=3 * 40, freq="8h", tz="UTC")      # 40 days of 8h labels
H = pd.date_range(G[0] + pd.Timedelta("1h"), G[-1] + pd.Timedelta("8h"), freq="1h")


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    F = pd.DataFrame(rng.normal(1e-4, 1e-4, (len(G), len(SYMS))).round(8), index=G, columns=SYMS)
    P = pd.DataFrame(rng.normal(0, 5e-4, (len(G), len(SYMS))), index=G, columns=SYMS)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(H), len(SYMS))), axis=0)),
                         index=H, columns=SYMS)
    ok = pd.DataFrame(True, index=G, columns=SYMS)
    return F, P, close, ok


def _run(F, P, close, ok, cfg, costs=bt.Costs(0, 0, 0)):
    score = C.signals(F, P)[cfg.signal]
    W, ta, log = C.targets(score, C.eligible(score, ok, ok), cfg.k, cfg.book, cfg.period, close.index)
    return bt.run(W, close, C.BARS_PER_DAY, costs, band=C.BAND, trade_at=ta), W, ta, log


# ── grid ─────────────────────────────────────────────────────────────────────

def test_grid_is_the_48_preregistered_configs():
    g = C.grid()
    assert len(g) == 48 and len({c.name for c in g}) == 48
    assert {c.signal for c in g} == set(C.SIGNALS) and {c.book for c in g} == {"so", "ls"}
    assert {c.k for c in g} == {3, 5} and {c.period for c in g} == {"8h", "24h", "72h"}
    names = lambda cs: sorted(x.name for x in cs)
    assert names(C.neighbours(C.Config("fund_lvl", "so", 3, "8h"))) == ["fund_lvl_so_k3_24h", "fund_lvl_so_k5_8h"]
    assert names(C.neighbours(C.Config("prem_chg", "ls", 5, "24h"))) == [
        "prem_chg_ls_k3_24h", "prem_chg_ls_k5_72h", "prem_chg_ls_k5_8h"]


# ── signals: alignment and no lookahead ──────────────────────────────────────

def test_trailing_mean_uses_labels_up_to_t_and_needs_a_full_window():
    x = pd.DataFrame({"A": np.arange(1.0, 13.0)}, index=G[:12])
    m = C.trailing_mean(x, 3)["A"]
    assert m.iloc[:2].isna().all()
    assert m.iloc[2] == pytest.approx(2.0)             # labels 0, 1, 2
    assert m.iloc[11] == pytest.approx(11.0)           # labels 9, 10, 11 — never 12
    x.iloc[6, 0] = np.nan                              # a missing window is not skipped over
    m = C.trailing_mean(x, 3)["A"]
    assert m.iloc[6:9].isna().all() and m.iloc[9] == pytest.approx(9.0)


def test_signal_formulas():
    F = pd.DataFrame({"A": np.r_[np.full(21, 1e-4), np.full(3, 4e-4)]}, index=G[:24])
    s = C.signals(F, F * 0)
    assert s["fund_lvl"]["A"].iloc[-1] == pytest.approx((6 * 1e-4 + 3 * 4e-4) / 9)
    assert s["fund_chg"]["A"].iloc[-1] == pytest.approx(4e-4 - (18 * 1e-4 + 3 * 4e-4) / 21)
    assert s["fund_chg"]["A"].iloc[:20].isna().all() and s["prem_lvl"]["A"].iloc[-1] == 0


def test_on_grid_marks_gaps_nan_and_rejects_off_grid_labels():
    x = pd.DataFrame({"A": [1.0, 2.0]}, index=G[[0, 2]])
    assert C.on_grid(x, G[:3])["A"].isna().tolist() == [False, True, False]
    with pytest.raises(ValueError):
        C.on_grid(pd.DataFrame({"A": [1.0]}, index=G[:1] + pd.Timedelta("4h")), G)


def test_nothing_before_t_plus_lag_depends_on_data_after_t():
    """Rewrite every funding and premium value after t0: signals, eligibility and
    targets up to t0 are unchanged, and so is every backtest return up to the
    first bar a later decision could reach (t0 + 8h decision, filled t0 + 9h)."""
    F, P, close, ok = _synthetic(1)
    t0 = G[70]
    F2, P2 = F.copy(), P.copy()
    rng = np.random.default_rng(99)
    F2.loc[F2.index > t0] = rng.normal(0, 1e-3, F2.loc[F2.index > t0].shape)
    P2.loc[P2.index > t0] = rng.normal(0, 1e-2, P2.loc[P2.index > t0].shape)
    for name in C.SIGNALS:
        a, b = C.signals(F, P)[name], C.signals(F2, P2)[name]
        pd.testing.assert_frame_equal(a[:t0], b[:t0], check_exact=True)
    for cfg in (C.Config("fund_lvl", "so", 3, "8h"), C.Config("prem_chg", "ls", 5, "8h")):
        r1, W1, ta1, _ = _run(F, P, close, ok, cfg)
        r2, W2, ta2, _ = _run(F2, P2, close, ok, cfg)
        pd.testing.assert_frame_equal(W1[:t0 + C.LAG], W2[:t0 + C.LAG], check_exact=True)
        cut = t0 + C.STEP + C.LAG
        pd.testing.assert_series_equal(r1.net[:cut], r2.net[:cut], check_exact=True)
        assert not np.allclose(r1.net[cut:], r2.net[cut:])     # the perturbation does matter later


def test_rebalance_weights_sit_on_the_row_one_hour_after_the_decision():
    F, P, close, ok = _synthetic(2)
    F["S00"] = 1e-3                                     # S00 is always the most crowded
    t = G[60]
    r, W, ta, _ = _run(F, P, close, ok, C.Config("fund_lvl", "so", 3, "8h"))
    assert W.loc[t + C.LAG, "S00"] == pytest.approx(-1 / 3)
    window = ta[t:t + C.STEP]                           # 1h rows t .. t+8h
    assert list(window[window].index) == [t + C.LAG]
    assert W.loc[t].eq(0).all()                         # nothing is placed on the decision row itself


def test_a_signal_change_at_t_cannot_trade_the_bar_ending_at_t_plus_1h():
    """A coin becomes most crowded only at decision t (its funding spikes in the
    window ending t) and its price jumps in (t, t+1h]: no return is earned on it
    in that bar; the position starts with the bar (t+1h, t+2h]."""
    F, P, close, ok = _synthetic(3)
    F.loc[:, :] = 1e-4
    F.iloc[:, :3] = 5e-4                                # S00..S02 crowded before t
    t = G[60]
    F.loc[t, "S11"] = 0.05                              # S11 becomes top crowded at t only
    close = close.copy()
    close.loc[:, :] = 100.0
    close.loc[t + pd.Timedelta("1h"):, "S11"] = 150.0   # jump in (t, t+1h]
    close.loc[t + pd.Timedelta("2h"):, "S11"] = 120.0   # -20% in (t+1h, t+2h]
    r, W, ta, _ = _run(F, P, close, ok, C.Config("fund_lvl", "so", 3, "8h"))
    assert W.loc[:t, "S11"].eq(0).all() and W.loc[t + C.LAG, "S11"] < 0
    assert r.gross[t + pd.Timedelta("1h")] == pytest.approx(0.0)      # the jump bar: not held
    assert r.gross[t + pd.Timedelta("2h")] == pytest.approx(-W.loc[t + C.LAG, "S11"] * 0.2)


# ── ranking and books ────────────────────────────────────────────────────────

def test_ties_at_the_cut_share_the_remaining_slots():
    s = pd.Series([5.0, 4.0, 3.0, 3.0, 3.0, 1.0, 0.0, 0.0, -1.0, -2.0], index=SYMS[:10])
    w = C.book_weights(s, 3, "so")
    assert w.iloc[:2].tolist() == pytest.approx([-1 / 3, -1 / 3])
    assert w.iloc[2:5].tolist() == pytest.approx([-1 / 9] * 3)        # one slot shared by three
    assert w.iloc[5:].eq(0).all() and w.sum() == pytest.approx(-1.0)


def test_float_noise_does_not_break_a_tie():
    s = pd.Series([0.1 + 0.2, 0.3, 0.5] + [0.0] * 9, index=SYMS)      # 0.30000000000000004 vs 0.3
    w = C.book_weights(s * 1e-4, 2, "so")
    assert w["S00"] == pytest.approx(w["S01"]) == pytest.approx(-0.25)
    assert w["S02"] == pytest.approx(-0.5)


def test_long_short_book_is_neutral_and_nets_out_when_all_tie():
    s = pd.Series(np.arange(12.0), index=SYMS)
    w = C.book_weights(s, 5, "ls")
    assert w.sum() == pytest.approx(0) and w.abs().sum() == pytest.approx(1)
    assert (w[SYMS[-5:]] == -0.1).all() and (w[SYMS[:5]] == 0.1).all()
    flat = C.book_weights(pd.Series(1e-4, index=SYMS), 5, "ls")       # all funding at the default
    assert flat.abs().max() == pytest.approx(0.0)


# ── eligibility and skipped rebalances ───────────────────────────────────────

def test_phantom_funding_after_spot_delisting_is_never_ranked():
    F, P, close, ok = _synthetic(4)
    F["S00"] = 1e-3                                     # always the most crowded...
    ok = ok.copy()
    ok.loc[G[50]:, "S00"] = False                       # ...but its spot pair stops trading
    score = C.signals(F, P)["fund_lvl"]
    W, ta, _ = C.targets(score, C.eligible(score, ok, ok.astype(bool) | True), 3, "so", "8h", close.index)
    assert W.loc[:G[49] + C.LAG, "S00"].min() < 0
    assert W.loc[G[50] + C.LAG:, "S00"].eq(0).all()
    # nor when only the 1h universe filter (liquidity / listing age) drops it
    W2, _, _ = C.targets(score, C.eligible(score, ok | True, ok), 3, "so", "8h", close.index)
    assert W2.loc[G[50] + C.LAG:, "S00"].eq(0).all()


def test_too_few_eligible_skips_the_rebalance_and_holds_positions():
    F, P, close, ok = _synthetic(5)
    ok = ok.copy()
    ok.loc[G[60]:G[62], SYMS[:3]] = False               # 9 eligible at 3 decisions
    cfg = C.Config("fund_lvl", "so", 3, "8h")
    r, W, ta, log = _run(F, P, close, ok, cfg)
    assert log.loc[G[60]:G[62], "skipped"].all() and (log.loc[G[60]:G[62], "n_eligible"] == 9).all()
    assert not log.loc[G[63]:, "skipped"].any()
    assert not ta[G[60:63] + C.LAG].any()
    assert r.turnover[G[60] + C.LAG + pd.Timedelta("1h"):G[63] + C.LAG].sum() == 0   # held, untraded
    assert r.gross_exposure[G[61]] > 0                  # still in the market while skipping
    assert log.loc[:G[7], "skipped"].all()              # warm-up: fund_lvl needs 9 windows
    assert not log.loc[G[8]:G[59], "skipped"].any()


def test_decision_schedule_anchors():
    g = pd.date_range("2025-03-30 00:00", "2025-04-10 00:00", freq="8h", tz="UTC")
    assert len(C.decision_times(g, "8h")) == len(g)
    d24 = C.decision_times(g, "24h")
    assert (d24.hour == 0).all() and len(d24) == 12
    d72 = C.decision_times(g, "72h")                    # day 20179 = 2025-04-01 is 1 mod 3
    assert [str(t.date()) for t in d72] == ["2025-03-31", "2025-04-03", "2025-04-06", "2025-04-09"]


def test_rebalances_only_on_schedule():
    F, P, close, ok = _synthetic(6)
    for period, every in (("8h", 8), ("24h", 24), ("72h", 72)):
        r, W, ta, log = _run(F, P, close, ok, C.Config("prem_lvl", "ls", 3, period), costs=bt.Costs())
        traded = r.turnover[r.turnover > 0].index - pd.Timedelta("1h")   # turnover is booked one bar later
        assert set(traded - C.LAG) <= set(C.decision_times(G, period))
        assert ta[ta].index.to_series().diff().min() == pd.Timedelta(hours=every)


# ── backtester hold mask ─────────────────────────────────────────────────────

def test_trade_at_all_true_is_the_default_and_false_holds_drift():
    _, _, close, _ = _synthetic(7)
    w = pd.DataFrame(np.tile(np.r_[-0.2, 0.2, np.zeros(10)], (len(close), 1)), index=close.index, columns=SYMS)
    a = bt.run(w, close, 24, band=0.5)
    b = bt.run(w, close, 24, band=0.5, trade_at=pd.Series(True, index=close.index))
    assert np.allclose(a.net, b.net) and np.allclose(a.turnover, b.turnover)
    once = pd.Series(False, index=close.index)
    once.iloc[0] = True
    h = bt.run(w, close, 24, bt.Costs(0, 0, 0), trade_at=once)
    assert h.turnover.iloc[1] == pytest.approx(0.4) and h.turnover.iloc[2:].sum() == 0
    held_short = -(close["S00"].iloc[-1] / close["S00"].iloc[0] - 1) * 0.2   # fixed quantities
    held_long = (close["S01"].iloc[-1] / close["S01"].iloc[0] - 1) * 0.2
    assert (1 + h.net).prod() - 1 == pytest.approx(held_short + held_long, rel=1e-9)


def test_cost_cases():
    base, x2, borrow = C.COSTS["base"], C.COSTS["2x"], C.COSTS["borrow"]
    assert (base.fee_side, base.slip_side, base.borrow_daily) == (0.0010, 0.0008, 0.0003)
    assert (x2.fee_side, x2.slip_side, x2.borrow_daily) == pytest.approx((0.0020, 0.0016, 0.0006))
    assert (borrow.fee_side, borrow.slip_side, borrow.borrow_daily) == (0.0010, 0.0008, 0.0010)
    F, P, close, ok = _synthetic(8)
    cfg = C.Config("fund_lvl", "so", 5, "24h")
    zero_fee = lambda b: bt.Costs(0, 0, b)
    lo, *_ = _run(F, P, close, ok, cfg, costs=zero_fee(0.0003))
    hi, *_ = _run(F, P, close, ok, cfg, costs=zero_fee(C.BORROW_STRESS_DAILY))
    assert hi.borrow.sum() == pytest.approx(lo.borrow.sum() * 0.0010 / 0.0003, rel=0.01)


# ── inputs (synthetic loaders) ───────────────────────────────────────────────

def test_load_inputs_requires_offline(monkeypatch):
    monkeypatch.setattr(C.D, "OFFLINE", False)
    with pytest.raises(RuntimeError):
        C.load_inputs(SYMS)


def test_load_inputs_truncates_at_end_and_aligns_funding_and_premium(monkeypatch):
    idx1h = pd.date_range(C.DATA_START + pd.Timedelta("1h"), C.END + pd.Timedelta("5D"), freq="1h")
    syms = ["AAA", "BBB"]

    def panel(symbols, interval, start, end=None):
        c = pd.DataFrame(100.0, index=idx1h, columns=symbols)
        return {"open": c, "high": c, "low": c, "close": c, "qv": c * 0 + 1e6}

    def funding(symbol, start, end=None):           # settlements with ms jitter; AAA settles every 4h
        step = "4h" if symbol == "AAA" else "8h"
        t = pd.date_range(C.DATA_START, C.END + pd.Timedelta("2D"), freq=step)
        return pd.Series(1e-4 if symbol == "BBB" else 5e-5, index=t + pd.Timedelta("8ms"))

    def premium(symbol, start, end=None):           # labelled by bar close, like research.data
        t = pd.date_range(C.DATA_START + pd.Timedelta("8h"), C.END + pd.Timedelta("2D"), freq="8h")
        return pd.Series(np.arange(len(t), dtype=float), index=t)

    monkeypatch.setattr(C.D, "OFFLINE", True)
    monkeypatch.setattr(C.D, "build_panel", panel)
    monkeypatch.setattr(C.D, "load_funding", funding)
    monkeypatch.setattr(C.D, "load_premium", premium)
    x = C.load_inputs(syms)
    assert x["p1h"]["close"].index.max() == C.END
    assert x["F"].index.max() == C.END and x["P"].index.max() == C.END
    assert x["F"].loc[C.END, "AAA"] == pytest.approx(1e-4)      # 4h settlements summed per 8h
    assert x["F"].loc[C.END, "BBB"] == pytest.approx(1e-4)      # settlement at END+8ms is in END's window
    assert x["P"].loc[C.DATA_START + pd.Timedelta("8h"), "AAA"] == 0.0
    assert set(x["F"].index.hour) == {0, 8, 16}
    assert not x["spot_ok"].loc[C.DATA_START].any()             # no spot close in the window yet
    assert x["liquid"].loc[C.DATA_START + pd.Timedelta("31D")].all()
    assert len(C.fingerprint(x)) == 64


# ── decision rules ───────────────────────────────────────────────────────────

def _good():
    m = {"sharpe": 2.0, "alpha_t": 3.0, "alpha_yr": 0.30}
    return {"is": dict(m), "oos": dict(m), "oos_2x": dict(m), "oos_borrow": dict(m),
            "oos_h1": dict(m), "oos_h2": dict(m), "skipped_oos": 0.0, "drop_one_min_alpha": 0.05}


def _rows(**override):
    bad = {"sharpe": -1.0, "alpha_t": -1.0, "alpha_yr": -0.1}
    rows = {c.name: {"is": bad, "oos": bad, "oos_2x": bad, "oos_borrow": bad, "oos_h1": bad,
                     "oos_h2": bad, "skipped_oos": 0.0} for c in C.grid()}
    rows["fund_lvl_so_k3_24h"] = _good()
    rows["fund_lvl_so_k5_24h"] = _good()                         # a stable neighbour
    for k, v in override.items():
        rows["fund_lvl_so_k3_24h"][k] = v
    return rows


def test_verdict_passes_only_when_every_rule_holds():
    v = C.verdict(_rows())
    assert v["fund_lvl_so_k3_24h"]["verdict"].startswith("PASSED")
    assert v["prem_lvl_ls_k3_8h"]["verdict"] == "rejected"
    assert v["prem_lvl_ls_k3_8h"]["failed"][:2] == ["is_sharpe", "is_alpha_t"]   # OOS failures listed too
    assert "drop_one" not in v["prem_lvl_ls_k3_8h"]["failed"]                     # never run for it
    assert set(v) == {c.name for c in C.grid()}


@pytest.mark.parametrize("override,check", [
    ({"oos": {"sharpe": 2.0, "alpha_t": 3.0, "alpha_yr": 0.09}}, "oos_alpha_min"),     # < 10%/yr
    ({"oos_h2": {"sharpe": 2.0, "alpha_t": 3.0, "alpha_yr": -0.01}}, "oos_h2_alpha"),
    ({"oos_borrow": {"sharpe": 0.4, "alpha_t": 3.0, "alpha_yr": 0.05}}, "borrow_sharpe"),
    ({"oos_borrow": {"sharpe": 0.6, "alpha_t": 3.0, "alpha_yr": -0.01}}, "borrow_alpha"),
    ({"oos_2x": {"sharpe": 0.49}}, "oos_sharpe_2x"),
    ({"skipped_oos": 0.06}, "coverage"),
    ({"drop_one_min_alpha": -0.01}, "drop_one"),
    ({"oos": {"sharpe": 0.74, "alpha_t": 3.0, "alpha_yr": 0.30}}, "oos_sharpe"),
])
def test_verdict_each_new_and_kept_rule_can_fail_a_candidate(override, check):
    v = C.verdict(_rows(**override))["fund_lvl_so_k3_24h"]
    assert v["verdict"] == "candidate (failed OOS)" and check in v["failed"]


def test_verdict_neighbour_needs_is_sharpe_and_oos_alpha():
    rows = _rows()
    rows["fund_lvl_so_k5_24h"]["oos"] = {"sharpe": 2.0, "alpha_t": 3.0, "alpha_yr": -0.01}
    v = C.verdict(rows)["fund_lvl_so_k3_24h"]
    assert v["verdict"] == "candidate (failed OOS)" and v["failed"] == ["neighbour"]
    nan = _rows()
    nan["fund_lvl_so_k3_24h"]["is"] = {"sharpe": float("nan"), "alpha_t": 3.0}
    assert C.verdict(nan)["fund_lvl_so_k3_24h"]["verdict"] == "rejected"
