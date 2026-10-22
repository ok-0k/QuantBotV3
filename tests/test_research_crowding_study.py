"""
End-to-end test of the crowding study runner (research/crowding_study.py) on
SYNTHETIC data only: no real cache file is read and no real result computed.

The synthetic market plants a crowding edge, so the whole decision path runs:
- four "crowded" coins carry high funding and drift down;
- premium is pure noise;
- a 16h outage on 9 of the 12 coins forces skipped rebalances.
The timeline is compressed (the real one is 35k hourly bars per run, too slow
for the suite). The real window constants are pinned separately below.
"""

import json

import numpy as np
import pandas as pd
import pytest

import research.crowding as C
import research.crowding_study as CS
from research.data import utc

SYMS = ["BTCUSDT"] + [f"S{i:02d}USDT" for i in range(1, 12)]
T = {"DATA_START": utc(2025, 1, 1), "IS_START": utc(2025, 2, 5), "OOS_START": utc(2025, 3, 15),
     "OOS_SPLIT": utc(2025, 4, 20), "END": utc(2025, 5, 26)}


def _outage_start():
    """A 16:00 label in OOS H2 whose next 00:00 is a 72h decision day."""
    for a in pd.date_range(T["OOS_SPLIT"] + pd.Timedelta("5D16h"), T["END"], freq="1D"):
        if ((a + pd.Timedelta("8h")) - C.EPOCH).days % 3 == 0:
            return a
    raise AssertionError


CROWDED = {"S03USDT": 1, "S05USDT": 1, "S07USDT": 1, "S09USDT": 1}


def _market(seed=0, mu=0.002, crowded=CROWDED, idio=0.003):
    """Funding, premium and hourly closes. The coins in `crowded` carry high
    funding and drift down by mu per hour for the whole sample."""
    rng = np.random.default_rng(seed)
    g = pd.date_range(T["DATA_START"], T["END"] + pd.Timedelta("2D"), freq="8h")
    state = pd.DataFrame({s: float(crowded.get(s, 0)) for s in SYMS}, index=g)
    fund = (1e-4 + 4e-4 * state + rng.normal(0, 2e-5, state.shape)).round(8)
    prem = pd.DataFrame(rng.normal(0, 5e-4, state.shape), index=g, columns=SYMS)
    h = pd.date_range(T["DATA_START"] + pd.Timedelta("1h"), T["END"] + pd.Timedelta("2D"), freq="1h")
    st_h = state.reindex(h.ceil("8h")).to_numpy()               # state of the window holding each bar
    lr = rng.normal(0, 0.004, (len(h), 1)) + rng.normal(0, idio, (len(h), len(SYMS))) - mu * st_h
    close = pd.DataFrame(100 * np.exp(np.cumsum(lr, axis=0)), index=h, columns=SYMS)
    a = _outage_start()
    close.loc[(close.index > a) & (close.index <= a + pd.Timedelta("16h")), SYMS[3:]] = np.nan
    return fund, prem, close


@pytest.fixture
def synthetic(monkeypatch):
    def install(**kw):
        fund, prem, close = _market(**kw)
        monkeypatch.setattr(C.D, "OFFLINE", True)
        monkeypatch.setattr(C.D, "build_panel", lambda symbols, interval, start, end=None: {
            "open": close[symbols], "high": close[symbols], "low": close[symbols], "close": close[symbols],
            "qv": close[symbols] * 0 + 1e6})
        monkeypatch.setattr(C.D, "load_funding",
                            lambda s, start, end=None: fund[s].set_axis(fund.index + pd.Timedelta("8ms")))
        monkeypatch.setattr(C.D, "load_premium", lambda s, start, end=None: prem[s])
        monkeypatch.setattr(CS, "SYMBOLS", SYMS)
        for k in ("DATA_START", "END", "OOS_SPLIT"):
            monkeypatch.setattr(C, k, T[k])
        for k in ("IS_START", "OOS_START"):
            monkeypatch.setattr(CS, k, T[k])
        return C.load_inputs(SYMS)
    return install


def test_real_windows_are_the_preregistered_ones():
    w = CS.windows()
    ns = pd.Timedelta("1ns")
    assert w["is"] == (utc(2023, 1, 1), utc(2025, 4, 1) - ns)
    assert w["oos"] == (utc(2025, 4, 1), utc(2026, 10, 7))
    assert w["oos_h1"] == (utc(2025, 4, 1), utc(2026, 1, 1) - ns)       # split 2026-01-01
    assert w["oos_h2"] == (utc(2026, 1, 1), utc(2026, 10, 7))
    assert C.END == pd.Timestamp("2026-10-07 00:00", tz="UTC")
    assert set(CS.CHECKS) == {"is_sharpe", "is_alpha_t", "oos_sharpe", "oos_alpha_pos", "oos_sharpe_2x",
                              "neighbour", "oos_alpha_min", "oos_h1_alpha", "oos_h2_alpha",
                              "borrow_sharpe", "borrow_alpha", "coverage", "drop_one"}


def test_study_end_to_end_on_synthetic_data(synthetic, tmp_path, capsys):
    synthetic()
    CS.main(["--out", str(tmp_path)])
    s = json.loads((tmp_path / "summary.json").read_text())
    names = [c.name for c in C.grid()]
    assert set(s["verdict"]) == set(names) == set(s["configs"]) == set(s["checks"])
    assert s["meta"]["end"] == str(T["END"]) and len(s["meta"]["fingerprint"]) == 64

    # every field the decision rules read, for every config
    for n in names:
        r = s["configs"][n]
        for key in ("is", "oos", "oos_h1", "oos_h2", "oos_2x", "oos_borrow", "oos_zero"):
            assert {"sharpe", "alpha_yr", "alpha_t", "cost_yr", "years"} <= set(r[key]), (n, key)
        assert 0 <= r["skipped_oos"] <= 1 and set(s["checks"][n]) == set(CS.CHECKS)
        assert set(s["checks"][n].values()) <= {"pass", "FAIL", "n/a"}
        # OOS halves: their own windows, split at OOS_SPLIT (36 + 36 days of 1h bars)
        assert r["oos_h1"]["years"] == pytest.approx(36 / 365, abs=0.006)
        assert r["oos_h2"]["years"] == pytest.approx(36 / 365, abs=0.006)
        assert r["oos"]["years"] == pytest.approx(72 / 365, abs=0.006)
        # cost cases: 2x and borrow stress cost more, zero cost costs nothing
        o, o2, ob, oz = r["oos"], r["oos_2x"], r["oos_borrow"], r["oos_zero"]
        assert oz["cost_yr"] == 0 and o2["cost_yr"] > o["cost_yr"] > 0
        assert ob["cost_yr"] > o["cost_yr"]
        assert ob["gross_return_yr"] == pytest.approx(o["gross_return_yr"], rel=0.05, abs=0.05)

    # skipped-rebalance accounting: the outage kills 2 8h decisions, 1 at 24h and 72h
    per = {"8h": 3, "24h": 1, "72h": 1 / 3}
    for n in names:
        cov, p = s["configs"][n]["coverage"], n.rsplit("_", 1)[1]
        assert cov["oos_skipped"] == {"8h": 2, "24h": 1, "72h": 1}[p], n
        assert cov["oos_decisions"] == pytest.approx(72 * per[p], abs=1), n
        assert s["configs"][n]["skipped_oos"] == pytest.approx(cov["oos_skipped"] / cov["oos_decisions"])
        assert cov["oos_executed"] == cov["oos_decisions"] - cov["oos_skipped"]
        assert cov["is_skipped"] == 0 and cov["oos_eligible_min"] == 3

    # the planted edge passes; pure-noise premium does not
    v = s["verdict"]
    passed = [n for n in names if v[n].startswith("PASSED")]
    assert passed and all(n.startswith("fund_lvl") for n in passed)
    assert all(not v[n].startswith("PASSED") for n in names if n.startswith("prem_"))
    for n in passed:
        r = s["configs"][n]
        assert s["checks"][n] == {c: "pass" for c in CS.CHECKS}
        assert len(r["drop_one"]) >= 3 and r["drop_one_min_alpha"] == min(r["drop_one"].values()) > 0
        assert set(r["phases"]) == {"8h": set(), "24h": {"24h@08:00", "24h@16:00"},
                                    "72h": {"72h@day%3==1", "72h@day%3==2"}}[n.rsplit("_", 1)[1]]
    for n in names:
        if not v[n].startswith("PASSED"):
            if s["failed"][n] == ["drop_one"]:            # ran and failed
                assert s["configs"][n]["drop_one_min_alpha"] <= 0
            else:                                         # never reached the drop-one stage
                assert s["failed"][n] and "drop_one" not in s["configs"][n]
            assert "phases" not in s["configs"][n]
            if v[n] == "rejected":
                assert s["checks"][n]["drop_one"] == "n/a"
    for ref in ("bench_ew", "bench_btc", "ref_short_all_8h", "ref_short_all_24h", "ref_short_all_72h"):
        assert "sharpe" in s["references"][ref]["oos"]

    text = (tmp_path / "report.txt").read_text()
    assert all(n in text for n in names) and "PASSED" in text
    eq = pd.read_csv(tmp_path / "daily_equity.csv", index_col=0)
    assert set(names) <= set(eq.columns) and "ref_short_all_24h" in eq.columns
    assert "verdict" in capsys.readouterr().out


def test_drop_one_catches_an_edge_that_lives_in_one_symbol(synthetic):
    """The MATIC lesson: an 'edge' that is one symbol's story must not pass."""
    x = synthetic(seed=3, crowded={"S05USDT": 1}, idio=0.0005)
    sig, elig = CS.books(x)
    cfg = C.Config("fund_lvl", "so", 3, "8h")
    r = CS.run_config(cfg, sig, elig, x["p1h"]["close"], x["bench"], cases=("base",))
    assert r["row"]["oos"]["alpha_yr"] > 0.10                     # looks great with S05...
    syms = CS.oos_symbols(r["W"], r["ta"])
    assert "S05USDT" in syms
    d = CS.drop_one(cfg, x, syms)
    assert set(d) == set(syms)
    assert d["S05USDT"] < 0 < max(d.values())                   # ...and is nothing without it
    # the dropped symbol is gone from every input but the benchmark
    sub_sig, _ = CS.books({"p1h": {k: v.drop(columns="S05USDT") for k, v in x["p1h"].items()},
                           **{k: x[k].drop(columns="S05USDT") for k in ("F", "P", "spot_ok", "liquid")}})
    assert "S05USDT" not in sub_sig["fund_lvl"].columns


def test_held_targets_and_oos_symbols(monkeypatch):
    idx = pd.date_range(T["OOS_START"] - pd.Timedelta("3h"), periods=8, freq="1h")
    W = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    ta = pd.Series(False, index=idx)
    W.iloc[1] = [-1.0, 0.0]; ta.iloc[1] = True                   # IS fill: A short
    W.iloc[5] = [0.0, -1.0]; ta.iloc[5] = True                   # OOS fill: B short, A closed
    monkeypatch.setattr(CS, "OOS_START", T["OOS_START"])
    assert CS.held_targets(W, ta)["A"].tolist() == [0, -1, -1, -1, -1, 0, 0, 0]
    assert CS.oos_symbols(W, ta) == ["B"]                         # A only had an IS target
