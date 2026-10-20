"""
End-to-end test of the SMC study runner (research/smc_study.py) on SYNTHETIC
data only: smc.load_symbol is monkeypatched, so no real market data is read.
Covers the full grid x all sides x both scopes x base/2x costs, verdicts
(short decides; portfolio input = short OOS base), the sanity counters and
the freeze, the refusals, the output files, and that no statistic is printed
before every result file is written.
"""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from research import data
from research import smc
from research import smc_study as S

SYMS = ["BTCUSDT", "ETHUSDT", "XUSDT"]
_BARS: dict = {}
_PREP: dict = {}


def _synthetic(sym: str) -> smc.Bars:
    """Fat-tailed 5m walk on the full study grid [DATA_START, END), more
    volatile 09:30-11:00 NY, prices on a 0.01 tick."""
    if sym not in _BARS:
        rng = np.random.default_rng(sum(map(ord, sym)))
        s, e = int(smc.DATA_START.timestamp() * 1000), int(smc.END.timestamp() * 1000)
        t = np.arange(s, e, smc.BASE_MS, dtype=np.int64)
        _, minute = smc.ny_day_minute(t)
        base = {"BTCUSDT": 300.0, "ETHUSDT": 150.0}.get(sym, 80.0)
        vol = 0.0025 * np.where((minute >= 570) & (minute < 660), 2.0, 1.0)
        c = np.round(base * np.exp(np.cumsum(rng.standard_t(3, t.size) * 0.6 * vol)), 2)
        o = np.r_[base, c[:-1]]
        h = np.round(np.maximum(o, c) + np.round(np.abs(rng.normal(0, 0.5, t.size)) * vol * c, 2), 2)
        lo = np.round(np.minimum(o, c) - np.round(np.abs(rng.normal(0, 0.5, t.size)) * vol * c, 2), 2)
        _BARS[sym] = smc.Bars(t, o, h, lo, c, 5, np.full(t.size, 1e5))
    return _BARS[sym]


_real_prepare = smc.prepare


def _memo_prepare(sym, b5, days):
    """smc.prepare is deterministic; reuse it across the runs of this module."""
    if sym not in _PREP:
        _PREP[sym] = _real_prepare(sym, b5, days)
    return _PREP[sym]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "universe", lambda: list(SYMS))
    monkeypatch.setattr(smc, "load_symbol", _synthetic)
    monkeypatch.setattr(smc, "prepare", _memo_prepare)
    monkeypatch.setattr(data, "OFFLINE", True)
    monkeypatch.setenv("RESEARCH_OFFLINE", "1")
    monkeypatch.setattr(S, "_git", lambda *a: "" if a[0] == "status" else "0123abcd\n")
    return tmp_path


def _capture_run(monkeypatch):
    box = {}
    real = S.run

    def spy(commit):
        box["res"] = real(commit)
        return box["res"]

    monkeypatch.setattr(S, "run", spy)
    calls = []
    real_verdict = smc.verdict
    monkeypatch.setattr(smc, "verdict", lambda b, s, p: calls.append((b, s, p)) or real_verdict(b, s, p))
    return box, calls


def test_runner_end_to_end(env, monkeypatch, capsys):
    box, calls = _capture_run(monkeypatch)
    out = env / "run"
    assert S.main(["--out", str(out)]) == 0
    files = {p.name for p in out.iterdir()}
    assert files == {"command.txt", "stdout.txt", "stderr.txt", "exit_code.txt", "summary.json", "report.txt",
                     "trades.csv.gz", "setups.csv.gz"}
    assert (out / "exit_code.txt").read_text().strip() == "0"
    assert "commit 0123abcd" in (out / "command.txt").read_text()
    summ = json.loads((out / "summary.json").read_text())
    res = box["res"]
    # meta: commit, fingerprint of exactly the data used, END
    assert summ["meta"]["commit"] == "0123abcd" and summ["meta"]["end"] == str(smc.END)
    assert summ["meta"]["fingerprint"] == smc.fingerprint({s: _synthetic(s) for s in SYMS})
    assert summ["meta"]["decision_sides"] == ["short"] and summ["meta"]["trading_days"] == 943
    # every config x side x scope x cost x period
    names = [c.name for c in smc.grid()]
    assert sorted(summ["runs"]) == sorted(names)
    for n in names:
        assert set(summ["runs"][n]) == {"long", "short", "both"}
        for sd in ("long", "short", "both"):
            assert set(summ["runs"][n][sd]) == {"symbol", "portfolio"}
            for sc in ("symbol", "portfolio"):
                assert set(summ["runs"][n][sd][sc]) == {"base", "2x"}
                for ck in ("base", "2x"):
                    r = summ["runs"][n][sd][sc][ck]
                    assert set(r["periods"]) == {"IS", "OOS", "OOS_H1", "OOS_H2"}
                    assert set(r["counters"]) == set(S.COUNTERS) and all(v == 0 for v in r["counters"].values())
    # sanity counters all 0 -> not frozen
    assert summ["sanity"] == {"totals": {k: 0 for k in S.COUNTERS}, "frozen": False}
    # verdicts: 8 x 3, only short decides; port_exp_pos uses the same side's portfolio OOS at base cost
    assert len(calls) == 24
    k = 0
    for cfg in smc.grid():
        for sd in smc.SIDES:
            v = summ["verdicts"][cfg.name][sd]
            assert v["decisional"] == (sd == "short")
            assert v["verdict"] == v["computed"] if sd == "short" else v["verdict"].startswith("informational")
            b, st, p = calls[k]
            run = res["runs"][cfg.name][sd]
            assert b is run["symbol"]["base"]["periods"] and st is run["symbol"]["2x"]["periods"]["OOS"]
            assert p is run["portfolio"]["base"]["periods"]["OOS"]
            k += 1
    # trades and setups
    with gzip.open(out / "trades.csv.gz", "rt") as fh:
        tr = pd.read_csv(fh)
    with gzip.open(out / "setups.csv.gz", "rt") as fh:
        su = pd.read_csv(fh)
    assert len(su) > 0 and len(tr) > 0
    assert set(tr["period"]) <= {"IS", "OOS_H1", "OOS_H2"}
    assert set(tr["scope"]) == {"symbol", "portfolio"} and set(tr["cost"]) == {"base", "2x"}
    assert set(tr["config"]) <= set(names) and set(tr["sides"]) <= {"long", "short", "both"}
    n_base = summ["runs"][names[3]]["short"]["symbol"]["base"]["periods"]
    sel = tr[(tr.config == names[3]) & (tr.sides == "short") & (tr.scope == "symbol") & (tr.cost == "base")]
    assert len(sel) == n_base["IS"]["n"] + n_base["OOS"]["n"]
    assert (tr[tr.sides == "short"]["side"] == -1).all() and (tr[tr.sides == "long"]["side"] == 1).all()
    # the report is printed only at the end, and is the same text as report.txt
    printed = capsys.readouterr().out
    assert printed.strip() == (out / "report.txt").read_text().strip()
    assert (out / "stdout.txt").read_text().strip() == printed.strip()
    assert "DECISION UNITS" in printed and "SANITY COUNTERS (must all be 0)" in printed
    err = (out / "stderr.txt").read_text()
    assert "Traceback" not in err and "outputs written" in err


def test_no_statistic_is_printed_if_writing_fails(env, monkeypatch, capsys):
    def boom(out, res):
        raise OSError("disk full")

    monkeypatch.setattr(S, "write_outputs", boom)
    out = env / "run"
    assert S.main(["--out", str(out)]) == 1
    assert (out / "exit_code.txt").read_text().strip() == "1"
    assert capsys.readouterr().out == "" and (out / "stdout.txt").read_text() == ""
    assert "disk full" in (out / "stderr.txt").read_text()


def test_any_nonzero_sanity_counter_freezes_every_verdict(env, monkeypatch):
    real_walk = smc.walk

    def walk(*a, **k):
        tr = real_walk(*a, **k)
        if k.get("stats") is not None and a[2].name == "smc_15m_tp2_nobias" and a[3] == (1,):
            k["stats"]["two_loss_binds"] = k["stats"].get("two_loss_binds", 0) + 1
        return tr

    monkeypatch.setattr(smc, "walk", walk)
    out = env / "run"
    assert S.main(["--out", str(out)]) == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["sanity"]["frozen"] and summ["sanity"]["totals"]["two_loss_binds"] == 4   # 2 scopes x 2 costs
    assert all(v["verdict"].startswith("FROZEN") for c in summ["verdicts"].values() for v in c.values())
    assert "VERDICTS FROZEN" in (out / "report.txt").read_text()


def test_refusals(env, monkeypatch, tmp_path):
    out = env / "never"
    monkeypatch.setattr(S, "_git", lambda *a: " M research/smc.py\n" if a[0] == "status" else "0123abcd\n")
    with pytest.raises(SystemExit, match="dirty"):
        S.main(["--out", str(out)])
    monkeypatch.setattr(S, "_git", lambda *a: "" if a[0] == "status" else "0123abcd\n")
    monkeypatch.delenv("RESEARCH_OFFLINE")
    with pytest.raises(SystemExit, match="RESEARCH_OFFLINE"):
        S.main(["--out", str(out)])
    monkeypatch.setenv("RESEARCH_OFFLINE", "1")
    monkeypatch.setattr(data, "OFFLINE", False)                 # env set after research.data was imported
    with pytest.raises(SystemExit, match="RESEARCH_OFFLINE"):
        S.main(["--out", str(out)])
    monkeypatch.setattr(data, "OFFLINE", True)
    secrets = tmp_path / "env"
    secrets.write_text("X=1\n")
    monkeypatch.setenv("TRADING_ENV_FILE", str(secrets))
    with pytest.raises(SystemExit, match="TRADING_ENV_FILE"):
        S.main(["--out", str(out)])
    assert not out.exists()
    monkeypatch.setenv("TRADING_ENV_FILE", str(tmp_path / "missing"))
    out.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        S.main(["--out", str(out)])


def test_streaming_fingerprint_matches_smc_fingerprint():
    import hashlib
    bars = {s: _synthetic(s) for s in SYMS}
    h = hashlib.sha256()
    for s in sorted(bars):
        smc.fingerprint_update(h, s, bars[s])
    assert h.hexdigest() == smc.fingerprint(bars)
