"""
End-to-end plumbing test for research.study on synthetic random-walk data
(no network, no real results): every pre-registered strategy runs, gets IS /
OOS metrics and a verdict, and outputs are written. On a random walk nothing
should be CONFIRMED.
"""

import json

import numpy as np
import pandas as pd

import research.study as study


def _synthetic_panel(symbols, interval, start, end=None):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-10-01 01:00", "2025-12-31", freq="1h", tz="UTC")
    n = len(symbols)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.006, (len(idx), n)), axis=0)),
                         index=idx, columns=symbols)
    close.iloc[: 24 * 200, -1] = np.nan                 # one late listing
    spread = close * 0.002
    return {"open": close.shift(1), "high": close + spread, "low": close - spread,
            "close": close, "qv": close * 0 + 1e6}       # $24M/day each


def _synthetic_funding(symbol, start, end=None):
    rng = np.random.default_rng(sum(symbol.encode()))
    idx = pd.date_range("2022-10-01", "2025-12-31", freq="8h", tz="UTC")
    return pd.Series(rng.normal(0.00005, 0.0001, len(idx)), index=idx)


def test_study_runs_end_to_end_on_synthetic_data(monkeypatch, tmp_path, capsys):
    syms = ["BTCUSDT"] + [f"S{i}USDT" for i in range(13)]
    monkeypatch.setattr(study, "SYMBOLS", syms)
    monkeypatch.setattr(study, "build_panel", _synthetic_panel)
    monkeypatch.setattr(study, "load_funding", _synthetic_funding)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    study.main()
    out = next((tmp_path / "results").iterdir())
    summary = json.loads((out / "summary.json").read_text())
    names = {s.name for s in study.specs()}
    assert set(summary["verdict"]) == names
    assert "CONFIRMED" not in summary["verdict"].values()
    for n in names | {"bench_ew", "bench_btc"}:
        assert "sharpe" in summary["metrics"][n]["is"] and "sharpe" in summary["metrics"][n]["oos"]
    eq = pd.read_csv(out / "daily_equity.csv", index_col=0)
    assert set(eq.columns) == names | {"bench_ew", "bench_btc"}
    assert "verdict" in capsys.readouterr().out
