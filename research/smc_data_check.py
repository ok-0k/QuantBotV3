"""
Data-only pass for the SMC playbook study (research/PREREG_SMC.md):
download the 5m spot klines and report coverage. Computes NO strategy
statistic (no returns, trades or signals), so data handling can be settled
before the study runs.

    TRADING_ENV_FILE=/nonexistent .venv/bin/python -m research.smc_data_check --download --workers 3
    TRADING_ENV_FILE=/nonexistent RESEARCH_OFFLINE=1 .venv/bin/python -m research.smc_data_check

Downloads go through research.data.load_klines (throttled ~4 requests/s per
worker, cache ROOT/klines_5m/<symbol>.npz), in 180-day chunks so an
interruption loses at most one chunk.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research.data import ROOT, load_klines, utc

DATA_START = utc(2022, 10, 1)
END = utc(2026, 10, 7)                     # hard END of the study (exclusive bar open)
STEP_MS = 300_000


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _download_one(symbol: str) -> tuple[str, int, float]:
    t0 = time.time()
    edge = DATA_START
    n = 0
    while edge < END:
        edge = min(edge + pd.Timedelta(days=180), END)
        n = load_klines(symbol, "5m", DATA_START, edge)["t"].size
    _log(f"{symbol:11} {n:7d} bars  {time.time() - t0:6.0f}s")
    return symbol, n, time.time() - t0


def download(symbols, workers: int) -> None:
    t0 = time.time()
    with Pool(workers) as pool:
        pool.map(_download_one, symbols, chunksize=1)
    _log(f"download done in {time.time() - t0:.0f}s")


def coverage(symbols) -> pd.DataFrame:
    """Per symbol, within [DATA_START, END): first/last bar, bars present,
    missing bars between first and last, number and longest of gaps."""
    rows = []
    for s in symbols:
        k = load_klines(s, "5m", DATA_START, END)
        t = k["t"]
        if t.size == 0:
            rows.append({"symbol": s, "bars": 0})
            continue
        d = np.diff(t) // STEP_MS
        span = int((t[-1] - t[0]) // STEP_MS) + 1
        gaps = d[d > 1] - 1
        rows.append({
            "symbol": s,
            "first": pd.Timestamp(int(t[0]), unit="ms", tz="UTC"),
            "last": pd.Timestamp(int(t[-1]), unit="ms", tz="UTC"),
            "bars": int(t.size),
            "missing": span - int(t.size),
            "gaps": int(gaps.size),
            "longest_gap_bars": int(gaps.max()) if gaps.size else 0,
            "zero_volume_bars": int((k["v"] == 0).sum()),
        })
    return pd.DataFrame(rows).set_index("symbol")


def market_outages(symbols) -> pd.Series:
    """5m bar opens where more than half of the symbols listed at that time
    have no bar (exchange-wide outages)."""
    have = {}
    for s in symbols:
        t = load_klines(s, "5m", DATA_START, END)["t"]
        if t.size:
            have[s] = pd.Series(True, index=t)
    grid = np.arange(int(DATA_START.timestamp() * 1000), int(END.timestamp() * 1000), STEP_MS)
    present = pd.DataFrame(have).reindex(grid).notna()
    listed = present.cumsum().gt(0) & present[::-1].cumsum()[::-1].gt(0)
    frac = (listed & ~present).sum(axis=1) / listed.sum(axis=1).clip(lower=1)
    out = frac[frac > 0.5]
    out.index = pd.to_datetime(out.index, unit="ms", utc=True)
    return out


def main(argv=None) -> None:
    from research.universe import RESEARCH_SYMBOLS as SYMBOLS
    ap = argparse.ArgumentParser(prog="research.smc_data_check")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if a.download:
        download(SYMBOLS, a.workers)
    cov = coverage(SYMBOLS)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(cov.to_string())
    print(f"\ntotal bars {int(cov['bars'].sum())}, symbols with interior gaps {int((cov['gaps'] > 0).sum())}")
    out = market_outages(SYMBOLS)
    print(f"market-wide missing 5m bars (>50% of listed symbols): {len(out)}")
    if len(out):
        runs = (out.index.to_series().diff() != pd.Timedelta("5min")).cumsum()
        for _, g in out.groupby(runs.values):
            print(f"   {g.index[0]} .. {g.index[-1]}  ({len(g)} bars)")
    files = sorted((ROOT / "klines_5m").glob("*.npz"))
    print(f"cache: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB in {ROOT / 'klines_5m'}")


if __name__ == "__main__":
    main()
