"""
Data-only pass for research.study: download/refresh the cache and report data
quality (coverage, gaps, incomplete days). Computes NO strategy results, so
data-handling choices can be settled before the study is ever run.

    .venv/bin/python -m research.data_check
"""

import sys

import numpy as np
import pandas as pd

from config import SYMBOLS
from research.data import build_panel, load_funding, resample
from research.study import DATA_START


def main() -> None:
    p = build_panel(SYMBOLS, "1h", DATA_START)
    c = p["close"]
    print(f"hourly panel {c.shape[0]} bars x {c.shape[1]} symbols, {c.index.min()} .. {c.index.max()}")
    missing = sorted(set(SYMBOLS) - set(c.columns))
    print("no spot data at all:", missing or "none")
    rows = []
    for s in c.columns:
        col = c[s]
        live = col.loc[col.first_valid_index():col.last_valid_index()]
        rows.append((s, col.first_valid_index().date(), col.last_valid_index().date(),
                     int(live.isna().sum()), len(live)))
    gaps = [r for r in rows if r[3] > 0]
    print(f"symbols with interior hourly gaps: {len(gaps)}")
    for s, a, b, g, n in sorted(gaps, key=lambda r: -r[3])[:8]:
        print(f"   {s:10} {a} .. {b}  missing {g} of {n} hours")
    late = [(s, a) for s, a, *_ in rows if a > DATA_START.date()]
    print("listed after data start:", ", ".join(f"{s} {a}" for s, a in late) or "none")
    ended = [(s, b) for s, _, b, *_ in rows if b < c.index.max().date()]
    print("history ends early (delisted):", ", ".join(f"{s} {b}" for s, b in ended) or "none")
    # whole-market outages: hours where most listed symbols are missing
    listed = c.notna().cumsum() > 0
    frac_missing = (c.isna() & listed).sum(axis=1) / listed.sum(axis=1).clip(lower=1)
    outage = frac_missing[frac_missing > 0.5]
    print(f"market-wide missing hours (>50% of listed symbols): {len(outage)}"
          + (f" e.g. {', '.join(str(t) for t in outage.index[:5])}" if len(outage) else ""))
    d = resample(p, "1D")["close"]
    dl = d.notna().cumsum() > 0
    inc = (d.isna() & dl & (d.index <= d.index.max())).sum().sum()
    print(f"daily bars dropped as incomplete (after listing): {inc} of {int(dl.sum().sum())}")
    nf = 0
    for s in SYMBOLS:
        try:
            f = load_funding(s, DATA_START)
            nf += len(f) > 0
        except Exception as exc:
            print(f"   funding {s}: {exc}", file=sys.stderr)
    print(f"symbols with perp funding history: {nf} of {len(SYMBOLS)}")


if __name__ == "__main__":
    main()
