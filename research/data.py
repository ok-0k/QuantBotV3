"""
Binance public market data with an on-disk cache.

- Spot klines  (api.binance.com/api/v3/klines)       -> load_klines()
- Perp funding (fapi.binance.com/fapi/v1/fundingRate) -> load_funding()
- Aligned multi-symbol panels                         -> build_panel(), resample()

Requests are throttled (~4/s, far below Binance's weight limits) and retried
with backoff, so a study run never competes with the live bot for rate limit.
Panels are indexed by bar CLOSE time (UTC): the row labelled T holds the bar
covering (T - interval, T], so a signal computed at row T only uses data
known at T.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.getenv("RESEARCH_DATA_DIR", "/home/admin/trading_data/research"))
SPOT_KLINES = "https://api.binance.com/api/v3/klines"
PERP_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
INTERVAL_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
_THROTTLE_S = 0.25
_FIELDS = ("t", "o", "h", "l", "c", "v", "qv")


def to_ms(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _get(url: str, retries: int = 5):
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 400:                      # unknown / delisted symbol
                return None
            if e.code in (418, 429) or e.code >= 500:
                wait = float(e.headers.get("Retry-After") or delay)
                time.sleep(wait)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {url}")


def _fetch_klines(symbol: str, interval: str, start: int, end: int) -> dict:
    step = INTERVAL_MS[interval]
    out = {k: [] for k in _FIELDS}
    t = start
    while t < end:
        rows = _get(f"{SPOT_KLINES}?symbol={symbol}&interval={interval}&startTime={t}&endTime={end - 1}&limit=1000")
        time.sleep(_THROTTLE_S)
        if not rows:
            break
        for k in rows:
            out["t"].append(int(k[0]))
            for name, idx in (("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5), ("qv", 7)):
                out[name].append(float(k[idx]))
        nxt = int(rows[-1][0]) + step
        if nxt <= t:
            break
        t = nxt
    return out


def load_klines(symbol: str, interval: str, start, end=None) -> dict[str, np.ndarray]:
    """Spot klines for [start, end) as arrays t(open ms), o, h, l, c, v, qv.

    Cached in ROOT/klines_<interval>/<symbol>.npz; only missing head/tail
    ranges are fetched. Bars still forming (open + interval > now) are dropped.
    """
    step = INTERVAL_MS[interval]
    now = int(time.time() * 1000)
    start = to_ms(start) // step * step
    end = min(to_ms(end) if end is not None else now, now // step * step)
    path = ROOT / f"klines_{interval}" / f"{symbol}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    have = dict(np.load(path)) if path.exists() else {k: np.array([]) for k in _FIELDS}
    chunks = []
    if have["t"].size == 0:
        chunks.append(_fetch_klines(symbol, interval, start, end))
    else:
        first, last = int(have["t"][0]), int(have["t"][-1])
        if start < first:
            chunks.append(_fetch_klines(symbol, interval, start, first))
        if last + step < end:
            chunks.append(_fetch_klines(symbol, interval, last + step, end))
    if any(len(ch["t"]) for ch in chunks):
        merged = {k: np.concatenate([have[k]] + [np.asarray(ch[k]) for ch in chunks]) for k in _FIELDS}
        merged["t"] = merged["t"].astype(np.int64)
        _, keep = np.unique(merged["t"], return_index=True)
        merged = {k: v[keep] for k, v in merged.items()}
        merged = {k: v[merged["t"] + step <= now] for k, v in merged.items()}
        np.savez(path, **merged)
        have = merged
    sel = (have["t"] >= start) & (have["t"] < end)
    return {k: np.asarray(v)[sel] for k, v in have.items()}


def load_funding(symbol: str, start, end=None) -> pd.Series:
    """USDT-M perpetual funding rates (per 8h settlement), indexed by UTC time."""
    start, end = to_ms(start), to_ms(end) if end is not None else int(time.time() * 1000)
    path = ROOT / "funding" / f"{symbol}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    have: dict[int, float] = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
    t = max(have) + 1 if have else start
    if have and min(have) > start + 8 * 3_600_000:
        t = start                                   # need older history too
    while t < end:
        rows = _get(f"{PERP_FUNDING}?symbol={symbol}&startTime={t}&endTime={end}&limit=1000")
        time.sleep(_THROTTLE_S)
        if not rows:
            break
        for r in rows:
            have[int(r["fundingTime"])] = float(r["fundingRate"])
        nxt = int(rows[-1]["fundingTime"]) + 1
        if nxt <= t:
            break
        t = nxt
    path.write_text(json.dumps({str(k): v for k, v in sorted(have.items())}))
    s = pd.Series(have, dtype=float).sort_index()
    s = s[(s.index >= start) & (s.index < end)]
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s


def build_panel(symbols, interval: str, start, end=None) -> dict[str, pd.DataFrame]:
    """Aligned OHLC + quote-volume panel (DataFrames time x symbol), close-time index.

    Missing bars (pre-listing, post-delisting, gaps) are NaN.
    """
    step = INTERVAL_MS[interval]
    frames: dict[str, dict[str, pd.Series]] = {k: {} for k in ("open", "high", "low", "close", "qv")}
    for sym in symbols:
        k = load_klines(sym, interval, start, end)
        if k["t"].size == 0:
            continue
        idx = pd.to_datetime(k["t"] + step, unit="ms", utc=True)
        for name, field in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("qv", "qv")):
            frames[name][sym] = pd.Series(k[field], index=idx)
    out = {name: pd.DataFrame(cols).sort_index() for name, cols in frames.items()}
    full = pd.date_range(out["close"].index.min(), out["close"].index.max(), freq=pd.Timedelta(milliseconds=step))
    return {name: df.reindex(full) for name, df in out.items()}


def resample(panel: dict[str, pd.DataFrame], rule: str, min_frac: float = 0.9) -> dict[str, pd.DataFrame]:
    """Coarsen a close-time-indexed panel. A coarse bar is kept only if at
    least min_frac of its fine bars are present (default 90%: a daily bar
    survives one or two missing hours, e.g. the 2023-03-24 exchange outage;
    a 4h bar still needs all four hours) — otherwise NaN."""
    r = {
        "open": panel["open"].resample(rule, closed="right", label="right").first(),
        "high": panel["high"].resample(rule, closed="right", label="right").max(),
        "low": panel["low"].resample(rule, closed="right", label="right").min(),
        "close": panel["close"].resample(rule, closed="right", label="right").last(),
        "qv": panel["qv"].resample(rule, closed="right", label="right").sum(min_count=1),
    }
    idx = panel["close"].index
    fine = idx[1] - idx[0]
    expected = int(pd.Timedelta(rule) / fine)
    need = math.ceil(expected * min_frac - 1e-9)
    n = panel["close"].notna().resample(rule, closed="right", label="right").sum()
    return {k: v.where(n >= need) for k, v in r.items()}


def utc(y: int, m: int, d: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(y, m, d, tzinfo=timezone.utc))
