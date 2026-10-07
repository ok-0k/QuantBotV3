"""
Entry-edge research (read-only): does the bot's short signal predict price
falling, independent of any exit logic?

For every short entry (taken, and SAC-vetoed skips) and for random
(symbol, minute) baselines, measure the gross forward return of a short at
5/15/30/60/120/240 min using Binance public 1m klines (cached locally).
"""
import json
import math
import random
import sqlite3
import statistics as st
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path("/home/admin/trading_data/research")
CACHE = HERE / "klines"
CACHE.mkdir(parents=True, exist_ok=True)
DB = "/home/admin/trading_data/trading.db"
REST = "https://api.binance.com/api/v3/klines"
HORIZONS = (5, 15, 30, 60, 120, 240)
COST = 0.36      # % round trip: 0.20 fee + 2 x 0.08 slippage


def ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


def fetch(symbol: str, start: int, end: int) -> dict[int, tuple]:
    """open_time_ms -> (open, high, low, close); cached per symbol."""
    f = CACHE / f"{symbol}.json"
    have = {int(k): tuple(v) for k, v in json.loads(f.read_text()).items()} if f.exists() else {}
    t = start
    if have:
        t = max(start, max(have) + 60_000)
    while t < end:
        url = f"{REST}?symbol={symbol}&interval=1m&startTime={t}&limit=1000"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                rows = json.loads(r.read())
        except Exception as exc:     # delisted / unknown symbol / transient
            print(f"  {symbol}: fetch failed ({exc})", file=sys.stderr)
            break
        if not rows:
            break
        for k in rows:
            have[int(k[0])] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
        t = int(rows[-1][0]) + 60_000
        time.sleep(0.25)             # ~4 req/s, far under Binance limits
    f.write_text(json.dumps({str(k): v for k, v in have.items()}))
    return have


def fwd(kl: dict, t_ms: int, p0: float) -> dict | None:
    """Short forward returns (%) from entry minute t_ms at price p0."""
    m0 = t_ms // 60_000 * 60_000
    out = {}
    lows, highs = [], []
    for i in range(max(HORIZONS)):
        k = kl.get(m0 + i * 60_000)
        if k is None:
            return None
        highs.append(k[1]); lows.append(k[2])
        h = i + 1
        if h in HORIZONS:
            out[f"r{h}"] = (p0 - k[3]) / p0 * 100
            out[f"mfe{h}"] = (p0 - min(lows)) / p0 * 100
            out[f"mae{h}"] = (p0 - max(highs)) / p0 * 100
    return out


def ci(xs):
    if len(xs) < 2:
        return float("nan"), float("nan")
    m = st.mean(xs)
    return m, 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    now_ms = int(time.time() * 1000)
    horizon_ms = max(HORIZONS) * 60_000
    rows = [dict(r) for r in c.execute(
        "SELECT ts, symbol, status, reason, price, entry_features FROM trades "
        "WHERE action='short' AND (status='filled' OR reason='sac_veto') AND price > 0 ORDER BY id")]
    rows = [r for r in rows if ms(r["ts"]) + horizon_ms < now_ms - 60_000]
    syms = sorted({r["symbol"] for r in rows})
    t_lo = min(ms(r["ts"]) for r in rows) - 60_000
    t_hi = min(max(ms(r["ts"]) for r in rows) + horizon_ms + 120_000, now_ms)
    print(f"{len(rows)} entries over {len(syms)} symbols, "
          f"{datetime.fromtimestamp(t_lo/1000, timezone.utc):%m-%d %H:%M} .. "
          f"{datetime.fromtimestamp(t_hi/1000, timezone.utc):%m-%d %H:%M} UTC; fetching klines...")
    K = {s: fetch(s, t_lo, t_hi) for s in syms}

    groups = defaultdict(list)
    feats = []
    for r in rows:
        f = fwd(K[r["symbol"]], ms(r["ts"]), float(r["price"]))
        if f is None:
            continue
        g = "taken" if r["status"] == "filled" else "sac_veto"
        groups[g].append(f)
        if g == "taken" and r["entry_features"]:
            feats.append((json.loads(r["entry_features"]), f))

    # random baseline: same symbols, same period, uniformly random minutes
    rnd = random.Random(7)
    for _ in range(6000):
        s = rnd.choice(syms)
        keys = K[s]
        if not keys:
            continue
        t = rnd.randrange(t_lo, t_hi - horizon_ms) // 60_000 * 60_000
        k = keys.get(t)
        if k is None:
            continue
        f = fwd(keys, t, k[0])
        if f:
            groups["random"].append(f)

    print("\nGross short return % (before ~0.36% costs), mean ± 95% CI")
    print(f"{'group':10}{'n':>6}" + "".join(f"{('+' + str(h) + 'm'):>16}" for h in HORIZONS))
    for g in ("taken", "sac_veto", "random"):
        xs = groups[g]
        cells = []
        for h in HORIZONS:
            m, e = ci([x[f"r{h}"] for x in xs])
            cells.append(f"{m:+.3f}±{e:.3f}")
        print(f"{g:10}{len(xs):>6}" + "".join(f"{c:>16}" for c in cells))

    print("\nShare reaching +0.36% (cost) favourable excursion within 30 / 60 min")
    for g in ("taken", "sac_veto", "random"):
        xs = groups[g]
        if xs:
            print(f"  {g:10} {100*sum(x['mfe30']>=COST for x in xs)/len(xs):4.0f}% / "
                  f"{100*sum(x['mfe60']>=COST for x in xs)/len(xs):4.0f}%")

    if feats:
        def rank(v):
            o = sorted(range(len(v)), key=v.__getitem__)
            r = [0.0] * len(v)
            for i, j in enumerate(o):
                r[j] = float(i)
            return r

        def rho(a, b):
            ra, rb = rank(a), rank(b)
            ma, mb = st.mean(ra), st.mean(rb)
            num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
            den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
            return num / den if den else float("nan")

        print(f"\nFeature vs 60-min forward return, taken entries with features (n={len(feats)}; "
              f"|rho| > {1.96/math.sqrt(len(feats)):.2f} ~ significant)")
        for k in ("ml_down", "short_score", "adx", "pdi", "mdi", "rvol", "ret5", "ret20",
                  "ema_spread", "atr_pct", "sac_fraction", "conviction_mult"):
            pairs = [(f.get(k), x["r60"]) for f, x in feats if isinstance(f.get(k), (int, float))]
            if len(pairs) >= 30:
                a, b = zip(*pairs)
                print(f"  {k:16} rho={rho(list(a), list(b)):+.3f}  (n={len(pairs)})")


if __name__ == "__main__":
    main()
