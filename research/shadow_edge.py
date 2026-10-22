# Copied unchanged (apart from this header) from
# /home/admin/.audit-2026-10/shadow_edge_20261021.py, the read-only evidence-gate
# shadow check the PM ran on 2026-10-21. Results: research/SHADOW_CHECK_2026-10-21.md.
# It opens trading.db with mode=ro, fetches Binance public 1m klines through
# scripts/entry_edge.py (sys.path below points at the worktree it was run from),
# and runs at import time: run it as a script, never import it.
"""Shadow check: do the signals the evidence gate refused since 2026-10-14 08:00Z
predict falling prices? Gross short forward returns vs random (symbol, minute)."""
import sys, sqlite3, time, random, statistics as st, math, json
from collections import defaultdict
sys.path.insert(0, "/home/admin/wt-research-merge/scripts")
import entry_edge as E

c = sqlite3.connect(f"file:{E.DB}?mode=ro", uri=True); c.row_factory = sqlite3.Row
now = int(time.time() * 1000); H = max(E.HORIZONS) * 60_000
rows = [dict(r) for r in c.execute("SELECT ts, symbol, price, entry_features FROM trades WHERE action='short' "
        "AND reason='no_validated_edge' AND price > 0 AND ts >= '2026-10-14T08:00' ORDER BY id")]
rows = [r for r in rows if E.ms(r["ts"]) + H < now - 60_000]
syms = sorted({r["symbol"] for r in rows})
t_lo = min(E.ms(r["ts"]) for r in rows) - 60_000; t_hi = min(max(E.ms(r["ts"]) for r in rows) + H + 120_000, now)
print(f"{len(rows)} refused signals, {len(syms)} symbols; fetching 1m klines...", flush=True)
K = {s: E.fetch(s, t_lo, t_hi) for s in syms}
G = defaultdict(list)
for r in rows:
    f = E.fwd(K[r["symbol"]], E.ms(r["ts"]), float(r["price"]))
    if f:
        f["day"] = r["ts"][:10]; G["refused"].append(f)
        tier = (json.loads(r["entry_features"] or "{}").get("ml_tier"))
        if tier is not None: G[f"ml{tier}"].append(f)
rnd = random.Random(7)
for _ in range(20000):
    s = rnd.choice(syms); t = rnd.randrange(t_lo, t_hi - H) // 60_000 * 60_000
    k = K[s].get(t)
    if k:
        f = E.fwd(K[s], t, k[0])
        if f: f["day"] = time.strftime("%Y-%m-%d", time.gmtime(t / 1000)); G["random"].append(f)
def day_ci(xs, key):
    by = defaultdict(list)
    for x in xs: by[x["day"]].append(x[key])
    d = [st.mean(v) for v in by.values()]
    return st.mean(d), (1.96 * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")), len(d)
print(f"\nGross short return % before ~0.36% round-trip cost; mean ± 95% CI over day means")
print(f"{'group':9}{'n':>7}" + "".join(f"{'+'+str(h)+'m':>17}" for h in E.HORIZONS))
order = ["refused", "random"] + sorted(k for k in G if k.startswith("ml"))
for g in order:
    xs = G[g]
    if len(xs) < 30: continue
    cells = []
    for h in E.HORIZONS:
        m, e, nd = day_ci(xs, f"r{h}")
        cells.append(f"{m:+.3f}±{e:.3f}")
    print(f"{g:9}{len(xs):>7}" + "".join(f"{x:>17}" for x in cells))
for g in ("refused", "random"):
    xs = G[g]
    print(f"{g}: share reaching +0.36% MFE within 30 min: {100*sum(x['mfe30'] >= 0.36 for x in xs)/len(xs):.1f}%")
