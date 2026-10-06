#!/usr/bin/env python3
"""
Read-only report for the active symbol-split experiment (config.EXPERIMENT_NAME).

    .venv/bin/python scripts/experiment_report.py [experiment_name]

Compares arms on closed trades from the trade journal: per-trade economics,
exit mix, hold time and pace, plus a 95% interval on the difference in mean
net return per trade so a winner is not called on noise. Opens the DB
read-only; safe to run while the bot is live.
"""

import json
import math
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, EXPERIMENT_NAME  # noqa: E402


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_NAME
    if not name:
        sys.exit("No experiment active (EXPERIMENT_NAME is empty) and none given.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    closed: dict[str, list[dict]] = defaultdict(list)
    entries: dict[str, list[str]] = defaultdict(list)
    skips: dict[str, Counter] = defaultdict(Counter)
    for r in conn.execute(
        "SELECT ts, action, status, reason, pnl, trade_value, fee_total, exit_reason, "
        "hold_time_seconds, entry_features FROM trades "
        "WHERE entry_features LIKE ? ORDER BY id", (f'%"experiment": "{name}"%',)
    ):
        f = json.loads(r["entry_features"])
        arm = f.get("arm")
        if f.get("experiment") != name or arm is None:
            continue
        if r["action"] in ("short", "buy"):
            if r["status"] == "filled":
                entries[arm].append(r["ts"])
            else:
                skips[arm][r["reason"] or "?"] += 1
        elif r["status"] == "filled" and r["pnl"] is not None:
            closed[arm].append(dict(r))

    open_now: Counter = Counter()
    for (feats,) in conn.execute(
        "SELECT entry_features FROM positions WHERE margin_reserved > 0 OR shares > 0"
    ):
        try:
            f = json.loads(feats or "{}")
        except ValueError:
            continue
        if f.get("experiment") == name:
            open_now[f.get("arm")] += 1

    arms = sorted(set(closed) | set(entries))
    if not arms:
        print(f"No journaled trades for experiment '{name}' yet.")
        return
    print(f"Experiment: {name}")
    rets: dict[str, list[float]] = {}
    for arm in arms:
        rows = closed.get(arm, [])
        first = entries[arm][0] if entries[arm] else None
        days = max((_ts(entries[arm][-1]) - _ts(first)).total_seconds() / 86400, 1e-9) if first else 0
        print(f"\n── Arm {arm} ──  entries={len(entries[arm])}  closed={len(rows)}  open now={open_now[arm]}"
              + (f"  ({len(entries[arm]) / days:.0f} entries/day)" if days > 0.04 else ""))
        if not rows:
            continue
        pnl = [x["pnl"] for x in rows]
        ret = [x["pnl"] / x["trade_value"] * 100 for x in rows if x["trade_value"]]
        rets[arm] = ret
        holds = [x["hold_time_seconds"] / 60 for x in rows if x["hold_time_seconds"] is not None]
        wins = sum(p > 0 for p in pnl)
        print(f"  net ${sum(pnl):+.2f}   win {100 * wins / len(pnl):.0f}%   "
              f"avg ${st.mean(pnl):+.3f}/trade   avg {st.mean(ret):+.3f}% of notional   "
              f"fees ${sum(x['fee_total'] or 0 for x in rows):.2f}")
        if holds:
            print(f"  hold: median {st.median(holds):.0f} min, mean {st.mean(holds):.0f} min")
        mix = Counter(x["exit_reason"] or "?" for x in rows)
        print("  exits: " + ", ".join(f"{k} {v}" for k, v in mix.most_common()))
        if skips[arm]:
            print("  skips: " + ", ".join(f"{k} {v}" for k, v in skips[arm].most_common(5)))

    if len(rets) == 2 and all(len(v) >= 2 for v in rets.values()):
        a, b = (rets[k] for k in sorted(rets))
        diff = st.mean(b) - st.mean(a)
        se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        verdict = ("not yet distinguishable" if lo <= 0 <= hi
                   else "B better" if lo > 0 else "A better")
        print(f"\nB − A mean net return/trade: {diff:+.3f}%  (95% CI {lo:+.3f} .. {hi:+.3f})  → {verdict}")


if __name__ == "__main__":
    main()
