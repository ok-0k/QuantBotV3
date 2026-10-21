#!/usr/bin/env python3
"""
Read-only shadow learner report (research/PREREG_SHADOW.md).

    TRADING_ENV_FILE=/nonexistent .venv/bin/python scripts/shadow_report.py

Running numbers are NOT evidence before the next evaluation date; candidates
are decided only at T0 + 8 weeks and T0 + 16 weeks. Opens the DB read-only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import insights  # noqa: E402
import shadow  # noqa: E402


def main() -> None:
    conn = insights.connect()
    try:
        s = shadow.summary(conn, top=25)
    finally:
        conn.close()
    print(f"Shadow learner: {'ON' if s['enabled'] else 'OFF'}   T0 {s['t0'] or '— (no shadow trade yet)'}")
    print(f"  shadow trades: {s['closed']} closed, {s['open']} open; mean net R (all) "
          f"{'—' if s.get('all_mean_r') is None else format(s['all_mean_r'], '+.3f')}")
    print(f"  next evaluation: {s['next_eval'] or '—'}   (running numbers are NOT evidence before it)")
    for e in s["evaluations"]:
        print(f"\n== Evaluation at {e['at']}: candidates {e['candidates'] or 'none'}")
        for name in e["eligible"]:
            st = e["stats"].get(name)
            if st:
                print(f"  {name:22} n {st['n']:>5}  days {st['days']:>3}  mean {st['mean_r']:+.3f}R  "
                      f"t {st['t'] if st['t'] is None else format(st['t'], '.2f')}  "
                      f"pass {'yes' if shadow.passes(st) else 'no'}")
    if s["running"]:
        print("\nRunning (≥ 30 closed, by t):")
        for r in s["running"]:
            t = "—" if r["t"] is None else f"{r['t']:.2f}"
            print(f"  {r['name']:22} n {r['n']:>5}  days {r['days']:>3}  mean {r['mean_r']:+.3f}R  t {t}")


if __name__ == "__main__":
    main()
