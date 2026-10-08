#!/usr/bin/env python3
"""
Read-only report for the active symbol-split experiment (config.EXPERIMENT_NAME).

    .venv/bin/python scripts/experiment_report.py [experiment_name]

Compares arms on closed trades from the trade journal: per-trade economics,
exit mix, hold time and pace, plus a 95% interval on the difference in mean
net return per trade so a winner is not called on noise. The numbers come from
insights.experiment(), the same code behind the dashboard's Experiment card.
Opens the DB read-only; safe to run while the bot is live.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import insights  # noqa: E402
from config import EXPERIMENT_NAME  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_NAME
    if not name:
        sys.exit("No experiment active (EXPERIMENT_NAME is empty) and none given.")
    conn = insights.connect()
    try:
        exp = insights.experiment(conn, name)
    finally:
        conn.close()
    if not exp or not exp["arms"]:
        print(f"No journaled trades for experiment '{name}' yet.")
        return
    print(f"Experiment: {name}")
    for arm, a in exp["arms"].items():
        pace = f"  ({a['entries_per_day']:.0f} entries/day)" if a.get("entries_per_day") else ""
        print(f"\n── Arm {arm} ──  entries={a['entries']}  closed={a['closed']}  open now={a['open']}{pace}")
        if not a["closed"]:
            continue
        print(f"  net ${a['net']:+.2f}   win {a['win_rate']:.0f}%   avg ${a['avg_trade']:+.3f}/trade   "
              f"avg {a['avg_pct']:+.3f}% of notional   fees ${a['fees']:.2f}")
        if a.get("median_hold_min") is not None:
            print(f"  hold: median {a['median_hold_min']:.0f} min, mean {a['mean_hold_min']:.0f} min")
        print("  exits: " + ", ".join(f"{k} {v}" for k, v in a["exits"].items()))
        if a["skips"]:
            print("  skips: " + ", ".join(f"{k} {v}" for k, v in a["skips"].items()))
    if exp["diff"] is not None:
        print(f"\nB − A mean net return/trade: {exp['diff']:+.3f}%  "
              f"(95% CI {exp['ci_lo']:+.3f} .. {exp['ci_hi']:+.3f})  → {exp['verdict']}")


if __name__ == "__main__":
    main()
