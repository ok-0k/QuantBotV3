#!/usr/bin/env python3
"""
Read-only report for the active symbol-split experiment (config.EXPERIMENT_NAME).

    .venv/bin/python scripts/experiment_report.py [experiment_name]

Compares arms on closed trades from the trade journal. Headline: net P&L per
symbol-day (the assignment unit is the symbol; zero-trade symbols count,
delisted ones are excluded), the B - A difference with a 95% bootstrap CI
resampled by calendar day, and the pre-registered decision rule's verdict.
Secondary: per-trade economics, exit mix, hold time and pace, and a 95%
interval on the difference in mean net return per trade. The numbers come from
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
    psd = exp.get("per_symbol_day")
    if psd:
        print(f"\n══ Net P&L per symbol-day ══  {psd['days']} UTC days "
              f"({psd['first_day']} .. {psd['last_day']}), delisted symbols excluded")
        for arm in ("A", "B"):
            print(f"  Arm {arm}: {psd['symbols'][arm]} symbols   net ${psd['net_per_symbol'][arm]:+.2f}/symbol   "
                  f"mean ${psd['mean'][arm]:+.3f}/symbol/day")
        ci = (f"95% CI ${psd['ci_lo']:+.3f} .. ${psd['ci_hi']:+.3f}, "
              f"day bootstrap, {psd['resamples']} resamples" if psd["ci_lo"] is not None else "no CI yet")
        print(f"  B − A mean daily net P&L per symbol: ${psd['diff']:+.3f}  ({ci})")
        print(f"  Decision rule (pre-registered): {psd['rule']}")
        print(f"  Verdict: {psd['verdict']}")
    if exp.get("note"):
        print(f"  Note: {exp['note']}")


if __name__ == "__main__":
    main()
