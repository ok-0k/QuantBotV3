#!/usr/bin/env python3
"""
rebase_equity_baseline.py — Declare a new dashboard performance epoch.

Sets equity_rebase_baseline / equity_rebase_ts in the portfolio table.
dashboard.py's return_pct and "peak equity" then measure performance
relative to this baseline/timestamp instead of STARTING_CASH and full
history.

Deliberately does NOT touch:
  - STARTING_CASH (config.py)   — still the fresh-install seed amount and
                                  bot.py's cash-invariant reference point.
  - realised_pnl / realised_pnl_net — the historical record of actual
                                  trading performance (gross vs net of fees).
  - adaptive_edge_profiles_v1   — the adaptive sizing learner's trained state.
  - Brain._peak_equity          — the risk circuit breaker's own drawdown
                                  high-water mark (brain_state table); this
                                  is a display-only rebase, not a risk-control
                                  change.

Usage:
    python rebase_equity_baseline.py            # dry-run: shows what would be set
    python rebase_equity_baseline.py --confirm  # backup, then set the rebase point
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from accounting_v2 import position_equity_components
from config import DATA_DIR, DB_PATH, STARTING_CASH
from db import get_all_positions, get_cash, get_equity_rebase, set_equity_rebase, set_portfolio_stat


def _current_equity() -> float:
    total = get_cash()
    for p in get_all_positions():
        # No live mark price available outside the bot's candle cache; use
        # avg_cost as the mark (zero assumed unrealised PnL). This script is
        # meant for a clean rebase point with no open positions — the normal
        # case here — so this fallback rarely matters in practice.
        _, _, contrib = position_equity_components(
            side=p.get("side", "long"), avg_cost=p.get("avg_cost", 0.0),
            quantity=p.get("shares", 0.0), mark_price=p.get("avg_cost", 0.0),
            margin_reserved=p.get("margin_reserved", 0.0),
        )
        total += contrib
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set equity_rebase_baseline/ts so the dashboard's return%% "
                    "and peak-equity are measured from current equity forward.",
    )
    parser.add_argument("--confirm", action="store_true",
                        help="actually write (default is a dry-run report)")
    args = parser.parse_args()

    equity = _current_equity()
    now = datetime.now(timezone.utc).isoformat()
    old_baseline, old_ts = get_equity_rebase()

    print(f"STARTING_CASH (unaffected):           {STARTING_CASH:.2f}")
    print(f"Existing rebase baseline (if any):    {old_baseline}")
    print(f"Existing rebase timestamp (if any):   {old_ts}")
    print(f"New rebase baseline (current equity): {equity:.6f}")
    print(f"New rebase timestamp:                 {now}")
    print()
    print("NOT touched: STARTING_CASH, realised_pnl, realised_pnl_net,")
    print("adaptive_edge_profiles_v1, Brain._peak_equity (risk circuit breaker).")

    if not args.confirm:
        print("\nDRY-RUN: nothing changed. Re-run with --confirm to write "
              "(a backup of the prior rebase state will be written first).")
        return

    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"equity-rebase-pre-{stamp}.json"
    backup_path.write_text(json.dumps({
        "old_rebase_baseline": old_baseline,
        "old_rebase_ts": old_ts,
        "new_rebase_baseline": equity,
        "new_rebase_ts": now,
        "starting_cash": STARTING_CASH,
    }, indent=2))
    print(f"Backup written: {backup_path}")

    set_equity_rebase(equity, now)
    # Also reset the dashboard's own displayed peak so it reads correctly the
    # instant the dashboard restarts, without waiting on its ratchet logic to
    # naturally decay the old value (it never would — peak-equity only ratchets
    # up).
    set_portfolio_stat("peak_equity_all_time", round(equity, 6))

    print("\nRebase complete.")
    print(f"DB: {DB_PATH}")
    print(f"New baseline: {equity:.2f} @ {now}")


if __name__ == "__main__":
    main()
