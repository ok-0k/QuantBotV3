#!/usr/bin/env python3
"""
set_cash_balance.py — Top up (or set) portfolio.cash to a specific value.

Use case: paper-trading capital has dwindled to the point where position
sizing (a % of equity) is no longer meaningfully testable, and you want to
reload the account with fresh capital without disturbing trade history.

Deliberately does NOT touch:
  - trades table / lifetime trade count  — full trading history stays intact.
  - realised_pnl / realised_pnl_net      — historical record of actual
                                            trading performance.
  - equity_rebase_baseline / ts          — run rebase_equity_baseline.py
                                            --confirm separately afterward if
                                            you also want return%/peak-equity
                                            to read a clean 0% from the new
                                            balance forward.
  - adaptive_edge_profiles_v1, Brain._peak_equity (risk circuit breaker).

Refuses to run (even in dry-run) while open positions exist, since "cash"
alone would not represent total balance in that case.

Usage:
    python set_cash_balance.py                  # dry-run, defaults to 10000
    python set_cash_balance.py --amount 5000     # dry-run a different target
    python set_cash_balance.py --confirm         # backup, then write
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from config import DATA_DIR, DB_PATH
from db import get_all_positions, get_cash, record_equity, set_cash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set portfolio.cash to a specific value (default 10000).",
    )
    parser.add_argument("--amount", type=float, default=10_000.0,
                        help="new cash balance (default: 10000.0)")
    parser.add_argument("--confirm", action="store_true",
                        help="actually write (default is a dry-run report)")
    args = parser.parse_args()

    positions = get_all_positions()
    if positions:
        print(f"REFUSING TO RUN: {len(positions)} open position(s) exist.")
        print("Close all positions first, or this would silently misrepresent "
              "total balance (cash excludes deployed capital).")
        for p in positions:
            print(f"  - {p.get('symbol')} {p.get('side')} qty={p.get('shares')}")
        raise SystemExit(1)

    old_cash = get_cash()
    new_cash = round(args.amount, 2)

    print(f"Current portfolio.cash: {old_cash:.2f}")
    print(f"New portfolio.cash:     {new_cash:.2f}")
    print()
    print("NOT touched: trades table / lifetime trade count, realised_pnl, "
          "realised_pnl_net, equity_rebase_baseline/ts, adaptive_edge_profiles_v1, "
          "Brain._peak_equity (risk circuit breaker).")
    print("Run rebase_equity_baseline.py --confirm afterward if you also want "
          "return%/peak-equity to read a clean 0% from this new balance forward.")

    if not args.confirm:
        print("\nDRY-RUN: nothing changed. Re-run with --confirm to write "
              "(a backup of the prior cash value will be written first).")
        return

    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"cash-balance-pre-{stamp}.json"
    backup_path.write_text(json.dumps({
        "old_cash": old_cash,
        "new_cash": new_cash,
        "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"Backup written: {backup_path}")

    set_cash(new_cash)
    record_equity(new_cash)

    print("\nCash balance set.")
    print(f"DB: {DB_PATH}")
    print(f"New cash: {new_cash:.2f}")


if __name__ == "__main__":
    main()
