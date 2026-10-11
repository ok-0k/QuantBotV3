#!/usr/bin/env python3
"""
set_ledger_anchor.py — Record a reviewed ledger drift in capital_adjustments.

The bot checks (at boot and every minute) that

    cash + Σ short margin_reserved + Σ long cost basis
        == STARTING_CASH + capital_adjustments + realised_pnl_net

and records the difference as portfolio.ledger_drift. Every booking moves
both sides by the same amount, so the drift is constant unless cash moved
without a matching booking. A constant non-zero drift is an out-of-band cash
change that was never recorded — today the 2026-09-30 top-up (43.75 ->
10,000, made before capital_adjustments existed). This script folds the
current drift into capital_adjustments so the drift reads ~0 and any later
move stands out.

Only run --confirm after checking the drift is the known offset (constant
across checks: compare with the bot's last recorded ledger_drift below),
not a fresh leak. --confirm must be given the reviewed drift as --amount and
refuses unless it matches the current drift to within $0.01, so it can never
plug an unreviewed number. Safe with positions open (deployed capital is part
of the formula) and does not move cash, positions or realised PnL.

Deliberately does NOT touch:
  - portfolio.cash, positions, trades  — the books themselves.
  - realised_pnl / realised_pnl_net    — the trading-performance record.
  - STARTING_CASH, equity_rebase_baseline/ts, Brain._peak_equity.

Usage:
    python scripts/set_ledger_anchor.py                          # dry-run: show components
    python scripts/set_ledger_anchor.py --confirm --amount 396.41  # backup, then write
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, DB_PATH  # noqa: E402
from db import (  # noqa: E402
    get_portfolio_stat, ledger_drift, set_portfolio_stat, transaction,
)

AMOUNT_TOLERANCE = 0.01   # --amount must match the live drift to the cent


def _print_components(led: dict) -> None:
    print(f"cash:                      {led['cash']:12,.2f}")
    print(f"+ short margin reserved:   {led['short_margin']:12,.2f}")
    print(f"+ long cost basis:         {led['long_cost']:12,.2f}")
    print(f"= books:                   {led['books']:12,.2f}")
    print(f"STARTING_CASH:             {led['starting_cash']:12,.2f}")
    print(f"+ capital_adjustments:     {led['capital_adjustments']:12,.2f}")
    print(f"+ realised_pnl_net:        {led['realised_net']:12,.2f}")
    print(f"= expected:                {led['expected']:12,.2f}")
    print(f"drift (books - expected):  {led['drift']:+12,.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold the current ledger drift into capital_adjustments "
                    "so the bot's cash invariant reads ~0.",
    )
    parser.add_argument("--confirm", action="store_true",
                        help="actually write (default is a dry-run report)")
    parser.add_argument("--amount", type=float, default=None,
                        help="the reviewed drift to record; required with --confirm "
                             "and must match the current drift within $0.01")
    args = parser.parse_args()

    led = ledger_drift()
    _print_components(led)
    print(f"\nBot's last recorded ledger_drift: {get_portfolio_stat('ledger_drift')} "
          f"at {get_portfolio_stat('ledger_checked_at')}")
    print(f"New capital_adjustments: {led['capital_adjustments']:.6f} + "
          f"{led['drift']:.6f} = {led['capital_adjustments'] + led['drift']:.6f}")
    print("\nNOT touched: cash, positions, trades, realised_pnl, realised_pnl_net, "
          "STARTING_CASH, equity rebase, Brain._peak_equity.")

    if not args.confirm:
        print(f"\nDRY-RUN: nothing changed. Re-run with --confirm --amount "
              f"{led['drift']:.2f} to write (a backup of the prior "
              "capital_adjustments will be written first).")
        return
    if args.amount is None or abs(args.amount - led["drift"]) > AMOUNT_TOLERANCE:
        print(f"\nREFUSING TO WRITE: --confirm needs --amount equal to the current "
              f"drift ({led['drift']:.2f}) within ${AMOUNT_TOLERANCE:.2f}; got {args.amount}.")
        raise SystemExit(1)

    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"ledger-anchor-pre-{stamp}.json"
    backup_path.write_text(json.dumps({
        "old_capital_adjustments": led["capital_adjustments"],
        "components": led,
        "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"Backup written: {backup_path}")

    # Re-check and write in one transaction so nothing can move the books
    # between the confirmation and the write.
    with transaction():
        cur = ledger_drift()
        if abs(args.amount - cur["drift"]) > AMOUNT_TOLERANCE:
            print(f"REFUSING TO WRITE: drift moved to {cur['drift']:.2f} since the "
                  "report above; review again.")
            raise SystemExit(1)
        new_adjustments = cur["capital_adjustments"] + cur["drift"]
        set_portfolio_stat("capital_adjustments", new_adjustments)

    after = ledger_drift()
    print("\nLedger anchor set.")
    print(f"DB: {DB_PATH}")
    print(f"capital_adjustments: {cur['capital_adjustments']:.6f} -> {new_adjustments:.6f}")
    print(f"drift now: {after['drift']:+.6f} (the bot records it within a minute)")


if __name__ == "__main__":
    main()
