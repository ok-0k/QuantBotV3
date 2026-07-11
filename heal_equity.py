#!/usr/bin/env python3
"""
One-off equity repair utility.

Recomputes true equity as:
    STARTING_CASH + SUM(net_pnl of filled trades)

Then overwrites:
  - portfolio.cash
  - latest equity_curve row (or inserts one if missing)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone

from config import DATA_DIR, DB_PATH, STARTING_CASH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute cash/equity from trade history and OVERWRITE "
                    "portfolio.cash + latest equity_curve row.",
    )
    parser.add_argument("--confirm", action="store_true",
                        help="actually overwrite (default is a dry-run report)")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n_filled,
                COALESCE(SUM(COALESCE(net_pnl, pnl, 0.0)), 0.0) AS sum_net_pnl
            FROM trades
            WHERE status='filled'
            """
        ).fetchone()

        n_filled = int(row["n_filled"] or 0)
        sum_net_pnl = float(row["sum_net_pnl"] or 0.0)
        healed_equity = float(STARTING_CASH) + sum_net_pnl

        cur_cash_row = conn.execute(
            "SELECT value FROM portfolio WHERE key='cash'").fetchone()
        current_cash = float(cur_cash_row[0]) if cur_cash_row else None

        print(f"Current portfolio.cash: {current_cash}")
        print(f"Recomputed (healed) value: {healed_equity:.2f}")
        print("NOTE: the trades table is pruned nightly; if history was pruned "
              "this recomputation may be WRONG. Also unsafe while the bot has "
              "open positions (cash excludes deployed capital).")

        if not args.confirm:
            print("DRY-RUN: nothing changed. Re-run with --confirm to overwrite "
                  "(a backup of the current values will be written first).")
            return

        backup_dir = DATA_DIR / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"portfolio-pre-heal-{stamp}.json"
        backup_path.write_text(json.dumps({
            "cash_before": current_cash,
            "healed_equity_written": healed_equity,
            "n_filled": n_filled,
            "sum_net_pnl": sum_net_pnl,
            "ts": _utc_now_iso(),
        }, indent=2))
        print(f"Backup written: {backup_path}")

        with conn:
            # Overwrite cash to healed equity.
            conn.execute(
                "INSERT OR REPLACE INTO portfolio(key, value) VALUES('cash', ?)",
                (str(healed_equity),),
            )

            # Overwrite latest equity row (authoritative dashboard source), or insert new.
            latest = conn.execute(
                "SELECT rowid FROM equity_curve ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if latest:
                conn.execute(
                    "UPDATE equity_curve SET ts=?, equity=? WHERE rowid=?",
                    (_utc_now_iso(), healed_equity, int(latest["rowid"])),
                )
            else:
                conn.execute(
                    "INSERT INTO equity_curve(ts, equity) VALUES(?, ?)",
                    (_utc_now_iso(), healed_equity),
                )

        print("Equity heal complete.")
        print(f"DB: {DB_PATH}")
        print(f"Filled trades: {n_filled}")
        print(f"SUM(net_pnl): {sum_net_pnl:.2f}")
        print(f"STARTING_CASH: {STARTING_CASH:.2f}")
        print(f"HEALED_EQUITY/CASH: {healed_equity:.2f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
