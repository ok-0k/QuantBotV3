#!/usr/bin/env python3
"""
reset_circuit_breaker.py — Re-arm the drawdown circuit breaker from today's equity.

The breaker (brain.check_circuit_breaker) halts new entries once equity is
MAX_DRAWDOWN_PCT below its persisted high-water mark, and only re-arms itself
when drawdown recovers to MAX_DRAWDOWN_PCT - CB_RECOVERY_PCT. With entries
halted, only positions still open can produce that recovery, so in practice a
trip is a manual halt. This script lifts it deliberately: it sets the persisted
peak (brain_state key circuit_breaker_peak_equity_v1) to the latest equity
snapshot, so drawdown is measured from today.

Deliberately does NOT touch cash, realised PnL, the dashboard rebase point or
the adaptive edge profiles. The running bot keeps its in-memory peak: restart
it afterwards (Brain.restore_peak_equity loads the new value at startup).

Usage:
    python scripts/reset_circuit_breaker.py            # dry-run: shows what would change
    python scripts/reset_circuit_breaker.py --confirm  # backup, then reset the peak
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CB_RECOVERY_PCT, DATA_DIR, DB_PATH, MAX_DRAWDOWN_PCT  # noqa: E402
from db import get_db, load_brain_key, save_brain_key  # noqa: E402

PEAK_KEY = "circuit_breaker_peak_equity_v1"


def _latest_equity() -> tuple[float, str] | tuple[None, None]:
    with get_db() as conn:
        row = conn.execute("SELECT equity, ts FROM equity_curve ORDER BY ts DESC LIMIT 1").fetchone()
    return (float(row[0]), row[1]) if row else (None, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-arm the drawdown circuit breaker from current equity.")
    parser.add_argument("--confirm", action="store_true", help="actually write (default is a dry-run report)")
    args = parser.parse_args(argv)

    old_peak = load_brain_key(PEAK_KEY, None)
    equity, eq_ts = _latest_equity()
    if equity is None:
        print("No equity_curve rows yet — nothing to reset from.")
        return 1
    dd = (float(old_peak) - equity) / float(old_peak) * 100 if old_peak else 0.0

    print(f"Persisted peak (current):   {old_peak}")
    print(f"Latest equity snapshot:     {equity:.2f} @ {eq_ts}")
    print(f"Drawdown from that peak:    {dd:.1f}%  (trips at {MAX_DRAWDOWN_PCT * 100:.0f}%, "
          f"re-arms at {max(0.0, MAX_DRAWDOWN_PCT - CB_RECOVERY_PCT) * 100:.0f}%)")
    print(f"New peak:                   {equity:.2f}")

    if not args.confirm:
        print("\nDRY-RUN: nothing changed. Re-run with --confirm to write "
              "(a backup of the current peak is written first).")
        return 0

    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"circuit-breaker-pre-{stamp}.json"
    backup_path.write_text(json.dumps({
        "key": PEAK_KEY, "old_peak": old_peak, "new_peak": equity,
        "equity_ts": eq_ts, "drawdown_pct_at_reset": round(dd, 3),
    }, indent=2))
    print(f"Backup written: {backup_path}")

    save_brain_key(PEAK_KEY, equity)
    print(f"\nPeak reset to {equity:.2f} in {DB_PATH}.")
    print("Restart the bot for it to take effect:  sudo systemctl restart quant-bot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
