"""
wipe_genes.py — DESTRUCTIVE: empties the evolved strategy pool + graveyard.

Refuses to run without --confirm, and always writes a timestamped JSON
backup of the current brain_state to DATA_DIR/backups/ before mutating,
so an accidental run is recoverable.

Usage:
    python wipe_genes.py            # dry-run: shows what would be wiped
    python wipe_genes.py --confirm  # backup, then wipe
"""

import argparse
import json
from datetime import datetime, timezone

from config import DATA_DIR
from db import load_brain_key, save_brain_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="actually wipe (default is a dry-run)")
    args = parser.parse_args()

    state = load_brain_key("brain_state")
    if not state:
        print("⚠️ No brain state found — nothing to wipe.")
        return

    n_strats = len(state.get("strategies") or [])
    n_grave = len(state.get("graveyard") or [])
    print(f"brain_state currently holds {n_strats} strategies, {n_grave} graveyard entries.")

    if not args.confirm:
        print("DRY-RUN: nothing changed. Re-run with --confirm to wipe "
              "(a backup will be written first).")
        return

    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"brain_state-pre-wipe-{stamp}.json"
    backup_path.write_text(json.dumps(state, indent=2))
    print(f"Backup written: {backup_path}")

    state["strategies"] = []
    state["graveyard"] = []
    save_brain_key("brain_state", state)
    print("✅ Gene pool wiped (strategies + graveyard emptied).")


if __name__ == "__main__":
    main()
