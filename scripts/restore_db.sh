#!/bin/bash
# Restore the live SQLite DB from a nightly backup made by prune_db.sh.
#
#   scripts/restore_db.sh <backup.db.gz>         # dry run: verify the backup only
#   scripts/restore_db.sh <backup.db.gz> --yes   # replace the live DB
#
# The DB runs in WAL mode and BOTH services hold it open (the dashboard keeps
# trading.db-wal / -shm mapped). Overwriting trading.db underneath them mixes
# stale WAL pages into the restored file, so this refuses unless quant-bot and
# quant-dashboard are both stopped:
#   sudo systemctl stop quant-bot quant-dashboard
# The current trading.db, -wal and -shm are moved aside (never deleted), the
# backup must pass PRAGMA integrity_check, then it is moved into place.
#
# env: DB (default /home/admin/trading_data/trading.db), SYSTEMCTL (default systemctl)
set -euo pipefail
DB="${DB:-/home/admin/trading_data/trading.db}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
BACKUP="${1:-}"
if [[ -z "$BACKUP" || ! -f "$BACKUP" ]]; then
    echo "usage: $0 <backup.db.gz> [--yes]"
    exit 2
fi

for svc in quant-bot quant-dashboard; do
    if $SYSTEMCTL is-active --quiet "$svc"; then
        echo "Refusing: $svc is running. Stop both first: sudo systemctl stop quant-bot quant-dashboard"
        exit 1
    fi
done

dir="$(dirname "$DB")"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
tmp="$dir/.restore-$stamp.db"
gunzip -c "$BACKUP" > "$tmp"
check="$(sqlite3 "$tmp" "PRAGMA integrity_check;" 2>&1 | head -1 || true)"
if [[ "$check" != "ok" ]]; then
    rm -f "$tmp"
    echo "Backup failed integrity_check: $check — nothing changed."
    exit 1
fi
echo "Backup OK: integrity_check=ok, $(sqlite3 "$tmp" "SELECT COUNT(*) FROM trades;") trade rows."

if [[ "${2:-}" != "--yes" ]]; then
    rm -f "$tmp"
    echo "Dry run: nothing changed. Re-run with --yes to replace $DB (current files are kept aside)."
    exit 0
fi

aside="$dir/pre-restore-$stamp"
mkdir -p "$aside"
for f in "$DB" "$DB-wal" "$DB-shm"; do
    if [[ -e "$f" ]]; then mv "$f" "$aside/"; fi
done
mv "$tmp" "$DB"
echo "Restored $DB from $BACKUP. Previous files kept in $aside"
echo "Now start both: sudo systemctl start quant-bot quant-dashboard"
