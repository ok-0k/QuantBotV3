#!/bin/bash
# Nightly DB maintenance — run by cron at 04:00 while quant-bot is stopped.
#
# 1. BACKUP first: a consistent SQLite snapshot (gzip) into $BACKUP_DIR,
#    keeping the newest $BACKUP_KEEP (default 14). If the backup fails,
#    nothing is pruned. Rotation only drops old backup COPIES — the live DB
#    keeps all trade data. Restore: stop the bot, then
#    gunzip -c <backup>.db.gz > /home/admin/trading_data/trading.db
#
# 2. PRUNE only high-volume, re-derivable data: candles (re-fetched from the
#    exchange) and equity_curve DETAIL — the 30 s snapshot series is kept in full
#    for 2 days, then thinned to one point per 5 minutes and kept forever
#    (~6 MB/yr), so long-range equity history survives for the dashboard.
#
# NEVER deletes trade data: `trades` (incl. the trade journal and skipped
# signals) and `rl_experience` are the bot's learning record and the lifetime
# trade history, and are kept forever (~150 MB/yr at current volume).
#
# Usage: prune_db.sh [db_path]   (defaults to the live DB)
#        env BACKUP_DIR (default <db dir>/backups/db), BACKUP_KEEP (default 14)
DB="${1:-/home/admin/trading_data/trading.db}"
BACKUP_DIR="${BACKUP_DIR:-$(dirname "$DB")/backups/db}"
BACKUP_KEEP="${BACKUP_KEEP:-14}"

mkdir -p "$BACKUP_DIR"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
tmp="$BACKUP_DIR/.trading-$stamp.db"
out="$BACKUP_DIR/trading-$stamp.db.gz"
if sqlite3 "$DB" ".backup '$tmp'" && gzip -c "$tmp" > "$out"; then
    rm -f "$tmp"
    echo "[$(date)] Backup written: $out ($(du -h "$out" | cut -f1))"
    ls -1t "$BACKUP_DIR"/trading-*.db.gz | tail -n +$((BACKUP_KEEP + 1)) | xargs -r rm -f
else
    rm -f "$tmp" "$out"
    echo "[$(date)] BACKUP FAILED for $DB — prune skipped"
    exit 1
fi

echo "[$(date)] Starting DB prune on $DB..."
sqlite3 "$DB" << 'SQL'
DELETE FROM candles WHERE (symbol, ts) NOT IN (SELECT symbol, ts FROM candles c2 WHERE c2.symbol = candles.symbol ORDER BY ts DESC LIMIT 500);
DELETE FROM equity_curve
WHERE ts < strftime('%Y-%m-%dT%H:%M:%S', 'now', '-2 days')
  AND rowid NOT IN (
    SELECT MIN(rowid) FROM equity_curve
    WHERE ts < strftime('%Y-%m-%dT%H:%M:%S', 'now', '-2 days')
    GROUP BY substr(ts, 1, 14) || (CAST(substr(ts, 15, 2) AS INTEGER) / 5)
  );
VACUUM;
SQL
echo "[$(date)] Prune complete. DB size: $(du -sh "$DB" | cut -f1)"
