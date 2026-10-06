#!/bin/bash
# Nightly DB maintenance — run by cron at 04:00 while quant-bot is stopped.
#
# Trims only high-volume, re-derivable tables (candles are re-fetched from the
# exchange; equity_curve is a dense 30 s snapshot series).
#
# NEVER deletes trade data: `trades` (incl. the trade journal and skipped
# signals) and `rl_experience` are the bot's learning record and the lifetime
# trade history, and are kept forever (~150 MB/yr at current volume).
#
# Usage: prune_db.sh [db_path]   (defaults to the live DB)
DB="${1:-/home/admin/trading_data/trading.db}"
echo "[$(date)] Starting DB prune on $DB..."
sqlite3 "$DB" << 'SQL'
DELETE FROM candles WHERE (symbol, ts) NOT IN (SELECT symbol, ts FROM candles c2 WHERE c2.symbol = candles.symbol ORDER BY ts DESC LIMIT 500);
DELETE FROM equity_curve WHERE rowid NOT IN (SELECT rowid FROM equity_curve ORDER BY rowid DESC LIMIT 5000);
VACUUM;
SQL
echo "[$(date)] Prune complete. DB size: $(du -sh "$DB" | cut -f1)"
