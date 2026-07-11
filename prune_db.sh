#!/bin/bash
DB="/home/admin/trading_data/trading.db"
echo "[$(date)] Starting DB prune..."
sqlite3 "$DB" << 'SQL'
DELETE FROM candles WHERE (symbol, ts) NOT IN (SELECT symbol, ts FROM candles c2 WHERE c2.symbol = candles.symbol ORDER BY ts DESC LIMIT 500);
DELETE FROM trades WHERE rowid NOT IN (SELECT rowid FROM trades ORDER BY rowid DESC LIMIT 2000);
DELETE FROM equity_curve WHERE rowid NOT IN (SELECT rowid FROM equity_curve ORDER BY rowid DESC LIMIT 5000);
DELETE FROM rl_experience WHERE rowid NOT IN (SELECT rowid FROM rl_experience ORDER BY rowid DESC LIMIT 2000);
VACUUM;
SQL
echo "[$(date)] Prune complete. DB size: $(du -sh $DB | cut -f1)"
