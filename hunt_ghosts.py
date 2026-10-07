import sqlite3
import sys
import os

# Point script to your bot folder so it can read config.py
sys.path.append('/home/admin/trading_bot')
from config import DB_PATH

target = sys.argv[1].upper() if len(sys.argv) > 1 else ""

print(f"🔍 Cracking the real vault at: {DB_PATH}")

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find the exact table name for trades
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
table_name = 'trades' if 'trades' in tables else tables[0]

if target:
    cursor.execute(f"SELECT * FROM {table_name} WHERE symbol LIKE ? ORDER BY rowid DESC LIMIT 15", (f'%{target}%',))
    print(f"\n👻 HUNTING GHOSTS FOR: {target} 👻")
else:
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 15")
    print(f"\n📂 RAW DATABASE DUMP (Last 15 rows) 📂")

rows = cursor.fetchall()
if not rows:
    print("No records found! The database never saw it.")
else:
    keys = rows[0].keys()
    print(" | ".join([f"{k:<12}" for k in keys]))
    print("-" * 120)
    for row in rows:
        # Truncate strings slightly so they fit cleanly on the terminal screen
        print(" | ".join([f"{str(row[k])[:12]:<12}" for k in keys]))
print("\n")
