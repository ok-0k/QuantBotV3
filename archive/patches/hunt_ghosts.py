import sqlite3
import sys

target = sys.argv[1].upper() if len(sys.argv) > 1 else ""

# Connect directly to the raw SQLite file
conn = sqlite3.connect('/home/admin/trading_bot/trading_bot.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find the exact table name (usually 'trades', 'history', etc.)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
table_name = 'trades' if 'trades' in tables else tables[0]

if target:
    cursor.execute(f"SELECT * FROM {table_name} WHERE symbol LIKE ? ORDER BY rowid DESC LIMIT 15", (f'%{target}%',))
    print(f"\n👻 HUNTING GHOSTS FOR: {target} (Table: {table_name}) 👻")
else:
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 15")
    print(f"\n📂 RAW DATABASE DUMP (Last 15 rows from {table_name}) 📂")

rows = cursor.fetchall()
if not rows:
    print("No records found! The database never saw it.")
else:
    keys = rows[0].keys()
    # Print Headers
    print(" | ".join([f"{k:<12}" for k in keys]))
    print("-" * 100)
    # Print Rows
    for row in rows:
        print(" | ".join([f"{str(row[k]):<12}" for k in keys]))
print("\n")
