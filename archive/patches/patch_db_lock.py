with open('/home/admin/trading_bot/db.py', 'r') as f:
    content = f.read()

# Add a 20-second timeout to the database connection
old_line = 'sqlite3.connect(str(DB_PATH), check_same_thread=False)'
new_line = 'sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=20.0)'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('/home/admin/trading_bot/db.py', 'w') as f:
        f.write(content)
    print("✅ Database Lock fixed! Engine will now wait in line to save trades.")
else:
    print("⚠️ Could not find the exact connection line, it might already be patched.")
