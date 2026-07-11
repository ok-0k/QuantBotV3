with open('/home/admin/trading_bot/dashboard.py', 'r') as f:
    content = f.read()

# Make the dashboard actually read the 'side' column instead of guessing
old_line = "const isShort=action.startsWith('short')||action==='short';"
new_line = "const isShort=action.startsWith('short')||action==='short'||action==='cover'||t.side==='short';"

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('/home/admin/trading_bot/dashboard.py', 'w') as f:
        f.write(content)
    print("✅ Dashboard UI fixed! Covers and Shorts will now be properly identified.")
else:
    print("⚠️ Could not find the exact JS line, it might be formatted differently.")
