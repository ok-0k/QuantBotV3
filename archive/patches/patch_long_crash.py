with open('/home/admin/trading_bot/bot.py', 'r') as f:
    content = f.read()

old_line = '"proceeds": round(cover_cost, 2),'
new_line = '"proceeds": round(locals().get("cover_cost", locals().get("sell_value", 0.0)), 2),'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('/home/admin/trading_bot/bot.py', 'w') as f:
        f.write(content)
    print("✅ Engine patched! Long trades will now survive and save to the database.")
else:
    print("⚠️ Could not find the exact line. It might be formatted differently.")
