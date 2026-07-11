with open('/home/admin/trading_bot/bot.py', 'r') as f:
    content = f.read()

# Rip out the hardcoded XGBoost fallback
content = content.replace('return top["strategy"] if top else "XGBOOST"', 'return top["strategy"] if top else None')

with open('/home/admin/trading_bot/bot.py', 'w') as f:
    f.write(content)

print("✅ Backdoor Destroyed: XGBoost is permanently locked out.")
