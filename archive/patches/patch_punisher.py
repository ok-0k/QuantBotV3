with open('/home/admin/trading_bot/punisher.py', 'r') as f:
    content = f.read()

old_path = "model_path = '/home/admin/trading_bot/models/xgboost_engine.json'"
new_path = "model_path = '/home/admin/trading_bot/offline_output/xgb_model.json'"

if old_path in content:
    content = content.replace(old_path, new_path)
    with open('/home/admin/trading_bot/punisher.py', 'w') as f:
        f.write(content)
    print("✅ Brain located. Coordinates locked in!")
else:
    print("⚠️ Could not find the exact line. Check the punisher.py file manually.")
