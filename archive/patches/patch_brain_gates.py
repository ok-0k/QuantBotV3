with open('/home/admin/trading_bot/brain.py', 'r') as f:
    content = f.read()

# 1. Unchain the Long Gate (Gen-0 can buy without ML permission)
old_long = 'if long_allowed and buy_weight > ENSEMBLE_THRESHOLD and ml_prob >= ML_SIGNAL_THRESHOLD:'
new_long = 'if long_allowed and buy_weight > ENSEMBLE_THRESHOLD:'
content = content.replace(old_long, new_long)

# 2. Crush the Short Gate Backdoor (Stops XGBoost from opening ghost shorts)
old_short = 'and ml_down_prob > 0.95'
new_short = 'and ml_down_prob > SHORT_ML_THRESHOLD'
content = content.replace(old_short, new_short)

with open('/home/admin/trading_bot/brain.py', 'w') as f:
    f.write(content)

print("✅ Gates Unlocked & Backdoors Destroyed! Gen-0 has full control.")
