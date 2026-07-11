import re

with open('/home/admin/trading_bot/config.py', 'r') as f:
    text = f.read()

# 1. The Genetic Gate (Lowered to 20% so Gen-0 can actually trade)
text = re.sub(r'ENSEMBLE_THRESHOLD\s*=\s*[0-9.]+', 'ENSEMBLE_THRESHOLD = 0.20', text)

# 2. The XGBoost Kill-Switches (Locked at 2.0 so ML cannot solo-trade)
text = re.sub(r'ML_SIGNAL_THRESHOLD\s*=\s*[0-9.]+', 'ML_SIGNAL_THRESHOLD = 2.0', text)
text = re.sub(r'SHORT_ML_THRESHOLD\s*=\s*[0-9.]+', 'SHORT_ML_THRESHOLD = 2.0', text)

# 3. The Executioner Settings (8 trades to prove themselves, 55% win rate required)
text = re.sub(r'TRIAL_TRADES\s*=\s*[0-9]+', 'TRIAL_TRADES = 8', text)
text = re.sub(r'TRIAL_MIN_WIN_RATE\s*=\s*[0-9.]+', 'TRIAL_MIN_WIN_RATE = 0.55', text)
text = re.sub(r'KILL_THRESHOLD\s*=\s*-[0-9.]+', 'KILL_THRESHOLD = -1.5', text)

with open('/home/admin/trading_bot/config.py', 'w') as f:
    f.write(text)

print("✅ Master Config Locked: Gen-0 is driving, XGBoost is benched, Executioner is ready.")
