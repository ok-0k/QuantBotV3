import re

with open('/home/admin/trading_bot/config.py', 'r') as f:
    text = f.read()

text = re.sub(r'ENSEMBLE_THRESHOLD\s*=\s*0\.10', 'ENSEMBLE_THRESHOLD = 0.45', text)
text = re.sub(r'TRIAL_TRADES\s*=\s*30', 'TRIAL_TRADES = 8', text)
text = re.sub(r'TRIAL_MIN_WIN_RATE\s*=\s*0\.40', 'TRIAL_MIN_WIN_RATE = 0.55', text)
text = re.sub(r'KILL_THRESHOLD\s*=\s*-3\.0', 'KILL_THRESHOLD = -1.5', text)
text = re.sub(r'SHORT_ML_THRESHOLD\s*=\s*0\.55', 'SHORT_ML_THRESHOLD = 0.85', text)

with open('/home/admin/trading_bot/config.py', 'w') as f:
    f.write(text)

print("✅ Configuration tightened to Sniper Mode!")
