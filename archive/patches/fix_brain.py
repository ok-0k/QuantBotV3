import re

with open('/home/admin/trading_bot/brain.py', 'r') as f:
    content = f.read()

# 1. Restore the proper imports at the top
content = re.sub(r'from config import \([\s\S]*?YOLO_TRIAL_TRADES[\s\S]*?\)',
'''from config import (
    MAX_POSITIONS, MAX_ALLOCATION, POSITION_SIZE_MULTIPLIER, ENABLE_SHORTING,
    SHORT_ML_THRESHOLD, SHORT_ENSEMBLE_THRESHOLD, SHORT_MIN_ADX,
    YOLO_TRIAL_TRADES
)''', content)

# 2. Hardcode the Gate logic to our new "God Mode" values
content = re.sub(r'gate_a = \(ENABLE_SHORTING[\s\S]*?adx >= SHORT_MIN_ADX\)',
                 'gate_a = (True and ml_down_prob > 0.80 and adx >= SHORT_MIN_ADX)', content)

content = re.sub(r'gate_b = \(ENABLE_SHORTING[\s\S]*?sell_weight > SHORT_ENSEMBLE_THRESHOLD\)',
                 'gate_b = (True and sell_weight >= 0.0)', content)

with open('/home/admin/trading_bot/brain.py', 'w') as f:
    f.write(content)
