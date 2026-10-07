with open('/home/admin/trading_bot/config.py', 'r') as f:
    content = f.read()

import re
# Lower the Long threshold to 10%
content = re.sub(r'ENSEMBLE_THRESHOLD\s*=\s*[0-9.]+', 'ENSEMBLE_THRESHOLD = 0.10', content)
# Lower the Short threshold to 10%
content = re.sub(r'SHORT_ENSEMBLE_THRESHOLD\s*=\s*[0-9.]+', 'SHORT_ENSEMBLE_THRESHOLD = 0.10', content)

with open('/home/admin/trading_bot/config.py', 'w') as f:
    f.write(content)

print("✅ Gates Lowered: Gen-0 only needs 10% consensus to trade.")
