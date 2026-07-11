import sqlite3
import json
import sys
import numpy as np
import xgboost as xgb

# Point to your config
sys.path.append('/home/admin/trading_bot')
from config import DB_PATH

# 1. Crack open the memory vault
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

print("🧠 Waking up the Punisher...")

# Grab the last 100 losing trades (Negative Reward)
cursor.execute("""
    SELECT symbol, state, action, reward 
    FROM rl_experience 
    WHERE reward < 0 
    ORDER BY id DESC LIMIT 100
""")
bad_memories = cursor.fetchall()

if not bad_memories:
    print("✅ No bad trades found. The bot is flawless right now.")
    sys.exit()

print(f"📉 Found {len(bad_memories)} losing trades. Extracting failure states...")

X_bad = []
y_punish = []

for row in bad_memories:
    symbol, state_json, action, reward = row
    
    # Unpack the raw market features (RSI, MACD, Volume, etc.) that tricked the bot
    state_array = json.loads(state_json)
    X_bad.append(state_array)
    
    # We want to train the model that the expected value of this state is highly negative
    # We amplify the pain by multiplying the negative reward
    y_punish.append(reward * 2.0) 

X_bad = np.array(X_bad)
y_punish = np.array(y_punish)

# 2. Load the existing XGBoost brain
model_path = '/home/admin/trading_bot/offline_output/xgb_model.json' # Adjust path if needed
booster = xgb.Booster()
booster.load_model(model_path)

# 3. Force the model to learn from the pain
print("⚡ Retraining XGBoost weights to avoid these setups...")
dtrain = xgb.DMatrix(X_bad, label=y_punish)

# Update the existing model (xgb_model=booster tells it to continue training, not start over)
updated_booster = xgb.train(
    {'learning_rate': 0.05, 'max_depth': 4}, 
    dtrain, 
    num_boost_round=5, 
    xgb_model=booster
)

# 4. Overwrite the brain
updated_booster.save_model(model_path)
print("🦾 Model updated. The bot will now avoid these specific market conditions.")

