import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. THE BRAIN (NumPy Inference Engine)
# ──────────────────────────────────────────────────────────────────────────────
class LiveNumpyActor:
    def __init__(self, npz_path="sac_actor.npz"):
        print(f"[SYSTEM] Booting Neural Network from {npz_path}...")
        data = np.load(npz_path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W_mu, self.b_mu = data["W_mu"], data["b_mu"]
        self.ln1_g, self.ln1_b = data["ln1_gamma"], data["ln1_beta"]
        self.ln2_g, self.ln2_b = data["ln2_gamma"], data["ln2_beta"]

    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mu = x.mean(); sigma = x.std()
        return gamma * (x - mu) / (sigma + eps) + beta

    def predict(self, state_features: np.ndarray) -> float:
        h = state_features @ self.W1 + self.b1
        h = self._layer_norm(h, self.ln1_g, self.ln1_b)
        h = np.maximum(0, h) 
        h = h @ self.W2 + self.b2
        h = self._layer_norm(h, self.ln2_g, self.ln2_b)
        h = np.maximum(0, h) 
        action = np.tanh(h @ self.W_mu + self.b_mu)
        return action[0]

# ──────────────────────────────────────────────────────────────────────────────
# 2. THE EYES (Binance API Fetcher)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_binance_data(symbol="BTCUSDT", interval="5m", limit=100):
    print(f"[API] Fetching live {interval} market data for {symbol}...")
    url = "https://api.binance.com/api/v3/klines"
    response = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(response.json(), columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    return df[["open", "high", "low", "close", "volume"]].astype(float)

# ──────────────────────────────────────────────────────────────────────────────
# 3. THE MATH (13-Point Feature Matrix)
# ──────────────────────────────────────────────────────────────────────────────
def _ema(series, span): return series.ewm(span=span, adjust=False).mean()
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    rs = gain.ewm(com=period-1, min_periods=period).mean() / (loss.ewm(com=period-1, min_periods=period).mean() + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))
def _atr(h, l, c, period=14):
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    print("[SYSTEM] Crunching live indicators...")
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    
    rsi = _rsi(c, 14) / 100.0                        
    ema9_rel  = (_ema(c, 9)  / c) - 1.0                    
    ema21_rel = (_ema(c, 21) / c) - 1.0                    
    ema50_rel = (_ema(c, 50) / c) - 1.0                    
    
    macd_line = _ema(c, 12) - _ema(c, 26)          
    macd_sig  = _ema(macd_line, 9)
    atr_safe  = _atr(h, lo, c, 14).replace(0, np.nan).ffill().fillna(1e-9)
    macd_n    = macd_line / atr_safe               
    signal_n  = macd_sig / atr_safe               
    hist_n    = (macd_line - macd_sig) / atr_safe               
    
    atr_ratio = atr_safe / c                            
    vol_surge = (v / (v.rolling(20).mean().bfill() + 1e-9)).clip(0, 5) 
    log_ret1 = np.log(c / c.shift(1)).fillna(0)      
    log_ret5 = np.log(c / c.shift(5)).fillna(0)      
    
    bb_std  = c.rolling(20).std().fillna(1e-9)
    bb_pctB = ((c - (c.rolling(20).mean() - 2 * bb_std)) / (4 * bb_std + 1e-9)).clip(0, 1)                     
    z_score = (((c - c.rolling(20).mean()) / bb_std).clip(-3, 3)) / 3.0            

    features = pd.concat([rsi, ema9_rel, ema21_rel, ema50_rel, macd_n, signal_n, hist_n, 
                          atr_ratio, vol_surge, log_ret1, log_ret5, bb_pctB, z_score], axis=1)
    return features.dropna().values.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 4. THE LIVE EXECUTION LOOP
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n========================================")
    print("   QUANT BOT V1 - LIVE MARKET FEED")
    print("========================================")
    
    # 1. Boot the brain
    brain = LiveNumpyActor("sac_actor.npz")
    
    # 2. Look at the market
    live_df = fetch_binance_data("BTCUSDT", interval="5m", limit=100)
    
    # 3. Translate the market for the brain
    feature_matrix = build_feature_matrix(live_df)
    current_state = feature_matrix[-1] # Grab the absolute most recent candle
    
    # 4. Make a decision
    signal = brain.predict(current_state)
    
    print("\n--- FINAL DECISION ---")
    print(f"Current BTC Price: ${live_df['close'].iloc[-1]:,.2f}")
    print(f"Network Signal:    {signal:.4f}")
    print("----------------------")
    
    if signal > 0.10:
        print("ACTION: 🟢 INITIATE MARKET BUY")
    elif signal < -0.10:
        print("ACTION: 🔴 INITIATE MARKET SELL")
    else:
        print("ACTION: ⚪ HOLD POSITION")
    print("========================================\n")
