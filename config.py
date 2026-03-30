"""
config.py — All tunable constants in one place.
Edit this file; never hard-code values in the modules.

Pi 5 edge constraints are explicitly documented for every setting.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (keep data off the SD card → USB SSD)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("TRADING_DATA_DIR", "/home/pi/trading_data"))
DB_PATH    = DATA_DIR / "trading.db"
XGB_MODEL  = DATA_DIR / "xgb_model.json"
SAC_WEIGHTS= DATA_DIR / "sac_actor.npz"
LOG_DIR    = Path("/dev/shm/trading_logs")   # tmpfs → zero SD card wear

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_WS_BASE  = "wss://stream.binance.com:9443/stream"
BINANCE_REST     = "https://api.binance.com/api/v3"

SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL  = "1m"
CANDLE_LIMIT = 300   # warm-up history fetched on startup

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
STARTING_CASH       = 10_000.0
SLIPPAGE_PCT        = 0.00015   # realistic for limit-order paper fills
TRADE_SIZE_PCT      = 0.08      # base allocation per trade (8% of cash)
YOLO_TRADE_SIZE_PCT = 0.12      # YOLO_FIRE starts larger (12% of cash)
MAX_TRADE_SIZE_PCT  = 0.25      # hard ceiling (25%)
MAX_OPEN_POSITIONS  = 4

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
STOP_LOSS_ATR_MULT   = 4.5
TAKE_PROFIT_ATR_MULT = 8.0
MAX_HOLD_CANDLES     = 180      # force exit after 3h on 1m chart
MAX_DRAWDOWN_PCT     = 0.20     # circuit breaker trips at 20% drawdown
CB_RECOVERY_PCT      = 0.10     # resumes when DD recovers to 10%

# ─────────────────────────────────────────────────────────────────────────────
# MACHINE LEARNING (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
# Pi 5: nthread=2 prevents thermal throttling; tree_method='hist' is ARM-optimal
XGB_PARAMS = {
    "n_estimators":   80,
    "max_depth":      4,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "eval_metric":    "logloss",
    "tree_method":    "hist",       # fastest on ARM64
    "nthread":        2,            # limits thermal load on Pi 5
    "use_label_encoder": False,
}
XGB_INCREMENTAL_TREES = 10         # new trees added per incremental update
ML_SIGNAL_THRESHOLD   = 0.60       # probability threshold for buy signal
ML_MIN_SAMPLES        = 100        # minimum samples before ML is trusted
ML_UPDATE_EVERY       = 100        # incremental retrain every N closed candles

# Target: price rises X% in N candles without hitting -Y% drawdown
TARGET_FORWARD_CANDLES = 10
TARGET_PROFIT_PCT      = 0.005     # 0.5%
TARGET_DRAWDOWN_PCT    = 0.003     # 0.3% max drawdown during hold

# ─────────────────────────────────────────────────────────────────────────────
# REINFORCEMENT LEARNING (SAC actor — inference only on Pi)
# ─────────────────────────────────────────────────────────────────────────────
SAC_STATE_DIM  = 8    # [ml_prob, balance_ratio, unrealised_pnl_pct, drawdown_pct,
                       #  atr_pct, adx_norm, regime_enc, vol_norm]
SAC_HIDDEN_DIM = 32
SAC_ACTION_DIM = 1    # position fraction ∈ [0, 1]

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE BRAIN
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_THRESHOLD  = 0.30
SOFTMAX_TEMPERATURE = 1.5

# Genetic mutation
MUTATION_PROB  = 0.30
MUTATION_SCALE = 0.15
KILL_THRESHOLD = -10.0

# Strategy lifecycle
TRIAL_TRADES       = 15
YOLO_TRIAL_TRADES  = 8
TRIAL_MIN_WIN_RATE = 0.42
TRIAL_MIN_PNL      = -5.0
MAX_STRATEGIES     = 14
MIN_CORE_STRATEGIES = 5
GENERATE_EVERY     = 50
REPLAY_EVERY       = 30

# ─────────────────────────────────────────────────────────────────────────────
# CONCURRENCY  (Pi 5 thermal management)
# ─────────────────────────────────────────────────────────────────────────────
PROCESS_POOL_WORKERS = 2   # two worker processes for CPU-bound ML inference
                            # keeps two cores for I/O + event loop

CHECK_EVERY_SECS     = 30  # exit monitor cadence (stop-loss / take-profit scan)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8000
SSE_HEARTBEAT_SECS = 2
