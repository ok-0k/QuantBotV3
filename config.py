"""
config.py — All tunable constants in one place.

SHORTING ADDITIONS:
  ENABLE_SHORTING         — master switch. False = long-only (your current behaviour).
  SHORT_MARGIN_PCT        — collateral reserved per short as % of cash (e.g. 0.20 = 20%).
                            Binance cross-margin requires ~10%, isolated margin ~20-50%.
                            Set conservatively; the SAC actor scales within this ceiling.
  SHORT_STOP_LOSS_ATR_MULT  — ATR multiplier for short stop losses. Slightly tighter than
                              longs (3.5 vs 4.5) because short squeezes are fast and violent.
  SHORT_TAKE_PROFIT_ATR_MULT — ATR multiplier for short take-profit. Slightly tighter than
                               longs (6.0 vs 8.0) because shorts have asymmetric risk
                               (max gain 100%, max loss unlimited).
  SHORT_ML_THRESHOLD / MIN_ML_CONFIDENCE — minimum ml_down = 1−P(long profit) for shorts
                            (e.g. 0.80 ⇒ M80+ tags). False shorts are dangerous in squeezes.
  SHORT_MAX_OPEN          — Maximum simultaneous short positions. Kept lower than longs
                            (5 vs 15) until you have sufficient margin and confidence.
  SHORT_MIN_ADX           — Minimum ADX required before shorting. Only short in confirmed
                            downtrends (ADX > 25 + minus_di > plus_di), never in ranging.
  SHORT_ENSEMBLE_THRESHOLD — Weighted sell signal required to open a short. Higher than
                             the close-long threshold (0.35 vs 0.20) — more conservative.
"""

import os
from pathlib import Path


def _load_env_file() -> None:
    """
    Best-effort KEY=VALUE loader for local secrets/overrides that must never
    live in the repo (e.g. DISCORD_WEBHOOK_URL, BINANCE_TESTNET_API_KEY).

    Path: $TRADING_ENV_FILE, else ~/.config/quant-bot/env. Real environment
    variables always win — values from the file never override them.
    """
    path = Path(os.getenv("TRADING_ENV_FILE") or Path.home() / ".config" / "quant-bot" / "env")
    try:
        if not path.is_file():
            return
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Config must never crash the bot over an optional file.
        pass


_load_env_file()

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
# Treat empty string as unset — Path("") breaks SAC_WEIGHTS / DB resolution in workers.
_td = (os.getenv("TRADING_DATA_DIR") or "").strip()
DATA_DIR = Path(_td) if _td else Path("/home/admin/trading_data")
DB_PATH     = DATA_DIR / "trading.db"
XGB_MODEL   = DATA_DIR / "xgb_model.json"
SAC_WEIGHTS = DATA_DIR / "sac_actor.npz"
_ld = (os.getenv("TRADING_LOG_DIR") or "").strip()
LOG_DIR     = Path(_ld) if _ld else Path("/dev/shm/trading_logs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOSS_AVERSION_PENALTY: float = 1.5
# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
BINANCE_REST    = "https://api.binance.com/api/v3"

SYMBOLS = [
    # Large caps — highest liquidity
    "BTCUSDT",   "ETHUSDT",   "BNBUSDT",   "SOLUSDT",   "XRPUSDT",
    # Mid caps — good volatility
    "DOGEUSDT",  "ADAUSDT",   "AVAXUSDT",  "DOTUSDT",   "MATICUSDT",
    "LINKUSDT",  "LTCUSDT",   "ATOMUSDT",  "UNIUSDT",   "APTUSDT",
    "NEARUSDT",  "FILUSDT",   "INJUSDT",   "OPUSDT",    "ARBUSDT",
    "SEIUSDT",   "SUIUSDT",   "TIAUSDT",   "STXUSDT",   "RUNEUSDT",
    # DeFi
    "AAVEUSDT",  "MKRUSDT",   "SNXUSDT",   "CRVUSDT",   "COMPUSDT",
    # AI / infra
    "FETUSDT",   "RENDERUSDT","WLDUSDT",   "OCEANUSDT", "AGIXUSDT",
    # Gaming / NFT
    "SANDUSDT",  "MANAUSDT",  "AXSUSDT",   "ENJUSDT",   "GALAUSDT",
    "APEUSDT",   "GMTUSDT",   "BLURUSDT",  "ILVUSDT",   "CHZUSDT",
    # Layer 1 alts
    "HBARUSDT",  "EGLDUSDT",  "FLOWUSDT",  "ICPUSDT",   "QNTUSDT",
]

INTERVAL     = "1m"
CANDLE_LIMIT = 300

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
STARTING_CASH       = 10_000.0
# Fix W2: a flat 1.5bps was applied identically to BTC and to the thinnest
# alt in SYMBOLS (e.g. AGIXUSDT, BLURUSDT, GALAUSDT) — unrealistic for the
# less liquid names at the position sizes this bot uses. 8bps (middle of a
# conservative 5-10bps range) is still a flat approximation, not a per-symbol
# liquidity model, but it no longer systematically understates cost across
# the board the way 1.5bps did.
SLIPPAGE_PCT        = 0.0008
TRADE_SIZE_PCT      = 0.2       # base per trade (SAC scales this up/down)
YOLO_TRADE_SIZE_PCT = 0.8
MAX_TRADE_SIZE_PCT  = 0.50       # SAC ceiling per position
MAX_OPEN_POSITIONS  = 15         # hold up to 15 positions simultaneously

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT — LONGS
# ─────────────────────────────────────────────────────────────────────────────
# 1m bars: wider initial stop reduces noise stops; TP ≥ 2× SL for base R:R.
STOP_LOSS_ATR_MULT   = 6.0
TAKE_PROFIT_ATR_MULT = 12.0
MAX_HOLD_CANDLES     = 90
MAX_DRAWDOWN_PCT     = 0.20
CB_RECOVERY_PCT      = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# SHORTING
# ─────────────────────────────────────────────────────────────────────────────

# Master switch — set to True when you're ready to enable shorting.
# When False, every short signal is silently ignored. Your existing
# long-only behaviour is completely unchanged.
ENABLE_SHORTING = True

# Margin collateral reserved per short position, as a fraction of cash.
# 0.20 = 20% of available cash locked per short (conservative isolated margin).
# Binance isolated margin for altcoins typically requires 10-20%.
# Do not set above 0.30 until you have validated performance.
SHORT_MARGIN_PCT = 0.20

# ATR stop/target multipliers for shorts.
# Tighter than longs because:
#   - Short squeezes are fast and can gap straight through your stop.
#   - Asymmetric risk: gains are capped at 100%, losses are theoretically unlimited.
SHORT_STOP_LOSS_ATR_MULT   = 3.5   # stop ABOVE entry; keep TP ≥ 2× this
SHORT_TAKE_PROFIT_ATR_MULT = 10.0  # target BELOW entry (2× short SL mult)

# Minimum bearish tilt for short entry: ml_down = 1 − ml_prob (model P(long profit)).
# 0.80 ⇒ strategy tags M80+ — fewer trades, less fee drag (was ~0.62 in logic).
MIN_ML_CONFIDENCE: float = 0.80

# Legacy alias / dashboard docs: same scale as MIN_ML_CONFIDENCE (ml_down floor).
SHORT_ML_THRESHOLD: float = MIN_ML_CONFIDENCE

# Maximum simultaneous open short positions.
# Keep low (3-5) until your model has proven short-side edge.
SHORT_MAX_OPEN = 5.0

# Minimum ADX required before opening a short.
# Only short confirmed downtrends. Never short ranging markets.
SHORT_MIN_ADX = 25.0

# Ensemble sell-weight required to open a SHORT (not to close a long).
# Higher than ENSEMBLE_THRESHOLD (0.20) to be more selective on shorts.
SHORT_ENSEMBLE_THRESHOLD = 0.10

# Minimum minus_di advantage over plus_di before shorting.
# Requires clear directional dominance on the short side.
# e.g. 5.0 means minus_di must exceed plus_di by at least 5 points.
SHORT_MIN_DI_DIFF = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# MACHINE LEARNING (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      80,
    "max_depth":         3,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "logloss",
    "tree_method":       "hist",
    "nthread":           2,
    "reg_lambda":        5.0,
}
XGB_INCREMENTAL_TREES = 10
# Min P(profitable) to treat as long conviction if wired to entries (sniper mode).
ML_SIGNAL_THRESHOLD = 0.80
ML_MIN_SAMPLES        = 50
ML_UPDATE_EVERY = 3000

TARGET_FORWARD_CANDLES = 10
TARGET_PROFIT_PCT      = 0.004
TARGET_DRAWDOWN_PCT    = 0.003

# ─────────────────────────────────────────────────────────────────────────────
# REINFORCEMENT LEARNING (SAC)
# ─────────────────────────────────────────────────────────────────────────────
SAC_STATE_DIM  = 13
SAC_HIDDEN_DIM = 32
SAC_ACTION_DIM = 1

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE BRAIN
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_THRESHOLD = 0.10
SOFTMAX_TEMPERATURE = 1.0

MUTATION_PROB  = 0.30
MUTATION_SCALE = 0.15
KILL_THRESHOLD = -1.5

TRIAL_TRADES = 8
YOLO_TRIAL_TRADES = 8
TRIAL_MIN_WIN_RATE = 0.55
TRIAL_MIN_PNL       = -5.0
MAX_STRATEGIES      = 14
MIN_CORE_STRATEGIES = 5
GENERATE_EVERY      = 50
REPLAY_EVERY        = 25

# ─────────────────────────────────────────────────────────────────────────────
# CONCURRENCY
# ─────────────────────────────────────────────────────────────────────────────
PROCESS_POOL_WORKERS = 2
CHECK_EVERY_SECS     = 30

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HOST     = "0.0.0.0"
DASHBOARD_PORT     = 8000
SSE_HEARTBEAT_SECS = 2
SYMBOL_COOLDOWN_SECS = 300

# ═════════════════════════════════════════════════════════════════════════════
# V2 — Capital efficiency, margin, correlation, BTC regime, fees, decay
# ═════════════════════════════════════════════════════════════════════════════

# Binance signed SAPI (optional — empty strings → synthetic margin health only)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

# ── FEE ARCHITECTURE (V2.4 — Wave 3 fix) ──────────────────────────────────────
# Fix W3: the previous version of this comment claimed "0.10% per fill" and
# then computed the round-trip using an unexplained 0.06%/leg — a silent BNB/
# VIP-discount assumption that was never gated on the account actually having
# that discount enabled, understating real trading costs by roughly 2x.
#
# Binance spot STANDARD taker fee: 0.10% per fill (VIP 0, no BNB discount).
# Standard round-trip (entry + exit): 2 × 0.10% = 0.20% = 0.0020 of notional.
# This is now the default — paper/testnet economics can no longer look
# rosier than an undiscounted live account would.
#
# Real-world drag example at the standard rate (typical $1,358 trade):
#   Entry fee  = $1,358 × 0.0010 = $1.358
#   Exit fee   = $1,358 × 0.0010 = $1.358   (exit notional ≈ entry notional)
#   Total drag = $2.72
#
# The discounted rate (0.06%/leg = 0.12% round-trip) is only used if
# USE_BNB_FEE_DISCOUNT is explicitly set True below — confirm BNB fee payment
# is actually enabled and funded on the account before turning this on.
#
# Slippage (SLIPPAGE_PCT) is applied to exec_price in _prepare_trade and is
# already baked into the gross PnL before fee deduction, so it is NOT
# double-counted here.
#
# TAKER_FEE_BPS / MAKER_FEE_BPS are legacy V2.2 zero-fee alpha-isolation
# constants kept for module compatibility.  All live fee accounting now flows
# exclusively through FEE_GATE_ROUND_TRIP → accounting_v2.entry_exit_fees_notional.
TAKER_FEE_BPS: float = 0.0   # legacy — DO NOT use for PnL accounting
MAKER_FEE_BPS: float = 0.0   # legacy — DO NOT use for PnL accounting
TAKER_FEE: float = 0.0        # legacy decimal alias
MAKER_FEE: float = 0.0        # legacy decimal alias

# Explicit opt-in only — defaults to False (the honest, undiscounted rate).
# Do not flip this on without confirming BNB fee payment is actually enabled
# and funded on the real account; it silently halves the fee-gate/PnL cost
# assumption used throughout accounting_v2 and the executor's fee-bleed gate.
USE_BNB_FEE_DISCOUNT: bool = os.getenv("USE_BNB_FEE_DISCOUNT", "false").strip().lower() in (
    "1", "true", "yes",
)

# Active round-trip fee constant used by accounting_v2 and the fee gate:
#   fee = (entry_notional + exit_notional) / 2  × FEE_GATE_ROUND_TRIP
# Standard (default): 2 × 0.10%/leg = 0.0020.
# BNB-discount (opt-in, see USE_BNB_FEE_DISCOUNT above): 2 × 0.06%/leg = 0.0012.
FEE_GATE_ROUND_TRIP: float = 0.0012 if USE_BNB_FEE_DISCOUNT else 0.0020

# ── EMERGENCY HARD STOPS ──────────────────────────────────────────────────────
# Instant market exit when open PnL drops below this USD amount.
# Bypasses _exit_in_flight, the fee gate, and all ML/brain gates.
# Override via env var HARD_STOP_LOSS_USD (e.g. export HARD_STOP_LOSS_USD=-20.00).
HARD_STOP_LOSS_USD: float = float(os.getenv("HARD_STOP_LOSS_USD", "-15.00"))

# Real-world max-hold in seconds — forces exit on losing positions held longer than
# this regardless of candle_count (fixes time-freeze bug on WS reconnects).
# Default: 14400 s = 4 hours.  Override via env MAX_HOLD_OPEN_SECS.
MAX_HOLD_OPEN_SECONDS: float = float(os.getenv("MAX_HOLD_OPEN_SECS", "14400"))
# Perpetual funding proxy (spot bot default 0). Set e.g. 1.0 to stress-test carry.
FUNDING_BPS_PER_8H_EST: float = float(os.getenv("FUNDING_BPS_PER_8H", "0"))
# Simple borrow drag on margin positions (annualized, applied in net PnL est.)
MARGIN_BORROW_APR_EST: float = float(os.getenv("MARGIN_BORROW_APR_EST", "0.0"))

# Halt new entries if live margin level (Binance) drops below this (liquidation buffer)
MARGIN_LEVEL_HALT_BELOW: float = float(os.getenv("MARGIN_LEVEL_HALT_BELOW", "1.12"))
# Synthetic halt: (equity−cash)/equity above this ⇒ block (no API keys)
SYNTHETIC_DEPLOYMENT_HALT_ABOVE: float = float(os.getenv("SYNTHETIC_DEPLOYMENT_HALT", "0.90"))

# Aggregate gross exposure cap: sum(|notional|)/equity must stay below this to add a leg
GLOBAL_POSITION_NOTIONAL_CAP: float = float(os.getenv("GLOBAL_NOTIONAL_CAP", "0.82"))

# Correlation gate on log-returns (last ~120 bars)
CORRELATION_THRESHOLD: float = float(os.getenv("CORR_THRESHOLD", "0.90"))
MAX_CORRELATED_OPEN_PEERS: int = int(os.getenv("MAX_CORR_PEERS", "2"))

# BTC / ETH “king” regime — block alt shorts when majors in structural bull leg
BTC_KING_EMA_FAST: int = 50
BTC_KING_EMA_SLOW: int = 200
BTC_KING_SHORT_BLOCK_RATIO: float = 1.015  # fast EMA / slow EMA above this ⇒ risk-on king

# Break-even ratchet trigger (handled in bot). Keep high to avoid noise-triggered BE
# exits that "win" pennies and starve the wider ATR take-profit.
BE_TRIGGER_ATR_MULT: float = 4.5

# Minimum unrealised profit (in ATR units) measured on the *close* price at the
# moment the BE ratchet fires. Prevents wick-triggered BE stops that snap back to
# entry on the very next candle and exit at a fee-loss.
BE_MIN_PROFIT_ATR: float = 1.5
# Fallback absolute minimum profit (fraction of entry price) used when ATR = 0.
BE_MIN_PROFIT_PCT: float = 0.01   # 1.0% of entry price

# ML time-decay: λ = 1 − exp(−hold/halflife); TP pulled toward entry, SL tightened
DECAY_HALFLIFE_CANDLES: float = 240.0
# Minimum candles held before time-decay is allowed to begin. Trades need this
# many bars to develop before the agent is permitted to become impatient.
DECAY_MIN_CANDLES: int = 30
DECAY_TP_PULL_STRENGTH: float = 0.15  # fraction of distance to entry per unit λ
DECAY_SL_TIGHTEN_STRENGTH: float = 0.10

# ═════════════════════════════════════════════════════════════════════════════
# V4 — Execution layer & order-safety protections
# ═════════════════════════════════════════════════════════════════════════════

# Execution backend: "paper" (default — simulated fills, current behaviour) or
# "testnet" (Binance Spot testnet at testnet.binance.vision; requires the
# BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET env vars). There is
# deliberately NO "live" mode.
EXECUTION_MODE: str = os.getenv("EXECUTION_MODE", "paper").strip().lower()

BINANCE_TESTNET_REST = "https://testnet.binance.vision/api/v3"
BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "").strip()
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "").strip()

# Stale-data gate: refuse NEW ENTRIES when the newest candle's open time is
# older than this many seconds (1m bars: 180s = current bar + 2 closed bars).
# Exits are never gated — the bot must always be able to close a position.
STALE_ENTRY_MAX_SECS: float = float(os.getenv("STALE_ENTRY_MAX_SECS", "180"))

# Fix W1: this used to be 0.50 — pure "defense in depth" that never actually
# triggered, since the SAC ceiling (0.35) × max edge multiplier (1.35) tops
# out at 0.4725. That meant a single altcoin could legitimately reach ~47%
# of total equity in one position with no combination of upstream multipliers
# ever hitting this floor. 0.15 makes this a REAL, actively-binding per-symbol
# concentration cap, not just a bug backstop: it now binds on most
# high-conviction entries (which would otherwise size toward the 35% SAC
# ceiling), not only the rare edge-multiplier-boosted tail case. That is a
# deliberate risk-policy change, not just a bugfix — 47 symbols, many of them
# thin-liquidity alts with correlated crypto-wide drawdowns, plus the
# "unlimited loss" asymmetry the SHORT_* comments elsewhere already call out,
# argue for capping single-name exposure well below what the sizing model
# alone would otherwise permit.
MAX_ORDER_EQUITY_FRAC: float = float(os.getenv("MAX_ORDER_EQUITY_FRAC", "0.15"))

# WS reconnect: if the stream was down longer than this, REST-backfill candle
# history before trading again so indicators don't run on a gapped cache.
WS_BACKFILL_AFTER_SECS: float = float(os.getenv("WS_BACKFILL_AFTER_SECS", "90"))

# Testnet adapter behaviour
ORDER_POLL_TIMEOUT_SECS: float = float(os.getenv("ORDER_POLL_TIMEOUT_SECS", "20"))
ORDER_MAX_RETRIES: int = int(os.getenv("ORDER_MAX_RETRIES", "2"))
REST_WEIGHT_LIMIT_PER_MIN: int = int(os.getenv("REST_WEIGHT_LIMIT_PER_MIN", "1100"))

# Anti reward-hacking guardrails:
# - Skip entries whose projected TP edge is too small in USD terms.
# - Penalize tiny positive exits so SAC does not farm "0.01 wins".
MIN_EXPECTED_TP_NET_USD: float = 0.35
MICRO_WIN_USD: float = 0.05
MICRO_WIN_REWARD_PENALTY: float = 0.12

# Adaptive edge learner (online): adjusts sizing and R:R by side/regime.
ADAPTIVE_EDGE_ENABLED: bool = True
EDGE_LEARN_RATE: float = 0.08
EDGE_REWARD_TARGET_PCT: float = 0.0025
EDGE_SIZE_MIN_MULT: float = 0.65
EDGE_SIZE_MAX_MULT: float = 1.35
EDGE_RR_MIN_MULT: float = 0.85
EDGE_RR_MAX_MULT: float = 1.45

# Disable hard max-hold exit — decay + stops replace fixed horizon
ENABLE_HARD_MAX_HOLD_EXIT: bool = os.getenv("ENABLE_HARD_MAX_HOLD", "false").lower() in (
    "1", "true", "yes",
)

# ═════════════════════════════════════════════════════════════════════════════
# V3 — Dynamic position sizing & dynamic R:R (volatility-aware)
# ═════════════════════════════════════════════════════════════════════════════

# Linear ML-conviction sizing band:
#   ml_conf == DYNAMIC_SIZE_FLOOR_CONF (0.80)  -> DYNAMIC_SIZE_MIN_MULT  (1.00x)
#   ml_conf == DYNAMIC_SIZE_CEIL_CONF  (0.99)  -> DYNAMIC_SIZE_MAX_MULT  (2.50x)
# Linearly interpolated in between; clamped at both ends. Multiplied onto the
# existing brain conviction multiplier — never bypasses _SAC_SIZE_CEILING.
DYNAMIC_SIZE_ENABLED: bool = True
DYNAMIC_SIZE_FLOOR_CONF: float = 0.80
DYNAMIC_SIZE_CEIL_CONF: float = 0.99
DYNAMIC_SIZE_MIN_MULT: float = 1.00
DYNAMIC_SIZE_MAX_MULT: float = 2.50

# Volatility-aware R:R multipliers. Vol = ATR / price.
#   vol <= DYNAMIC_RR_VOL_LOW_PCT  -> low-vol regime  (tighten SL, extend TP)
#   vol >= DYNAMIC_RR_VOL_HIGH_PCT -> high-vol regime (widen SL, pull TP in)
#   otherwise: linearly interpolate between the two anchor points.
# These multipliers are applied on top of brain.get_stop_take() — they shift
# the stop/take distances around the entry, never invert them.
DYNAMIC_RR_ENABLED: bool = True
DYNAMIC_RR_VOL_LOW_PCT: float = 0.0030    # 0.30% ATR/price = quiet tape
DYNAMIC_RR_VOL_HIGH_PCT: float = 0.0120   # 1.20% ATR/price = hectic tape
DYNAMIC_RR_LOWVOL_SL_MULT: float = 0.85   # tighter stop in chop
DYNAMIC_RR_LOWVOL_TP_MULT: float = 1.20   # let winners run further
DYNAMIC_RR_HIGHVOL_SL_MULT: float = 1.30  # wider stop in storms
DYNAMIC_RR_HIGHVOL_TP_MULT: float = 0.85  # bank profit faster
# Hard floors so a degenerate vol reading can never collapse R:R.
DYNAMIC_RR_MIN_RR_RATIO: float = 1.50     # tp_dist / sl_dist must stay >= this
