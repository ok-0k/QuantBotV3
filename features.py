"""
features.py — Vectorized feature engineering pipeline.

All computation uses pandas-ta / NumPy C-extensions — never Python loops.
This module is designed to run inside a ProcessPoolExecutor worker so it
never blocks the asyncio event loop.

Features extracted:
  Momentum  : RSI(7,14,21), MACD histogram, Stoch RSI, ROC(5,10,20)
  Volatility: ATR(14), BB width, BB %B, realised vol(20)
  Trend     : EMA distances (9,21,50,200), ADX, +DI, -DI
  Micro     : Volume ratio, candle body ratio, wick ratios, VWAP deviation
  Lag       : 1/5/10 period log-returns

Binary target (for XGBoost training):
  1 if price rises TARGET_PROFIT_PCT in next N candles without -TARGET_DRAWDOWN_PCT
  0 otherwise
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # type: ignore
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False

from config import (TARGET_FORWARD_CANDLES, TARGET_PROFIT_PCT,
                    TARGET_DRAWDOWN_PCT)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(candles: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """
    Given a list of OHLCV candle dicts, return:
      (feature_df, feature_names)

    The DataFrame has NaN rows dropped.  Use the last row for live inference,
    or the full frame (with targets) for training.
    """
    if len(candles) < 60:
        return pd.DataFrame(), []

    df = pd.DataFrame(candles)
    df = df.rename(columns={"time": "ts"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df.sort_values("ts").reset_index(drop=True)

    feat = pd.DataFrame(index=df.index)

    # ── Momentum ────────────────────────────────────────────────────────────
    for period in (7, 14, 21):
        feat[f"rsi_{period}"] = _rsi(df["close"], period)

    # MACD histogram
    macd_fast, macd_slow, macd_sig = _macd(df["close"], 12, 26, 9)
    feat["macd_hist"]   = macd_fast - macd_sig
    feat["macd_signal"] = macd_sig

    # Stochastic RSI
    feat["stoch_rsi"] = _stoch_rsi(df["close"], 14, 14)

    # Rate of change
    for period in (5, 10, 20):
        feat[f"roc_{period}"] = df["close"].pct_change(period) * 100

    # ── Volatility ───────────────────────────────────────────────────────────
    feat["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)
    feat["atr_pct"] = feat["atr_14"] / df["close"]

    # Bollinger Bands
    bb_mid   = df["close"].rolling(20).mean()
    bb_std   = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    feat["bb_pct_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # Realised volatility (20-period)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    feat["realised_vol"] = log_ret.rolling(20).std() * np.sqrt(1440)  # annualised for 1m

    # ── Trend ────────────────────────────────────────────────────────────────
    for period in (9, 21, 50, 200):
        ema = df["close"].ewm(span=period, adjust=False).mean()
        feat[f"ema_dist_{period}"] = (df["close"] - ema) / (df["close"] + 1e-9)

    adx_df = _adx(df["high"], df["low"], df["close"], 14)
    feat["adx"]      = adx_df["adx"]
    feat["plus_di"]  = adx_df["plus_di"]
    feat["minus_di"] = adx_df["minus_di"]
    feat["di_diff"]  = adx_df["plus_di"] - adx_df["minus_di"]

    # ── Microstructure ───────────────────────────────────────────────────────
    vol_avg          = df["volume"].rolling(20).mean()
    feat["vol_ratio"] = df["volume"] / (vol_avg + 1e-9)

    # Candle body / wick structure
    candle_range     = df["high"] - df["low"] + 1e-9
    feat["body_ratio"]  = abs(df["close"] - df["open"]) / candle_range
    feat["upper_wick"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    feat["lower_wick"]  = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    feat["is_green"]    = (df["close"] > df["open"]).astype(float)

    # VWAP deviation (intra-session approximation — rolling 20-bar VWAP)
    typical  = (df["high"] + df["low"] + df["close"]) / 3
    vwap     = (typical * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-9)
    feat["vwap_dev"] = (df["close"] - vwap) / (vwap + 1e-9)

    # ── Lag returns ──────────────────────────────────────────────────────────
    for lag in (1, 5, 10):
        feat[f"log_ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))

    # ── Consecutive candle count (bullish/bearish run) ────────────────────────
    feat["consec_bull"] = _consec_run(df["close"] > df["open"])
    feat["consec_bear"] = _consec_run(df["close"] < df["open"])

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    feature_names = feat.columns.tolist()
    return feat, feature_names


def compute_targets(candles: list[dict]) -> pd.Series:
    """
    Binary target: 1 if price rises >= TARGET_PROFIT_PCT within
    TARGET_FORWARD_CANDLES candles WITHOUT hitting -TARGET_DRAWDOWN_PCT.

    This bakes risk management into the ML target — the model learns to
    predict entries with immediate favourable asymmetry.
    """
    closes = np.array([c["close"] for c in candles], dtype=float)
    n      = len(closes)
    N      = TARGET_FORWARD_CANDLES
    profit = TARGET_PROFIT_PCT
    dd_lim = TARGET_DRAWDOWN_PCT

    targets = np.zeros(n, dtype=int)
    for i in range(n - N):
        entry = closes[i]
        future = closes[i + 1: i + N + 1]
        pct    = (future - entry) / entry
        # Target=1 only if we hit profit threshold without blowing the drawdown
        if np.any(pct >= profit) and not np.any(pct <= -dd_lim):
            targets[i] = 1

    return pd.Series(targets, name="target")


def get_live_feature_vector(candles: list[dict]) -> np.ndarray | None:
    """
    Return a single 1-D feature vector for the latest closed candle.
    Used for live XGBoost inference.  Returns None if insufficient data.
    """
    feat, _ = compute_features(candles)
    if feat.empty:
        return None
    return feat.iloc[-1].values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL INDICATOR IMPLEMENTATIONS
# All use vectorized Pandas/NumPy — no Python-level loops.
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs     = avg_g / (avg_l + 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26,
          sig: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=sig, adjust=False).mean()
    return macd_line, ema_slow, sig_line


def _stoch_rsi(close: pd.Series, rsi_period: int = 14,
               stoch_period: int = 14) -> pd.Series:
    rsi     = _rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.DataFrame:
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / period
    atr_w    = pd.Series(plus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_s   = pd.Series(plus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_s  = pd.Series(minus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    tr_s     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di  = 100 * plus_s / (tr_s + 1e-9)
    minus_di = 100 * minus_s / (tr_s + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx      = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame({
        "adx":      adx.values,
        "plus_di":  plus_di.values,
        "minus_di": minus_di.values,
    }, index=high.index)


def _consec_run(condition: pd.Series) -> pd.Series:
    """Count of consecutive True values (resets on False)."""
    s     = condition.astype(int)
    groups = (s != s.shift()).cumsum()
    return s.groupby(groups).cumcount() + 1


# ─────────────────────────────────────────────────────────────────────────────
# SAC STATE VECTOR
# ─────────────────────────────────────────────────────────────────────────────

REGIME_ENCODING = {
    "trending_up":   [1.0, 0.0, 0.0, 0.0],
    "trending_down": [0.0, 1.0, 0.0, 0.0],
    "ranging":       [0.0, 0.0, 1.0, 0.0],
    "volatile":      [0.0, 0.0, 0.0, 1.0],
}


def build_sac_state(prob: float, balance_ratio: float,
                    unrealised_pnl_pct: float, drawdown_pct: float,
                    atr_pct: float, adx_norm: float,
                    regime: str, vol_norm: float) -> np.ndarray:
    """
    Construct the 8-dimensional state vector for the SAC actor.

    All inputs normalised to roughly [-1, 1] or [0, 1] range.
    """
    # Clamp everything to reasonable ranges
    state = np.array([
        float(np.clip(prob,             0.0, 1.0)),
        float(np.clip(balance_ratio,       0.0, 1.0)),
        float(np.clip(unrealised_pnl_pct, -1.0, 1.0)),
        float(np.clip(drawdown_pct,        0.0, 1.0)),
        float(np.clip(atr_pct * 100,       0.0, 5.0) / 5.0),  # norm to [0,1]
        float(np.clip(adx_norm / 100,      0.0, 1.0)),
        # Regime: one-hot (4 dims compressed to 1 via dot with weights)
        {"trending_up": 1.0, "trending_down": -1.0,
         "ranging": 0.0, "volatile": -0.5}.get(regime, 0.0),
        float(np.clip(vol_norm,            0.0, 1.0)),
    ], dtype=np.float32)

    return state
