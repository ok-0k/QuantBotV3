"""
strategies.py — All trading strategies
Each returns: {"signal": "buy"/"sell"/"none", "price": float, "meta": {...}}

QUANT UPGRADES v2:
 - Wilder's smoothed RSI (fixes Cutler's RSI bias — more stable overbought/oversold readings)
 - Wilder's smoothed ATR  (less noisy, matches how stops/targets are actually calculated)
 - ADX function           (true trend-strength measurement for regime detection)
 - Stochastic RSI         (faster oscillator for generated strategies)
 - YOLO_FIRE strategy     (🔥 experimental high-risk / high-reward momentum chaser)
 - Sharpe ratio + returns_buffer on StrategyParams (risk-adjusted performance tracking)
 - position_size_mult in result dict (YOLO signals the bot to trade bigger)
"""

import math
import random as _random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SHARED INDICATOR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ema(values: list, period: int) -> list:
    """Standard EMA. Seeded from first SMA to reduce warm-up bias."""
    if len(values) < period:
        return [values[-1]] * len(values) if values else []
    k = 2 / (period + 1)
    # Seed from first SMA — avoids artificially pulling EMA toward values[0]
    seed = sum(values[:period]) / period
    result = [values[0]] * (period - 1) + [seed]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def sma(values: list, period: int) -> float:
    if len(values) < period:
        return values[-1] if values else 0.0
    return sum(values[-period:]) / period


def stdev(values: list, period: int) -> float:
    if len(values) < period:
        return 0.0
    window = values[-period:]
    mean = sum(window) / period
    return math.sqrt(sum((x - mean) ** 2 for x in window) / period)


def rsi_val(closes: list, period: int = 14) -> float:
    """
    Wilder's Smoothed RSI — the correct implementation.

    The original code used Cutler's RSI (simple average of gains/losses over
    a rolling window). Wilder's version uses a recursive EMA-style smoothing
    which is what every serious charting platform actually uses.  The
    difference matters most at extreme readings (< 30 / > 70) where signal
    quality is highest.
    """
    if len(closes) < period * 2:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]

    # Seed: simple average of first `period` values
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing (recursive)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-12:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def stoch_rsi_val(closes: list, rsi_period: int = 14, stoch_period: int = 14) -> float:
    """
    Stochastic RSI — applies the stochastic formula to RSI values.
    Returns 0–100.  < 20 = oversold, > 80 = overbought.
    Faster than raw RSI at turning points.
    """
    if len(closes) < rsi_period + stoch_period + 2:
        return 50.0
    # Build rolling RSI series
    rsi_series = []
    for i in range(rsi_period + 1, len(closes) + 1):
        rsi_series.append(rsi_val(closes[:i], rsi_period))
    if len(rsi_series) < stoch_period:
        return 50.0
    window = rsi_series[-stoch_period:]
    lo, hi = min(window), max(window)
    if hi - lo < 1e-9:
        return 50.0
    return (window[-1] - lo) / (hi - lo) * 100


def atr_val(candles: list, period: int = 14) -> float:
    """
    Wilder's Smoothed ATR — matches the industry standard.

    Original code used simple average of last `period` true ranges.
    Wilder's uses recursive smoothing: ATR = (prev_ATR * (n-1) + TR) / n
    This is how all professional platforms compute it and is what stop-loss
    and take-profit levels should be based on.
    """
    if len(candles) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    # Seed: simple average of first `period` TRs
    atr = sum(trs[:period]) / period

    # Wilder's smoothing
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr


def adx_val(candles: list, period: int = 14) -> tuple:
    """
    Average Directional Index — returns (adx, plus_di, minus_di).

    ADX > 25  → strong trend (momentum strategies thrive)
    ADX < 20  → ranging / directionless (mean-reversion strategies thrive)
    plus_di > minus_di → uptrend
    minus_di > plus_di → downtrend

    This is used by the Brain for a much more accurate regime classification
    than a simple MA cross.
    """
    if len(candles) < period * 2 + 1:
        return 20.0, 20.0, 20.0

    plus_dms, minus_dms, trs = [], [], []

    for i in range(1, len(candles)):
        curr, prev = candles[i], candles[i - 1]
        up_move   = curr["high"] - prev["high"]
        down_move = prev["low"]  - curr["low"]
        plus_dms.append(up_move   if (up_move   > down_move and up_move   > 0) else 0)
        minus_dms.append(down_move if (down_move > up_move   and down_move > 0) else 0)
        tr = max(curr["high"] - curr["low"],
                 abs(curr["high"] - prev["close"]),
                 abs(curr["low"]  - prev["close"]))
        trs.append(tr)

    def _wilder_sum(vals, p):
        """Wilder's initial sum then rolling add/subtract."""
        s = sum(vals[:p])
        result = [s]
        for v in vals[p:]:
            s = s - s / p + v
            result.append(s)
        return result

    s_tr  = _wilder_sum(trs, period)
    s_pdm = _wilder_sum(plus_dms, period)
    s_mdm = _wilder_sum(minus_dms, period)

    pdi = [100 * p / (t + 1e-9) for p, t in zip(s_pdm, s_tr)]
    mdi = [100 * m / (t + 1e-9) for m, t in zip(s_mdm, s_tr)]

    dx_vals = [100 * abs(p - m) / (p + m + 1e-9) for p, m in zip(pdi, mdi)]

    if len(dx_vals) < period:
        return 20.0, pdi[-1] if pdi else 20.0, mdi[-1] if mdi else 20.0

    # ADX = Wilder-smoothed DX
    adx = sum(dx_vals[-period:]) / period
    return round(adx, 2), round(pdi[-1], 2), round(mdi[-1], 2)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY BASE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyParams:
    """
    Mutable parameter set — mutated by the Brain.
    Now tracks Sharpe ratio via a rolling returns buffer.
    """
    name: str
    params: dict

    # Performance
    score:          float = 0.0
    weight:         float = 1.0
    total_trades:   int   = 0
    wins:           int   = 0
    losses:         int   = 0
    total_pnl:      float = 0.0
    generation:     int   = 0

    # Sharpe tracking (last 30 normalised trade returns)
    returns_buffer: list  = field(default_factory=list)

    # Peak equity / drawdown (strategy-level)
    peak_score:     float = 0.0
    max_drawdown:   float = 0.0   # worst peak-to-trough score drop

    def __post_init__(self):
        self.is_generated = False
        self.is_yolo      = False
        self.blueprint    = {}

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.wins / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def sharpe(self) -> float:
        """
        Information-ratio-style Sharpe from recent trade returns.
        Positive and > 0.5 is decent. > 1.0 is strong.
        """
        n = len(self.returns_buffer)
        if n < 5:
            return 0.0
        mean  = sum(self.returns_buffer) / n
        var   = sum((r - mean) ** 2 for r in self.returns_buffer) / n
        if var < 1e-12:
            return 10.0 if mean > 0 else -10.0
        return round(mean / math.sqrt(var), 3)

    def record_trade(self, pnl: float, trade_value: float = 1.0):
        self.total_trades += 1
        self.total_pnl    += pnl

        # Normalised return for Sharpe computation (% of trade value)
        norm_r = pnl / max(abs(trade_value), 1.0)
        self.returns_buffer.append(norm_r)
        if len(self.returns_buffer) > 30:
            self.returns_buffer.pop(0)

        if pnl > 0:
            self.wins  += 1
            self.score += 1.0
        else:
            self.losses += 1
            self.score  -= 0.5   # asymmetric — wins rewarded more than losses penalised

        # Decay so recent performance matters more
        self.score *= 0.95

        # Track strategy-level drawdown
        if self.score > self.peak_score:
            self.peak_score = self.score
        drop = self.peak_score - self.score
        if drop > self.max_drawdown:
            self.max_drawdown = drop


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — RSI + EMA CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────

def strategy_rsi_ema(candles: list, p: dict) -> dict:
    closes  = [c["close"]  for c in candles]
    volumes = [c["volume"] for c in candles]
    if len(closes) < 50:
        return {"signal": "none"}

    r        = rsi_val(closes, p["rsi_len"])
    ef       = ema(closes, p["ema_fast"])
    es       = ema(closes, p["ema_slow"])
    vol_avg  = sma(volumes, 20)
    high_vol = volumes[-1] > vol_avg * p["vol_mult"]

    cross_up   = ef[-2] < es[-2] and ef[-1] > es[-1]
    cross_down = ef[-2] > es[-2] and ef[-1] < es[-1]

    buy  = (cross_up  or r < p["rsi_os"]) and high_vol
    sell = (cross_down or r > p["rsi_ob"]) and high_vol

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  closes[-1],
        "meta":   {"rsi": round(r, 2), "ema_fast": round(ef[-1], 4), "ema_slow": round(es[-1], 4)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — BOLLINGER BANDS MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────────────

def strategy_bollinger(candles: list, p: dict) -> dict:
    closes = [c["close"] for c in candles]
    if len(closes) < p["bb_period"] + 10:
        return {"signal": "none"}

    mid   = sma(closes, p["bb_period"])
    std   = stdev(closes, p["bb_period"])
    upper = mid + p["bb_std"] * std
    lower = mid - p["bb_std"] * std
    price, prev = closes[-1], closes[-2]

    buy  = prev < lower and price > lower   # recross back up through lower band
    sell = prev > upper and price < upper   # recross back down through upper band

    bb_pct = (price - lower) / (upper - lower + 1e-9) * 100

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  price,
        "meta":   {"bb_upper": round(upper, 2), "bb_lower": round(lower, 2),
                   "bb_mid": round(mid, 2), "bb_pct": round(bb_pct, 1)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — MACD
# ─────────────────────────────────────────────────────────────────────────────

def strategy_macd(candles: list, p: dict) -> dict:
    closes = [c["close"] for c in candles]
    if len(closes) < p["macd_slow"] + p["macd_signal"] + 10:
        return {"signal": "none"}

    fast_ema    = ema(closes, p["macd_fast"])
    slow_ema    = ema(closes, p["macd_slow"])
    macd_line   = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema(macd_line, p["macd_signal"])

    hist_now  = macd_line[-1] - signal_line[-1]
    hist_prev = macd_line[-2] - signal_line[-2]

    # Histogram zero-cross
    buy  = hist_prev < 0 and hist_now > 0
    sell = hist_prev > 0 and hist_now < 0

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  closes[-1],
        "meta":   {"macd": round(macd_line[-1], 4), "signal": round(signal_line[-1], 4),
                   "hist": round(hist_now, 4)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4 — MOMENTUM / BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────

def strategy_momentum(candles: list, p: dict) -> dict:
    closes  = [c["close"]  for c in candles]
    volumes = [c["volume"] for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]

    if len(closes) < p["mom_period"] + 10:
        return {"signal": "none"}

    mom         = (closes[-1] - closes[-p["mom_period"]]) / closes[-p["mom_period"]] * 100
    recent_high = max(highs[-p["mom_period"]:-1])
    recent_low  = min(lows[-p["mom_period"]:-1])
    vol_surge   = volumes[-1] > sma(volumes, 20) * p["vol_mult"]

    buy  = closes[-1] > recent_high and vol_surge and mom >  p["mom_threshold"]
    sell = closes[-1] < recent_low  and vol_surge and mom < -p["mom_threshold"]

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  closes[-1],
        "meta":   {"momentum_pct": round(mom, 2), "recent_high": round(recent_high, 2),
                   "recent_low": round(recent_low, 2)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5 — MEAN REVERSION + ATR
# ─────────────────────────────────────────────────────────────────────────────

def strategy_mean_reversion(candles: list, p: dict) -> dict:
    closes = [c["close"] for c in candles]
    if len(closes) < p["mr_period"] + 10:
        return {"signal": "none"}

    mean      = sma(closes, p["mr_period"])
    atr       = atr_val(candles, 14)
    price     = closes[-1]
    deviation = (price - mean) / (atr + 1e-9)

    buy  = deviation < -p["mr_threshold"] and closes[-1] > closes[-2]  # starting to recover
    sell = deviation >  p["mr_threshold"] and closes[-1] < closes[-2]  # starting to fall

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  price,
        "meta":   {"deviation_atr": round(deviation, 2), "mean": round(mean, 2), "atr": round(atr, 2)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 6 — 🔥 YOLO FIRE (experimental high-risk / high-reward)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_yolo_fire(candles: list, p: dict) -> dict:
    """
    🔥 YOLO FIRE — Experimental momentum chaser.  Designed to catch large
    moves EARLY before conservative strategies even trigger.

    Philosophy:
    ─ No waiting for indicator confirmation — fires on momentum + volume alone
    ─ Ultra-short lookback (5–15 candles) = faster, noisier signals
    ─ Requires consecutive candles in direction (3 green / 3 red)
    ─ Volume explosion filter (2–4× average) reduces false breakouts
    ─ Returns on_fire=True when BOTH momentum AND volume are extreme
      → bot sizes up to 2.5× normal position when on_fire

    Risks:
    ─ Will generate false signals in choppy / ranging markets
    ─ Stop losses are critical — brain gives it aggressive ATR stops
    ─ Short trial period (8 trades) so the brain kills it fast if it's bad

    That's the deal: high risk, high speed, learns the hard way but fast.
    """
    closes  = [c["close"]  for c in candles]
    volumes = [c["volume"] for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]

    lb = p.get("yolo_lookback", 10)
    if len(closes) < lb + 5:
        return {"signal": "none"}

    price = closes[-1]

    # ── 1. Short-term momentum (last 5 candles) ──────────────────────────────
    short_mom = (closes[-1] - closes[-5]) / (closes[-5] + 1e-9) * 100

    # ── 2. Volume explosion ───────────────────────────────────────────────────
    vol_avg       = sma(volumes, 20)
    vol_ratio     = volumes[-1] / (vol_avg + 1e-9)
    vol_explosion = vol_ratio >= p.get("yolo_vol_mult", 2.5)

    # ── 3. Short-term breakout from recent range ──────────────────────────────
    recent_high = max(highs[-lb:-1]) if lb > 1 else highs[-2]
    recent_low  = min(lows[-lb:-1])  if lb > 1 else lows[-2]

    # ── 4. Ultra-fast RSI (period 7 for speed) ────────────────────────────────
    r7 = rsi_val(closes, 7)

    # ── 5. 3 consecutive candles in one direction ─────────────────────────────
    consec_up   = closes[-1] > closes[-2] > closes[-3]
    consec_down = closes[-1] < closes[-2] < closes[-3]

    # ── 6. ATR context ────────────────────────────────────────────────────────
    atr     = atr_val(candles, 7)
    atr_pct = atr / (price + 1e-9) * 100

    # ── BUY: breakout + volume explosion + momentum + RSI not extreme + green run ──
    buy = (
        closes[-1] > recent_high
        and vol_explosion
        and short_mom >  p.get("yolo_mom_threshold", 0.8)
        and r7        <  p.get("yolo_rsi_max", 78)     # not already overbought
        and consec_up
    )

    # ── SELL: breakdown + same conditions inverted ────────────────────────────
    sell = (
        closes[-1] < recent_low
        and vol_explosion
        and short_mom <  -p.get("yolo_mom_threshold", 0.8)
        and r7        >   p.get("yolo_rsi_min", 22)    # not already oversold
        and consec_down
    )

    # ── 🔥 ON FIRE: volume AND momentum are both extreme → size up hard ───────
    on_fire = (
        vol_ratio     >= p.get("yolo_vol_mult", 2.5) * 1.5
        and abs(short_mom) >= p.get("yolo_fire_threshold", 2.0)
    )

    # Position size multiplier: normal YOLO = 1.5×, on_fire = 2.5×
    pos_mult = 2.5 if on_fire else 1.5

    return {
        "signal":           "buy" if buy else "sell" if sell else "none",
        "price":            price,
        "on_fire":          on_fire,
        "position_size_mult": pos_mult if (buy or sell) else 1.0,
        "meta": {
            "short_mom_pct":    round(short_mom, 2),
            "vol_ratio":        round(vol_ratio, 2),
            "rsi7":             round(r7, 1),
            "atr_pct":          round(atr_pct, 2),
            "on_fire":          on_fire,
            "recent_high":      round(recent_high, 4),
            "recent_low":       round(recent_low,  4),
            "consec_up":        consec_up,
            "consec_down":      consec_down,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def default_strategies() -> list:
    strats = [
        StrategyParams("RSI_EMA", {
            "rsi_len": 14, "rsi_ob": 70.0, "rsi_os": 30.0,
            "ema_fast": 9, "ema_slow": 21, "vol_mult": 1.5,
        }),
        StrategyParams("BOLLINGER", {
            "bb_period": 20, "bb_std": 2.0,
        }),
        StrategyParams("MACD", {
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        }),
        StrategyParams("MOMENTUM", {
            "mom_period": 20, "mom_threshold": 1.5, "vol_mult": 1.5,
        }),
        StrategyParams("MEAN_REVERSION", {
            "mr_period": 30, "mr_threshold": 1.8,
        }),
        StrategyParams("YOLO_FIRE", {
            "yolo_lookback":      10,
            "yolo_vol_mult":       2.5,
            "yolo_mom_threshold":  0.8,
            "yolo_rsi_max":       78.0,
            "yolo_rsi_min":       22.0,
            "yolo_fire_threshold": 2.0,
        }),
    ]
    # Mark YOLO so the brain treats it specially
    for s in strats:
        if s.name == "YOLO_FIRE":
            s.is_yolo = True
    return strats


STRATEGY_FNS = {
    "RSI_EMA":        strategy_rsi_ema,
    "BOLLINGER":      strategy_bollinger,
    "MACD":           strategy_macd,
    "MOMENTUM":       strategy_momentum,
    "MEAN_REVERSION": strategy_mean_reversion,
    "YOLO_FIRE":      strategy_yolo_fire,
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATED STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# ── Entry indicators ──────────────────────────────────────────────────────────

def _entry_rsi_oversold(candles, params):
    closes = [c["close"] for c in candles]
    r = rsi_val(closes, params.get("rsi_len", 14))
    return r < params.get("rsi_os", 35), r > params.get("rsi_ob", 65), {"rsi": round(r, 2)}


def _entry_stoch_rsi(candles, params):
    """StochRSI entry — faster oscillator for short-term reversals."""
    closes = [c["close"] for c in candles]
    sr = stoch_rsi_val(closes, 14, 14)
    buy  = sr < params.get("stoch_os", 20)
    sell = sr > params.get("stoch_ob", 80)
    return buy, sell, {"stoch_rsi": round(sr, 1)}


def _entry_ema_cross(candles, params):
    closes = [c["close"] for c in candles]
    ef = ema(closes, params.get("ema_fast", 8))
    es = ema(closes, params.get("ema_slow", 21))
    buy  = ef[-2] < es[-2] and ef[-1] > es[-1]
    sell = ef[-2] > es[-2] and ef[-1] < es[-1]
    return buy, sell, {"ema_diff": round(ef[-1] - es[-1], 4)}


def _entry_macd_cross(candles, params):
    closes = [c["close"] for c in candles]
    if len(closes) < 40:
        return False, False, {}
    fl  = ema(closes, params.get("macd_fast", 12))
    sl  = ema(closes, params.get("macd_slow", 26))
    ml  = [f - s for f, s in zip(fl, sl)]
    sig = ema(ml, params.get("macd_signal", 9))
    h_now, h_prev = ml[-1] - sig[-1], ml[-2] - sig[-2]
    return h_prev < 0 and h_now > 0, h_prev > 0 and h_now < 0, {"hist": round(h_now, 4)}


def _entry_bb_bounce(candles, params):
    closes = [c["close"] for c in candles]
    period = params.get("bb_period", 20)
    if len(closes) < period + 2:
        return False, False, {}
    mid   = sma(closes, period)
    std   = stdev(closes, period)
    upper = mid + params.get("bb_std", 2.0) * std
    lower = mid - params.get("bb_std", 2.0) * std
    buy  = closes[-2] < lower and closes[-1] > lower
    sell = closes[-2] > upper and closes[-1] < upper
    pct  = (closes[-1] - lower) / (upper - lower + 1e-9) * 100
    return buy, sell, {"bb_pct": round(pct, 1)}


def _entry_momentum_break(candles, params):
    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    period  = params.get("mom_period", 20)
    if len(closes) < period + 2:
        return False, False, {}
    rh  = max(highs[-period:-1])
    rl  = min(lows[-period:-1])
    mom = (closes[-1] - closes[-period]) / (closes[-period] + 1e-9) * 100
    buy  = closes[-1] > rh and mom >  params.get("mom_threshold", 1.5)
    sell = closes[-1] < rl and mom < -params.get("mom_threshold", 1.5)
    return buy, sell, {"momentum": round(mom, 2)}


def _entry_mean_dev(candles, params):
    closes = [c["close"] for c in candles]
    period = params.get("mr_period", 30)
    if len(closes) < period + 2:
        return False, False, {}
    mean = sma(closes, period)
    atr  = atr_val(candles, 14)
    dev  = (closes[-1] - mean) / (atr + 1e-9)
    buy  = dev < -params.get("mr_threshold", 1.8) and closes[-1] > closes[-2]
    sell = dev >  params.get("mr_threshold", 1.8) and closes[-1] < closes[-2]
    return buy, sell, {"deviation": round(dev, 2)}


# ── Filter indicators ─────────────────────────────────────────────────────────

def _filter_volume(candles, params) -> bool:
    vols = [c["volume"] for c in candles]
    return vols[-1] > sma(vols, 20) * params.get("vol_mult", 1.3)


def _filter_trend_aligned(candles, params) -> bool:
    closes = [c["close"] for c in candles]
    long_e = ema(closes, params.get("trend_ema", 50))
    return closes[-1] > long_e[-1]


def _filter_rsi_not_extreme(candles, params) -> bool:
    closes = [c["close"] for c in candles]
    r = rsi_val(closes, 14)
    return params.get("rsi_floor", 25) < r < params.get("rsi_ceil", 75)


def _filter_atr_calm(candles, params) -> bool:
    atr     = atr_val(candles, 14)
    closes  = [c["close"] for c in candles]
    atr_pct = atr / (closes[-1] + 1e-9) * 100
    return atr_pct < params.get("atr_max_pct", 3.0)


def _filter_adx_trending(candles, params) -> bool:
    """Only pass signals when ADX confirms a strong trend."""
    adx, _, _ = adx_val(candles, 14)
    return adx > params.get("adx_min", 22)


def _filter_always(candles, params) -> bool:
    return True


ENTRY_FNS = {
    "RSI_OB_OS":   _entry_rsi_oversold,
    "STOCH_RSI":   _entry_stoch_rsi,
    "EMA_CROSS":   _entry_ema_cross,
    "MACD_CROSS":  _entry_macd_cross,
    "BB_BOUNCE":   _entry_bb_bounce,
    "MOM_BREAK":   _entry_momentum_break,
    "MEAN_DEV":    _entry_mean_dev,
}

FILTER_FNS = {
    "VOLUME":      _filter_volume,
    "TREND":       _filter_trend_aligned,
    "RSI_NEUTRAL": _filter_rsi_not_extreme,
    "ATR_CALM":    _filter_atr_calm,
    "ADX_TREND":   _filter_adx_trending,
    "NONE":        _filter_always,
}

# Logic modes:
#  AND  — entry AND filter (more selective, fewer but better signals)
#  OR   — entry OR filter  (more permissive, catches more moves)
#  GATE — filter must be true; if true, ALSO require entry (adds regime awareness)
LOGIC_MODES = ["AND", "OR", "GATE"]


# Default param pools for generated strategies (including YOLO params so
# generated strategies can explore that region too)
_PARAM_POOLS = {
    "rsi_len":           [10, 12, 14, 16, 20],
    "rsi_os":            [25, 28, 30, 32, 35],
    "rsi_ob":            [65, 68, 70, 72, 75],
    "rsi_floor":         [20, 25, 30],
    "rsi_ceil":          [70, 75, 80],
    "stoch_os":          [15, 20, 25],
    "stoch_ob":          [75, 80, 85],
    "ema_fast":          [5, 7, 8, 9, 10, 12],
    "ema_slow":          [18, 20, 21, 26, 30, 35],
    "trend_ema":         [40, 50, 60, 100],
    "macd_fast":         [8, 10, 12, 14],
    "macd_slow":         [22, 24, 26, 28],
    "macd_signal":       [7, 9, 11],
    "bb_period":         [15, 18, 20, 25],
    "bb_std":            [1.8, 2.0, 2.2, 2.5],
    "mom_period":        [10, 15, 20, 25],
    "mom_threshold":     [1.0, 1.5, 2.0, 2.5],
    "mr_period":         [20, 25, 30, 40],
    "mr_threshold":      [1.5, 1.8, 2.0, 2.5],
    "vol_mult":          [1.2, 1.3, 1.5, 1.8, 2.0],
    "atr_max_pct":       [2.0, 2.5, 3.0, 4.0],
    "adx_min":           [18, 20, 22, 25],
    "yolo_lookback":      [5, 7, 10],                 # Faster breakout detection
    "yolo_vol_mult":      [2.0, 2.5, 3.0],            # Lower volume requirement (2x is enough for a 1m scalp)
    "yolo_mom_threshold": [0.2, 0.3, 0.4, 0.5],       # 0.2% to 0.5% move over 5 mins is realistic
    "yolo_rsi_max":       [75, 80, 85],               # Give it more room before calling it "overbought"
    "yolo_rsi_min":       [15, 20, 25],
    "yolo_fire_threshold":[0.6, 0.8, 1.0],
}


def _random_params() -> dict:
    return {k: _random.choice(v) for k, v in _PARAM_POOLS.items()}


def build_generated_strategy(entry: str, filter_: str, logic: str, params: dict, name: str):
    """
    Returns a callable strategy built from modular building blocks.
    Logic modes:
      AND  — entry must trigger AND filter must pass
      OR   — entry OR filter passes (more sensitive)
      GATE — filter acts as regime gate; entry must ALSO be true (not contrarian any more)
    """
    entry_fn  = ENTRY_FNS[entry]
    filter_fn = FILTER_FNS[filter_]

    def _fn(candles, p):
        if len(candles) < 50:
            return {"signal": "none"}
        try:
            buy_e, sell_e, meta = entry_fn(candles, p)
            filt = filter_fn(candles, p)

            if logic == "AND":
                buy, sell = buy_e and filt, sell_e and filt
            elif logic == "OR":
                buy, sell = buy_e or filt, sell_e or filt
            elif logic == "GATE":
                # Filter is a regime gate: signal only valid when regime matches
                buy, sell = buy_e and filt, sell_e and filt
            else:
                buy, sell = buy_e, sell_e

            meta.update({"entry": entry, "filter": filter_, "logic": logic})
            return {
                "signal": "buy" if buy else "sell" if sell else "none",
                "price":  candles[-1]["close"],
                "meta":   meta,
            }
        except Exception:
            return {"signal": "none"}

    _fn.__name__ = name
    return _fn


def generate_random_strategy(existing_names: set) -> tuple:
    """
    Randomly assemble a new strategy from building blocks.
    Returns (StrategyParams, strategy_fn).
    """
    entry   = _random.choice(list(ENTRY_FNS.keys()))
    filter_ = _random.choice(list(FILTER_FNS.keys()))
    logic   = _random.choice(LOGIC_MODES)
    params  = _random_params()

    short = f"GEN_{entry[:3]}_{filter_[:3]}_{logic}"
    idx, candidate = 1, short
    while candidate in existing_names:
        candidate = f"{short}_{idx}"
        idx += 1
    name = candidate

    sp = StrategyParams(name=name, params=params)
    sp.is_generated = True
    sp.blueprint    = {"entry": entry, "filter": filter_, "logic": logic}

    fn = build_generated_strategy(entry, filter_, logic, params, name)
    return sp, fn
