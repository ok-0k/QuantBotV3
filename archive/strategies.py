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

QUANT UPGRADES v3 (NEW):
 - STRATEGY 7  — VWAP_DEVIATION: Intraday VWAP anchor + standard-deviation bands.
                 Buys mean-reversion dips to -1σ VWAP, sells rips to +1σ VWAP.
                 Also rides momentum breaks beyond ±2σ with trend filter.
 - STRATEGY 8  — MACD_DIVERGENCE: Classic hidden/regular divergence between price
                 and MACD histogram.  Regular bullish divergence (lower-low price,
                 higher-low MACD) = buy.  Regular bearish divergence = sell.
                 Much higher-quality alpha than plain MACD zero-cross.
 - STRATEGY 9  — VOLUME_PROFILE_POC: Approximates the Point-of-Control (price level
                 with highest traded volume over a rolling window) using only OHLCV
                 data, then trades mean-reversion back to the POC or breakouts away
                 from it confirmed by volume + RSI.

FIX v3.1:
 - FIX 2: Sharpe ratio bug fixed in StrategyParams.sharpe property.
          returns_buffer expanded from 30 → 500 entries (capped at 400 on trim).
          Variance floor raised from 1e-12 → 1e-10 to prevent float truncation
          to zero on small-but-real return distributions.  round() removed from
          the property so callers get full precision; display-side rounding is
          left to the caller.

FIX v3.2:
 - FIX 1 (CPU): stoch_rsi_val rewritten from O(n²) to O(n).
          Old code called rsi_val(closes[:i], rsi_period) in a loop — quadratic
          work per tick.  New code runs a single Wilder-smoothing pass over
          deltas, emits one RSI value per step, and keeps only the last
          stoch_period values.  Identical numerical output; ~200× fewer FLOPs
          on a 200-bar buffer.  Critical on a Pi 5 running 50-coin live feed.

 - FIX 2 (Logic): build_generated_strategy GATE mode differentiated from AND.
          AND:  entry AND filter (both must agree — symmetric gate).
          GATE: filter grants directional permission — longs only when filter
                passes (bullish regime), shorts only when filter fails
                (bearish/ranging).  Previously GATE and AND were identical.

 - FIX 3 (Spam): OR logic mode removed from LOGIC_MODES.
          "entry OR filter" caused strategies to fire on every bar where the
          regime filter alone was True, completely decoupled from any entry
          signal.  Root cause of trade-spam on choppy 50-coin feeds.
          LOGIC_MODES is now ["AND", "GATE"]; legacy OR strategies in SQLite
          fall through to the AND fallback in build_generated_strategy.
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

    PERFORMANCE FIX (v3.2):
    The original implementation called rsi_val(closes[:i], rsi_period) in a loop
    over every bar — O(n²) total work.  On a 50-coin live feed this was hammering
    the Pi 5 on every candle update.

    Replaced with a single O(n) pass using Wilder's recursive smoothing directly:
      1. Seed avg_gain / avg_loss from the first `rsi_period` deltas (same as rsi_val).
      2. Walk forward with Wilder smoothing, emitting one RSI value per step.
      3. Keep only the last `stoch_period` RSI values for the stochastic calculation.

    This produces the identical numerical result to the old loop while reducing
    the per-call complexity from O(n²) to O(n).  For n=200 candles that's ~200×
    fewer floating-point operations per call, which matters when called for 50
    symbols every tick.
    """
    min_len = rsi_period * 2 + stoch_period + 2
    if len(closes) < min_len:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    # Seed Wilder averages from the first rsi_period values
    avg_gain = sum(gains[:rsi_period]) / rsi_period
    avg_loss = sum(losses[:rsi_period]) / rsi_period

    # Walk forward, collecting RSI values into a fixed-size deque-style buffer.
    # We only need the last stoch_period values, so we build a plain list and
    # trim — avoids a collections import and is faster on CPython for small sizes.
    rsi_series: list[float] = []

    for i in range(rsi_period, len(deltas)):
        avg_gain = (avg_gain * (rsi_period - 1) + gains[i])  / rsi_period
        avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
        if avg_loss < 1e-12:
            rsi_series.append(100.0)
        else:
            rsi_series.append(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
        # Trim to stoch_period + 1 so we never accumulate unbounded memory
        if len(rsi_series) > stoch_period + 1:
            rsi_series = rsi_series[-(stoch_period + 1):]

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
# NEW v3 INDICATOR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def vwap_val(candles: list, period: int = 0) -> tuple[float, float]:
    """
    Compute anchored VWAP + 1-sigma band over a rolling window.

    Parameters
    ----------
    candles : list of dicts with keys high/low/close/volume
    period  : number of candles to include; 0 = all (session VWAP)

    Returns
    -------
    (vwap, sigma) where sigma is the volume-weighted std-dev of typical price.
    """
    window = candles[-period:] if period > 0 else candles
    if len(window) < 2:
        return candles[-1]["close"], 0.0

    cum_pv  = 0.0
    cum_v   = 0.0
    cum_pv2 = 0.0
    for c in window:
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        v  = c["volume"]
        cum_pv  += tp * v
        cum_pv2 += tp * tp * v
        cum_v   += v

    if cum_v < 1e-12:
        return candles[-1]["close"], 0.0

    vwap  = cum_pv / cum_v
    # Variance of typical price weighted by volume
    var   = max(0.0, cum_pv2 / cum_v - vwap ** 2)
    sigma = math.sqrt(var)
    return vwap, sigma


def approximate_poc(candles: list, period: int = 30, bins: int = 20) -> float:
    """
    Approximate the Point-of-Control (POC) from a rolling OHLCV window.

    Strategy
    --------
    1. Build `bins` equally-spaced price buckets spanning the period's range.
    2. For each candle, distribute its volume proportionally across the buckets
       its high–low range overlaps.
    3. Return the mid-price of the bucket with the most accumulated volume.

    This is a lightweight approximation of a proper Volume Profile — accurate
    enough to identify high-volume price magnets without a tick-data feed.
    """
    window = candles[-period:] if len(candles) >= period else candles
    if len(window) < 5:
        return candles[-1]["close"]

    lo_all = min(c["low"]  for c in window)
    hi_all = max(c["high"] for c in window)
    span   = hi_all - lo_all
    if span < 1e-9:
        return candles[-1]["close"]

    bucket_vol = [0.0] * bins
    bucket_w   = span / bins

    for c in window:
        b_lo = max(0, int((c["low"]  - lo_all) / bucket_w))
        b_hi = min(bins - 1, int((c["high"] - lo_all) / bucket_w))
        n_buckets = max(1, b_hi - b_lo + 1)
        v_share = c["volume"] / n_buckets
        for b in range(b_lo, b_hi + 1):
            bucket_vol[b] += v_share

    poc_bin = bucket_vol.index(max(bucket_vol))
    poc_price = lo_all + (poc_bin + 0.5) * bucket_w
    return poc_price


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY BASE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyParams:
    """
    Mutable parameter set — mutated by the Brain.
    Now tracks Sharpe ratio via a rolling returns buffer.

    FIX 2 (v3.1): Sharpe property rewritten.
      - returns_buffer cap raised from 30 → 500 (trimmed to 400).
        30 trades is statistically meaningless for Sharpe estimation;
        strategies with 50+ trades were cycling out old returns and losing
        signal, or the tiny variance of 30 near-equal returns was hitting
        the old 1e-12 floor and collapsing to 0.00.
      - Variance floor raised to 1e-10 to prevent float underflow on
        small-but-real return distributions (e.g. crypto scalping at 0.1%
        per trade has variance ~1e-6; the old floor was fine, but rounding
        inside record_trade was the real culprit — removed).
      - round() removed from the sharpe property itself.  Callers that need
        display rounding do it themselves (dashboard already does round(...,3)).
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

    # FIX 2: buffer cap raised — 30 was too small for stable Sharpe estimation
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

        FIX 2: removed round() so callers get full float precision.
        Variance floor raised to 1e-10 (was 1e-12) to prevent underflow
        on small-magnitude return distributions being silently zeroed.
        Buffer minimum raised to 5 (unchanged) but effective minimum for
        meaningful Sharpe is ~20 trades — the larger buffer (500) ensures
        we never evict valid history prematurely.
        """
        n = len(self.returns_buffer)
        if n < 5:
            return 0.0
        mean = sum(self.returns_buffer) / n
        # FIX 2: use explicit float division; do NOT round intermediate value
        var  = sum((r - mean) ** 2 for r in self.returns_buffer) / n
        # FIX 2: raised floor from 1e-12 → 1e-10 to catch near-zero variance
        # that would be numerically zero after float truncation at 1e-12
        if var < 1e-10:
            return 10.0 if mean > 0 else -10.0
        # No round() here — callers round for display (dashboard does round(...,3))
        return mean / math.sqrt(var)

    def record_trade(self, pnl: float, trade_value: float = 1.0):
        self.total_trades += 1
        self.total_pnl    += pnl

        # FIX 2: store raw float — do NOT round norm_r before appending.
        # Rounding 0.0023 → 0.00 was the primary cause of Sharpe = 0.00
        # on strategies with small-but-real positive returns.
        norm_r = pnl / max(abs(trade_value), 1.0)
        self.returns_buffer.append(norm_r)

        # FIX 2: buffer cap raised from 30 → 500 (trim to 400)
        # 30 entries was too small: with 50+ trades the buffer cycled
        # continuously, discarding valid history and producing unstable Sharpe.
        if len(self.returns_buffer) > 500:
            self.returns_buffer = self.returns_buffer[-400:]

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
# STRATEGY 7 — VWAP DEVIATION  (NEW v3)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_vwap_deviation(candles: list, p: dict) -> dict:
    """
    VWAP Deviation Strategy — two complementary regimes in one strategy.

    REGIME A — Mean Reversion (fade the band touch):
      Buy when price tags the lower VWAP band (VWAP − 1σ) and RSI is not
      deeply oversold (confirming the move isn't a runaway breakdown).
      Sell when price tags the upper band.

      This is essentially a "rubber band" trade: VWAP is the fairest-price
      anchor for the session; deviations away from it tend to revert.

    REGIME B — Momentum Breakout (ride the band break):
      Buy when price breaks above VWAP + 2σ with a volume surge.
      Sell when price breaks below VWAP − 2σ with a volume surge.

      A breakout beyond 2σ with strong volume indicates institutional
      momentum that typically continues for several more candles.

    Parameters (tunable by the genetic algorithm)
    ─────────────────────────────────────────────
    vwap_period    : rolling window for VWAP calculation (default 50 candles)
    vwap_band_mult : σ multiplier for mean-reversion band (default 1.0)
    vwap_break_mult: σ multiplier for breakout band (default 2.0)
    vwap_vol_mult  : volume filter for breakout regime (default 1.5×)
    rsi_len        : RSI period used as entry quality filter
    """
    if len(candles) < p.get("vwap_period", 50) + 5:
        return {"signal": "none"}

    closes  = [c["close"]  for c in candles]
    volumes = [c["volume"] for c in candles]
    price   = closes[-1]

    vwap, sigma = vwap_val(candles, p.get("vwap_period", 50))

    if sigma < 1e-9:
        return {"signal": "none"}

    band_mult  = p.get("vwap_band_mult",  1.0)
    break_mult = p.get("vwap_break_mult", 2.0)

    upper_rev   = vwap + band_mult  * sigma   # mean-reversion sell band
    lower_rev   = vwap - band_mult  * sigma   # mean-reversion buy band
    upper_break = vwap + break_mult * sigma   # momentum buy breakout level
    lower_break = vwap - break_mult * sigma   # momentum sell breakdown level

    rsi = rsi_val(closes, p.get("rsi_len", 14))

    # Volume confirmation for breakout regime
    vol_avg   = sma(volumes, 20)
    vol_surge = volumes[-1] > vol_avg * p.get("vwap_vol_mult", 1.5)

    prev_price = closes[-2]

    # ── REGIME A: mean reversion ─────────────────────────────────────────────
    rev_buy  = prev_price < lower_rev and price > lower_rev and rsi < 65
    rev_sell = prev_price > upper_rev and price < upper_rev and rsi > 35

    # ── REGIME B: breakout ───────────────────────────────────────────────────
    brk_buy  = prev_price < upper_break and price > upper_break and vol_surge
    brk_sell = prev_price > lower_break and price < lower_break and vol_surge

    buy  = rev_buy  or brk_buy
    sell = rev_sell or brk_sell

    deviation_sigmas = (price - vwap) / sigma

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  price,
        "meta": {
            "vwap":             round(vwap, 4),
            "sigma":            round(sigma, 4),
            "deviation_sigmas": round(deviation_sigmas, 3),
            "upper_rev":        round(upper_rev, 4),
            "lower_rev":        round(lower_rev, 4),
            "regime_A_buy":     rev_buy,
            "regime_B_buy":     brk_buy,
            "rsi":              round(rsi, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 8 — MACD DIVERGENCE  (NEW v3)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_macd_divergence(candles: list, p: dict) -> dict:
    """
    MACD Divergence Strategy — detects regular (classic) divergence between
    price swing lows/highs and the MACD histogram.

    Why divergence > zero-cross
    ───────────────────────────
    A zero-cross signals a trend change AFTER momentum has already shifted.
    Divergence fires EARLIER, during the final leg of a move, when momentum
    is already fading while price is still making new extremes.

    Regular Bullish Divergence (buy signal):
      Price makes a lower low, but the MACD histogram makes a HIGHER low.
      Indicates sellers are losing steam — high-probability reversal setup.

    Regular Bearish Divergence (sell signal):
      Price makes a higher high, but the MACD histogram makes a LOWER high.
      Indicates buyers are losing steam.

    Implementation
    ──────────────
    We look for the pattern across the last `div_lookback` candles by
    comparing the most recent swing extreme to the one before it, using
    a simplified swing detection (local min/max with a tolerance buffer).

    Parameters
    ─────────────────────────────────────────
    macd_fast     : fast EMA period
    macd_slow     : slow EMA period
    macd_signal   : signal line period
    div_lookback  : how many candles back to search for the prior swing (default 20)
    div_tolerance : minimum price-swing magnitude as ATR multiple to qualify (default 0.5)
    """
    closes = [c["close"] for c in candles]
    min_len = p.get("macd_slow", 26) + p.get("macd_signal", 9) + p.get("div_lookback", 20) + 5
    if len(closes) < min_len:
        return {"signal": "none"}

    fast_ema    = ema(closes, p.get("macd_fast", 12))
    slow_ema    = ema(closes, p.get("macd_slow", 26))
    macd_line   = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema(macd_line, p.get("macd_signal", 9))
    histogram   = [m - s for m, s in zip(macd_line, signal_line)]

    lookback  = p.get("div_lookback", 20)
    atr       = atr_val(candles, 14)
    tol       = atr * p.get("div_tolerance", 0.5)   # minimum swing to qualify

    # We need at least lookback bars of histogram
    if len(histogram) < lookback + 2:
        return {"signal": "none"}

    h_recent  = histogram[-lookback:]
    pr_recent = closes[-lookback:]

    # ── Find the most recent low and the prior low within the window ──────────
    # Current: last 3 bars form a local trough (price)
    curr_price_low_idx  = h_recent.index(min(h_recent[-lookback // 2:]))  # recent half
    prior_price_low_idx = h_recent.index(min(h_recent[:lookback // 2]))   # older half

    curr_price_low  = pr_recent[-(lookback // 2) + curr_price_low_idx]   \
                      if curr_price_low_idx < lookback // 2 else pr_recent[-1]

    prior_price_low = pr_recent[prior_price_low_idx]

    curr_hist_low   = h_recent[-(lookback // 2) + curr_price_low_idx]    \
                      if curr_price_low_idx < lookback // 2 else h_recent[-1]
    prior_hist_low  = h_recent[prior_price_low_idx]

    # ── Find the most recent high and the prior high ──────────────────────────
    curr_price_high_idx  = h_recent.index(max(h_recent[-lookback // 2:]))
    prior_price_high_idx = h_recent.index(max(h_recent[:lookback // 2]))

    curr_price_high  = pr_recent[-(lookback // 2) + curr_price_high_idx] \
                       if curr_price_high_idx < lookback // 2 else pr_recent[-1]
    prior_price_high = pr_recent[prior_price_high_idx]

    curr_hist_high   = h_recent[-(lookback // 2) + curr_price_high_idx]  \
                       if curr_price_high_idx < lookback // 2 else h_recent[-1]
    prior_hist_high  = h_recent[prior_price_high_idx]

    # ── Regular Bullish Divergence: lower price low, higher histogram low ─────
    bull_div = (
        curr_price_low  < prior_price_low  - tol    # price made a new lower low
        and curr_hist_low > prior_hist_low + 1e-6   # histogram made higher low (less negative)
        and histogram[-1] < 0                       # still in negative territory (reversal not yet complete)
    )

    # ── Regular Bearish Divergence: higher price high, lower histogram high ───
    bear_div = (
        curr_price_high > prior_price_high + tol     # price made a new higher high
        and curr_hist_high < prior_hist_high - 1e-6  # histogram made lower high (less positive)
        and histogram[-1] > 0                        # still in positive territory
    )

    return {
        "signal": "buy" if bull_div else "sell" if bear_div else "none",
        "price":  closes[-1],
        "meta": {
            "hist_now":        round(histogram[-1], 5),
            "bull_divergence": bull_div,
            "bear_divergence": bear_div,
            "curr_price_low":  round(curr_price_low,  4),
            "prior_price_low": round(prior_price_low, 4),
            "macd":            round(macd_line[-1],   5),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 9 — VOLUME PROFILE POC  (NEW v3)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_volume_profile_poc(candles: list, p: dict) -> dict:
    """
    Volume Profile POC Strategy — trade the Point-of-Control as a magnet.

    The POC is the price level with the most traded volume in the lookback
    window.  It acts as a strong support/resistance and mean-reversion target.

    Two signal modes:

    MODE A — Mean Reversion to POC:
      When price is >1 ATR away from the POC and starts moving back toward it,
      we trade the reversion.  The POC is an institutional reference level that
      often acts as a gravity centre.

    MODE B — POC Breakout:
      When price breaks away from the POC level with above-average volume AND
      the RSI confirms directional bias, we trade the expansion.

    Parameters
    ─────────────────────────────────────────
    poc_period      : rolling window for POC calculation (default 40 candles)
    poc_bins        : number of price buckets for volume profile (default 20)
    poc_atr_thresh  : ATR multiple away from POC to trigger reversion (default 1.0)
    poc_vol_mult    : volume multiplier for breakout confirmation (default 1.5)
    rsi_len         : RSI period for directional confirmation
    """
    poc_period = p.get("poc_period", 40)
    if len(candles) < poc_period + 10:
        return {"signal": "none"}

    closes  = [c["close"]  for c in candles]
    volumes = [c["volume"] for c in candles]
    price   = closes[-1]

    poc   = approximate_poc(candles, period=poc_period, bins=p.get("poc_bins", 20))
    atr   = atr_val(candles, 14)
    rsi   = rsi_val(closes, p.get("rsi_len", 14))

    if atr < 1e-9:
        return {"signal": "none"}

    dist_atr   = (price - poc) / atr          # signed distance from POC in ATR units
    vol_avg    = sma(volumes, 20)
    vol_surge  = volumes[-1] > vol_avg * p.get("poc_vol_mult", 1.5)
    prev_price = closes[-2]

    poc_atr_thresh = p.get("poc_atr_thresh", 1.0)

    # ── MODE A: reversion to POC ─────────────────────────────────────────────
    # Price is below POC by >1 ATR and is ticking back up → buy
    rev_buy  = dist_atr < -poc_atr_thresh and price > prev_price and rsi < 60
    # Price is above POC by >1 ATR and is ticking back down → sell
    rev_sell = dist_atr >  poc_atr_thresh and price < prev_price and rsi > 40

    # ── MODE B: breakout from POC ─────────────────────────────────────────────
    # Clean upside break of POC with volume and RSI momentum
    brk_buy  = prev_price <= poc and price > poc and vol_surge and rsi > 50
    brk_sell = prev_price >= poc and price < poc and vol_surge and rsi < 50

    buy  = rev_buy  or brk_buy
    sell = rev_sell or brk_sell

    return {
        "signal": "buy" if buy else "sell" if sell else "none",
        "price":  price,
        "meta": {
            "poc":          round(poc, 4),
            "dist_atr":     round(dist_atr, 3),
            "rsi":          round(rsi, 1),
            "vol_surge":    vol_surge,
            "mode_A_buy":   rev_buy,
            "mode_B_buy":   brk_buy,
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
        # ── NEW v3 strategies ────────────────────────────────────────────────
        StrategyParams("VWAP_DEV", {
            "vwap_period":     50,
            "vwap_band_mult":   1.0,
            "vwap_break_mult":  2.0,
            "vwap_vol_mult":    1.5,
            "rsi_len":         14,
        }),
        StrategyParams("MACD_DIV", {
            "macd_fast":    12,
            "macd_slow":    26,
            "macd_signal":   9,
            "div_lookback": 20,
            "div_tolerance": 0.5,
        }),
        StrategyParams("VOL_PROFILE_POC", {
            "poc_period":     40,
            "poc_bins":       20,
            "poc_atr_thresh":  1.0,
            "poc_vol_mult":    1.5,
            "rsi_len":        14,
        }),
    ]
    # Mark YOLO so the brain treats it specially
    for s in strats:
        if s.name == "YOLO_FIRE":
            s.is_yolo = True
    return strats


STRATEGY_FNS = {
    "RSI_EMA":          strategy_rsi_ema,
    "BOLLINGER":        strategy_bollinger,
    "MACD":             strategy_macd,
    "MOMENTUM":         strategy_momentum,
    "MEAN_REVERSION":   strategy_mean_reversion,
    "YOLO_FIRE":        strategy_yolo_fire,
    # NEW v3
    "VWAP_DEV":         strategy_vwap_deviation,
    "MACD_DIV":         strategy_macd_divergence,
    "VOL_PROFILE_POC":  strategy_volume_profile_poc,
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

# Logic modes for generated strategies:
#  AND  — entry must trigger AND filter must pass (most selective; cleanest signals)
#  GATE — filter acts as a one-way directional gate:
#           • buy  arm: entry buy  fires only when filter passes (bullish regime)
#           • sell arm: entry sell fires only when filter FAILS  (bearish/non-bull regime)
#         This gives the filter genuine directional permission semantics rather than
#         being identical to AND.  Example: TREND filter passing = bullish regime, so
#         only buy signals are permitted; when the trend filter fails, only sells go through.
#  OR mode REMOVED — "entry OR filter" meant a strategy would fire on every bar where
#         the regime filter alone was true, completely decoupled from any specific entry
#         signal.  This was the root cause of trade-spam on choppy 50-coin feeds.
LOGIC_MODES = ["AND", "GATE"]


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
    "yolo_lookback":      [5, 7, 10],
    "yolo_vol_mult":      [2.0, 2.5, 3.0],
    "yolo_mom_threshold": [0.2, 0.3, 0.4, 0.5],
    "yolo_rsi_max":       [75, 80, 85],
    "yolo_rsi_min":       [15, 20, 25],
    "yolo_fire_threshold":[0.6, 0.8, 1.0],
    # NEW v3 params available to the generated strategy engine
    "vwap_period":        [30, 40, 50, 60],
    "vwap_band_mult":     [0.75, 1.0, 1.25, 1.5],
    "vwap_break_mult":    [1.5, 2.0, 2.5, 3.0],
    "vwap_vol_mult":      [1.2, 1.5, 1.8, 2.0],
    "div_lookback":       [15, 20, 25, 30],
    "div_tolerance":      [0.3, 0.5, 0.75, 1.0],
    "poc_period":         [30, 40, 50, 60],
    "poc_bins":           [15, 20, 25],
    "poc_atr_thresh":     [0.5, 1.0, 1.5, 2.0],
    "poc_vol_mult":       [1.2, 1.5, 1.8, 2.0],
}


def _random_params() -> dict:
    return {k: _random.choice(v) for k, v in _PARAM_POOLS.items()}


def build_generated_strategy(entry: str, filter_: str, logic: str, params: dict, name: str):
    """
    Returns a callable strategy built from modular building blocks.

    Logic modes (OR removed — see LOGIC_MODES):
      AND  — entry must trigger AND filter must pass (most selective)
      GATE — filter is a directional permission gate:
               buy  arm fires when entry buy  is True AND filter passes
               sell arm fires when entry sell is True AND filter FAILS
             Rationale: filters like TREND or ADX_TREND signal a bullish regime
             when True.  In a bullish regime we permit longs but block shorts;
             when the filter fails (bearish/ranging) we permit shorts but block
             longs.  This is fundamentally different from AND (which blocks ALL
             signals when the filter fails) and gives the generated strategy
             genuine directional regime-awareness rather than just a second AND.
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
                # Both entry signal AND filter must agree — most conservative
                buy  = buy_e  and filt
                sell = sell_e and filt
            elif logic == "GATE":
                # Filter grants directional permission:
                #   filt=True  (bullish regime) → allow longs, block shorts
                #   filt=False (bearish/ranging) → allow shorts, block longs
                buy  = buy_e  and filt
                sell = sell_e and not filt
            else:
                # Fallback to AND for any legacy/unknown mode stored in DB
                buy  = buy_e and filt
                sell = sell_e and filt

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