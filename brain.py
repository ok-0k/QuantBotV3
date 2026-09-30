"""
brain.py — Quantitative ensemble + risk engine.

Mission-critical contract with bot.py:
  • _top_strategy() picks strategy_name from ensemble["breakdown"] where each item has
    keys: signal ("buy"|"sell"), strategy (str), alloc (float). For SHORT entries the
    winning row must use signal=="sell" or SQLite logs a blank strategy.
  • get_ensemble_signal(symbol, candles, position_side=...) supplies price, regime,
    breakdown, and multi-factor entry logic.
  • get_stop_take(exec_price, candles, is_yolo, side=...) returns initial stop/take;
    bot.py applies additional trailing / BE logic on ticks.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Sequence

import numpy as np

from config import (
    ADAPTIVE_EDGE_ENABLED,
    BTC_KING_EMA_FAST,
    BTC_KING_EMA_SLOW,
    BTC_KING_SHORT_BLOCK_RATIO,
    CB_RECOVERY_PCT,
    EDGE_LEARN_RATE,
    EDGE_REWARD_TARGET_PCT,
    EDGE_RR_MAX_MULT,
    EDGE_RR_MIN_MULT,
    EDGE_SIZE_MAX_MULT,
    EDGE_SIZE_MIN_MULT,
    ENABLE_SHORTING,
    MAX_DRAWDOWN_PCT,
    MIN_ML_CONFIDENCE,
    SAC_STATE_DIM,
    SHORT_STOP_LOSS_ATR_MULT,
    SHORT_TAKE_PROFIT_ATR_MULT,
    STARTING_CASH,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_ATR_MULT,
)
# Brain is otherwise pure logic (no DB access) so it stays trivially unit-
# testable via a bare Brain(); these two are the sole persistence primitives,
# used only for the circuit breaker's peak-equity high-water mark below.
from db import save_brain_key as _db_save_brain_key, load_brain_key as _db_load_brain_key

# Fix O1: _wilder_atr used to be a standalone implementation here that had
# already drifted from bot.py's copy once (see bot.py's Fix #9 changelog)
# before being manually re-aligned by hand. Now the single shared
# implementation lives in indicators.py; imported under the original local
# name so every call site below (compute_sac_state, get_stop_take,
# get_ensemble_signal) is unchanged.
from indicators import wilder_atr as _wilder_atr

log = logging.getLogger(__name__)

# brain_state key for the circuit breaker's all-time peak equity (Fix #21).
_PEAK_EQUITY_BRAIN_KEY = "circuit_breaker_peak_equity_v1"


# ── Pure numerics (Wilder / ADX / EMA) ───────────────────────────────────────


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ohlcv_arrays(candles: Sequence[dict]) -> tuple[np.ndarray, ...]:
    """Stack OHLCV into column vectors for vectorised math."""
    n = len(candles)
    o = np.empty(n)
    h = np.empty(n)
    l = np.empty(n)
    c = np.empty(n)
    v = np.empty(n)
    for i, bar in enumerate(candles):
        o[i] = _safe_float(bar.get("open"))
        h[i] = _safe_float(bar.get("high"))
        l[i] = _safe_float(bar.get("low"))
        c[i] = _safe_float(bar.get("close"))
        v[i] = _safe_float(bar.get("volume"))
    return o, h, l, c, v

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """EMA: more weight on recent prices — standard trend proxy."""
    if len(x) < span:
        return np.full_like(x, np.nan)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _adx_and_di(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14
) -> tuple[float, float, float]:
    """
    ADX + directional indicators (Wilder). ADX measures trend strength (not direction);
    +DI vs −DI identifies bullish vs bearish pressure — we require −DI dominance to short
    against a potential squeeze in an uptrend.
    """
    n = len(c)
    if n < period + 2:
        return 20.0, 25.0, 25.0  # neutral prior — avoid division blow-ups on short history

    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = float(tr[:period].sum())
    p_dm = float(plus_dm[:period].sum())
    m_dm = float(minus_dm[:period].sum())
    dx_vals: list[float] = []
    for i in range(period, len(tr)):
        atr = atr - atr / period + tr[i]
        p_dm = p_dm - p_dm / period + plus_dm[i]
        m_dm = m_dm - m_dm / period + minus_dm[i]
        atr_s = max(atr, 1e-12)
        pdi_i = 100.0 * p_dm / atr_s
        mdi_i = 100.0 * m_dm / atr_s
        dx_vals.append(100.0 * abs(pdi_i - mdi_i) / max(pdi_i + mdi_i, 1e-12))

    atr_f = max(atr, 1e-12)
    pdi = 100.0 * p_dm / atr_f
    mdi = 100.0 * m_dm / atr_f
    # ADX ≈ smoothed mean of DX (last window) — trend strength scalar
    adx = float(np.mean(dx_vals[-period:])) if len(dx_vals) >= period else (
        float(dx_vals[-1]) if dx_vals else 20.0
    )
    return float(adx), float(pdi), float(mdi)


def _relative_volume(v: np.ndarray, lookback: int = 20) -> float:
    """Last bar volume / SMA(volume): >1 means participation spike (squeeze fuel)."""
    if len(v) < lookback + 1:
        return 1.0
    sma = float(np.mean(v[-lookback - 1 : -1]))
    if sma <= 0:
        return 1.0
    return float(v[-1] / sma)


def _log_ret_window(c: np.ndarray, bars: int) -> float:
    """Cumulative log return over `bars` closes — symmetric, scale-free drift."""
    if len(c) < bars + 1:
        return 0.0
    return float(math.log(c[-1] / max(c[-1 - bars], 1e-12)))


class Brain:
    """
    Multi-factor short engine:
      • ML bearish tilt (1 - ml_prob)
      • Trend/microstructure: EMA stack + ADX/DI so we do not short high-ADX bull legs
      • Liquidity shock: high relative volume + positive drift ⇒ skip (classic squeeze setup)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ml_probs: dict[str, float] = {}
        self.current_regime: str = "neutral"
        self.circuit_open: bool = False
        self._peak_equity: float = 0.0
        # Online adaptive edge model keyed by "side:regime" (e.g., "long:trend_up").
        self._edge_profiles: dict[str, dict[str, float]] = {}
        # Rolling peak for drawdown. Starts at 0.0 here (a bare Brain() must stay
        # side-effect-free for unit tests); call restore_peak_equity() once at
        # bot startup to load the persisted high-water mark (Fix #21) — without
        # that call this is RAM-only for the life of the process, same as before.
        print(
            "BOOT: QuantBrain V2 — ML-sized shorts, BTC-king regime gate, ATR risk."
        )

    def update_ml_prob(self, symbol: str, prob: float, *args: Any, **kwargs: Any) -> None:
        self.ml_probs[symbol] = max(0.0, min(1.0, float(prob)))

    def save(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def export_edge_profiles(self) -> dict[str, dict[str, float]]:
        """Serializable snapshot of adaptive edge state."""
        out: dict[str, dict[str, float]] = {}
        for k, p in self._edge_profiles.items():
            out[str(k)] = {
                "rr_mult": float(p.get("rr_mult", 1.0)),
                "size_mult": float(p.get("size_mult", 1.0)),
                "score_ema": float(p.get("score_ema", 0.0)),
                "samples": float(p.get("samples", 0.0)),
            }
        return out

    def import_edge_profiles(self, payload: Any) -> int:
        """Restore adaptive edge state from a dict-like payload."""
        if not isinstance(payload, dict):
            return 0
        restored = 0
        for k, p in payload.items():
            if not isinstance(k, str) or not isinstance(p, dict):
                continue
            rr = float(p.get("rr_mult", 1.0))
            sm = float(p.get("size_mult", 1.0))
            se = float(p.get("score_ema", 0.0))
            n = float(p.get("samples", 0.0))
            self._edge_profiles[k] = {
                "rr_mult": max(EDGE_RR_MIN_MULT, min(EDGE_RR_MAX_MULT, rr)),
                "size_mult": max(EDGE_SIZE_MIN_MULT, min(EDGE_SIZE_MAX_MULT, sm)),
                "score_ema": max(-1.0, min(1.0, se)),
                "samples": max(0.0, n),
            }
            restored += 1
        return restored

    # ── brain_state persistence primitives (Fix #21) ─────────────────────────
    # Best-effort by design: a persistence hiccup (missing schema in a bare
    # unit test, a transient disk error) must never break risk-engine logic,
    # so both methods swallow and log rather than raise.

    def save_brain_key(self, key: str, value: Any) -> None:
        try:
            _db_save_brain_key(key, value)
        except Exception as exc:
            log.warning("Brain.save_brain_key(%s) failed — continuing without persistence: %s", key, exc)

    def load_brain_key(self, key: str, default: Any = None) -> Any:
        try:
            return _db_load_brain_key(key, default)
        except Exception as exc:
            log.warning("Brain.load_brain_key(%s) failed — using default: %s", key, exc)
            return default

    def restore_peak_equity(self, default: float) -> float:
        """
        Restore the circuit breaker's high-water mark from brain_state.

        Call once, early in bot.py's startup() — before the exit monitor
        starts ratcheting it via check_circuit_breaker() — so a restart
        cannot silently re-arm MAX_DRAWDOWN_PCT from post-restart equity.
        `default` should be the best available current-equity estimate at
        boot (or STARTING_CASH if that isn't computable yet); the restored
        peak never sits below it, so a stale/missing value can never imply
        a larger drawdown than what's actually observable right now.
        """
        loaded = self.load_brain_key(_PEAK_EQUITY_BRAIN_KEY, None)
        try:
            loaded_f = float(loaded) if loaded is not None else 0.0
        except (TypeError, ValueError):
            loaded_f = 0.0
        self._peak_equity = max(self._peak_equity, loaded_f, float(default))
        return self._peak_equity

    def check_circuit_breaker(self, equity: float, *args: Any, **kwargs: Any) -> bool:
        """
        Trip on peak-to-trough drawdown; clear when drawdown recovers by CB_RECOVERY_PCT
        (hysteresis avoids chatter at the threshold).
        """
        eq = max(0.0, float(equity))
        prev_peak = self._peak_equity
        self._peak_equity = max(self._peak_equity, eq)
        if self._peak_equity > prev_peak:
            # Fix #21: persist immediately on every new peak so a restart can
            # never forget how high equity has actually been.
            self.save_brain_key(_PEAK_EQUITY_BRAIN_KEY, self._peak_equity)
        peak = max(self._peak_equity, 1e-9)
        dd = (peak - eq) / peak

        if not self.circuit_open and dd >= MAX_DRAWDOWN_PCT:
            self.circuit_open = True
            log.warning("Circuit breaker OPEN — drawdown %.2f%% (limit %.2f%%)", dd * 100, MAX_DRAWDOWN_PCT * 100)
        elif self.circuit_open and dd <= max(0.0, MAX_DRAWDOWN_PCT - CB_RECOVERY_PCT):
            self.circuit_open = False
            log.info("Circuit breaker CLOSED — drawdown recovered to %.2f%%", dd * 100)

        return self.circuit_open

    def compute_sac_state(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Pack SAC_STATE_DIM features for the NumPy actor. Uses positional convention from bot.py:
        compute_sac_state(symbol, candles, cash, unrealised_pnl, total_equity).
        """
        symbol = args[0] if len(args) > 0 else kwargs.get("symbol", "")
        candles = args[1] if len(args) > 1 else kwargs.get("candles") or []
        cash = _safe_float(args[2] if len(args) > 2 else kwargs.get("cash"))
        unreal = _safe_float(args[3] if len(args) > 3 else kwargs.get("unrealised_pnl"))
        teq = _safe_float(args[4] if len(args) > 4 else kwargs.get("total_equity"), STARTING_CASH)

        ml_p = self.ml_probs.get(str(symbol), 0.5)
        vec = np.zeros(SAC_STATE_DIM, dtype=np.float32)
        if not candles or len(candles) < 3:
            vec[0] = np.float32(ml_p)
            vec[1] = np.float32(cash / max(teq, 1e-6))
            vec[2] = np.float32(unreal / max(teq, 1e-6))
            return vec

        _, h, l, c, v = _ohlcv_arrays(candles)
        price = float(c[-1]) if len(c) else 1.0
        atr = _wilder_atr(h, l, c, 14)
        adx, pdi, mdi = _adx_and_di(h, l, c, 14)
        rvol = _relative_volume(v, 20)

        vec[0] = np.float32(ml_p)
        vec[1] = np.float32(cash / max(teq, 1e-6))
        vec[2] = np.float32(unreal / max(teq, 1e-6))
        vec[3] = np.float32(atr / max(price, 1e-12))  # vol as % of price
        vec[4] = np.float32(adx / 100.0)
        vec[5] = np.float32((mdi - pdi) / 100.0)  # bearish DI edge
        vec[6] = np.float32(min(rvol / 3.0, 1.0))  # cap participation spike
        vec[7] = np.float32(_log_ret_window(c, 20) / 0.05)  # normalised drift (~5% move ref)
        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        stack = 0.0
        if len(ema12) and len(ema26) and not math.isnan(ema12[-1]) and not math.isnan(ema26[-1]):
            stack = float((ema12[-1] - ema26[-1]) / max(price, 1e-12))
        vec[8] = np.float32(math.tanh(stack * 50.0))
        vec[9] = np.float32(1.0 if self.current_regime == "trend_down" else 0.0)
        vec[10] = np.float32(1.0 if self.current_regime == "trend_up" else 0.0)
        vec[11] = np.float32(1.0 if self.circuit_open else 0.0)
        vec[12] = np.float32(min(max((1.0 - ml_p) - 0.5, -0.5), 0.5) * 2.0)  # bearish ML tilt
        return vec

    def _conf01(self, confidence_mult: float) -> float:
        # Confidence multiplier usually sits in ~[0.35, 1.45].
        return max(0.0, min(1.0, (float(confidence_mult) - 0.35) / 1.10))

    def _edge_key(self, side: str, regime: str) -> str:
        s = "short" if str(side).lower() == "short" else "long"
        r = str(regime or "ranging")
        return f"{s}:{r}"

    def _edge_profile(self, side: str, regime: str) -> dict[str, float]:
        k = self._edge_key(side, regime)
        p = self._edge_profiles.get(k)
        if p is None:
            p = {"rr_mult": 1.0, "size_mult": 1.0, "score_ema": 0.0, "samples": 0.0}
            self._edge_profiles[k] = p
        return p

    def edge_position_mult(self, side: str, regime: str, confidence_mult: float) -> float:
        """
        Dynamic position scaling:
        - high confidence => larger allocation
        - low confidence => smaller allocation
        - online edge profile nudges per side/regime.
        """
        if not ADAPTIVE_EDGE_ENABLED:
            return 1.0
        c01 = self._conf01(confidence_mult)
        conf_scale = 0.55 + 0.90 * c01
        p = self._edge_profile(side, regime)
        raw = conf_scale * p["size_mult"]
        return max(EDGE_SIZE_MIN_MULT, min(EDGE_SIZE_MAX_MULT, raw))

    def reward(self, *args: Any, **kwargs: Any) -> float:
        """
        Online learner update from realized shaped reward.
        Returns pnl for compatibility with callers.
        """
        pnl = _safe_float(kwargs.get("pnl", args[0] if args else 0.0), 0.0)
        if not ADAPTIVE_EDGE_ENABLED:
            return pnl

        trade_value = _safe_float(kwargs.get("trade_value"), 1.0)
        regime = str(kwargs.get("regime", self.current_regime or "ranging"))
        side = str(kwargs.get("side", "long"))
        p = self._edge_profile(side, regime)

        norm = pnl / max(abs(trade_value), 1.0)
        tgt = max(1e-6, EDGE_REWARD_TARGET_PCT)
        adv = math.tanh((norm - tgt) / tgt)

        p["samples"] += 1.0
        p["score_ema"] = 0.92 * p["score_ema"] + 0.08 * adv
        p["rr_mult"] = max(
            EDGE_RR_MIN_MULT,
            min(EDGE_RR_MAX_MULT, p["rr_mult"] + EDGE_LEARN_RATE * p["score_ema"]),
        )
        p["size_mult"] = max(
            EDGE_SIZE_MIN_MULT,
            min(EDGE_SIZE_MAX_MULT, p["size_mult"] + EDGE_LEARN_RATE * 0.75 * p["score_ema"]),
        )
        return pnl

    def _classify_regime(self, adx: float, ema_fast: float, ema_slow: float, price: float) -> str:
        """ADX gates trend strength; EMA cross gives sign."""
        if adx < 18:
            return "ranging"
        if ema_fast < ema_slow and price < ema_fast:
            return "trend_down"
        if ema_fast > ema_slow and price > ema_fast:
            return "trend_up"
        return "ranging"

    def _king_ema_ratio(self, candles: list[dict], fast: int, slow: int) -> float | None:
        """Fast/slow EMA ratio on closes — structural bull when ratio > 1 + ε."""
        if len(candles) < slow + 5:
            return None
        _, _, _, c, _ = _ohlcv_arrays(candles)
        ef = _ema(c, fast)
        es = _ema(c, slow)
        if math.isnan(ef[-1]) or math.isnan(es[-1]):
            return None
        return float(ef[-1] / max(es[-1], 1e-12))

    def btc_eth_macro_risk_on(
        self, btc_candles: list[dict] | None, eth_candles: list[dict] | None
    ) -> bool:
        """
        “BTC King” filter: when majors trade in sustained bull structure, systematic
        alt shorts load negatively on the same beta factor — block shorts.
        """
        for series in (btc_candles, eth_candles):
            if not series:
                continue
            r = self._king_ema_ratio(series, BTC_KING_EMA_FAST, BTC_KING_EMA_SLOW)
            if r is not None and r >= BTC_KING_SHORT_BLOCK_RATIO:
                return True
        return False

    @staticmethod
    def ml_conviction_size_mult(ml_prob: float, short_score: float) -> float:
        """
        Map ML probability distance from 0.5 → position weight.

        |p−0.5| is a proper scoring-inspired notion of “confidence” on a calibrated
        classifier; combine with structural short_score for final scale ∈ ~[0.45, 1.45].

        Contract: bot.py re-invokes this at order-commit time (`_authoritative_conviction_mult`)
        so live notional cannot silently diverge from ensemble dict serialization.
        """
        conv = 2.0 * abs(float(ml_prob) - 0.5)
        base = 0.42 + 0.58 * (conv ** 1.15)
        struct = 0.62 + 0.38 * min(1.0, max(0.0, float(short_score)))
        return round(min(1.45, max(0.35, base * struct)), 4)

    def _generate_strategy_tag(
        self, regime: str, short_score: float, ml_down: float, block_reason: str | None
    ) -> str:
        """Human-readable, unique-enough tag for SQLite + dashboards."""
        bucket = int(round(min(9, max(0, short_score * 10))))
        mlb = int(round(ml_down * 100))
        suffix = "BLK_" + block_reason if block_reason else f"S{bucket}_M{mlb}"
        return f"QF_{regime[:2].upper()}_{suffix}"

    def get_stop_take(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """
        Volatility-normalised exits: stop distance ∝ ATR (tighter stop in chop if ATR small).
        YOLO widens take-profit multipliers so winners can run vs noise.
        """
        exec_price = kwargs.get("price")
        if exec_price is None and args:
            exec_price = args[0]
        p = _safe_float(exec_price, 0.0)

        candles: list = []
        if len(args) > 1:
            candles = list(args[1]) if args[1] is not None else []
        elif kwargs.get("candles") is not None:
            candles = list(kwargs["candles"])

        is_yolo = bool(kwargs.get("is_yolo", args[2] if len(args) > 2 else False))
        side = str(kwargs.get("side", "long"))
        regime = str(kwargs.get("regime", self.current_regime or "ranging"))
        confidence_mult = _safe_float(kwargs.get("confidence_mult", 1.0), 1.0)

        if p <= 0 or len(candles) < 15:
            # Fallback: tiny % bands only if ATR undefined (degenerate series)
            pad = max(p * 0.005, 1e-8)
            if side == "short":
                return p + pad, p - pad * 1.6
            return p - pad, p + pad * 1.6

        _, h, l, c, _ = _ohlcv_arrays(candles)
        atr = _wilder_atr(h, l, c, 14)
        # Floor distance so we never place a zero-width stop on flat tape
        min_pct = p * 0.002
        atr_eff = max(atr, min_pct)

        # Risk/reward asymmetry: shorts use config SHORT_* (tighter — squeeze risk)
        c01 = self._conf01(confidence_mult)
        p_edge = self._edge_profile(side, regime)
        rr_edge = p_edge["rr_mult"] if ADAPTIVE_EDGE_ENABLED else 1.0
        conf_rr = 0.90 + 0.40 * c01      # strong setups let TP run further
        conf_sl = 1.08 - 0.18 * c01      # weaker setups get a bit more breathing room
        if side == "short":
            sl_m = SHORT_STOP_LOSS_ATR_MULT * conf_sl * (1.15 if is_yolo else 1.0)
            tp_m = SHORT_TAKE_PROFIT_ATR_MULT * rr_edge * conf_rr * (1.25 if is_yolo else 1.0)
            stop = p + sl_m * atr_eff
            take = p - tp_m * atr_eff
        else:
            sl_m = STOP_LOSS_ATR_MULT * conf_sl * (1.1 if is_yolo else 1.0)
            tp_m = TAKE_PROFIT_ATR_MULT * rr_edge * conf_rr * (1.2 if is_yolo else 1.0)
            stop = p - sl_m * atr_eff
            take = p + tp_m * atr_eff

        # Enforce minimum base R:R floor no matter what the learner suggests.
        if tp_m < 2.0 * sl_m:
            tp_m = 2.0 * sl_m
            if side == "short":
                take = p - tp_m * atr_eff
            else:
                take = p + tp_m * atr_eff

        return stop, take

    def get_ensemble_signal(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        symbol = kwargs.get("symbol")
        if symbol is None and args:
            symbol = args[0]
        symbol = str(symbol or "")

        candles = args[1] if len(args) > 1 else kwargs.get("candles") or []
        if not isinstance(candles, list):
            candles = list(candles)
        position_side = kwargs.get("position_side")

        ml_prob = kwargs.get("ml_prob")
        if ml_prob is None:
            ml_prob = self.ml_probs.get(symbol, 0.5)
        ml_prob = max(0.0, min(1.0, _safe_float(ml_prob, 0.5)))
        ml_down = 1.0 - ml_prob

        current_price = 0.0
        if candles:
            current_price = _safe_float(candles[-1].get("close"), 0.0)

        block_reason: str | None = None
        short_score = 0.0
        regime = "ranging"

        # Default: flat — breakdown empty so _top_strategy falls through only when we trade
        breakdown: list[dict[str, Any]] = []
        signal = "none"
        strategy_name = self._generate_strategy_tag(regime, 0.0, ml_down, "NOOP")

        if not ENABLE_SHORTING:
            block_reason = "shorting_disabled"
            strategy_name = self._generate_strategy_tag(regime, 0.0, ml_down, block_reason)
            return self._pack(
                symbol,
                signal,
                strategy_name,
                breakdown,
                current_price,
                regime,
                ml_prob,
                block_reason,
            )

        min_bars = 60
        if len(candles) < min_bars:
            block_reason = "warmup"
            strategy_name = self._generate_strategy_tag(regime, 0.0, ml_down, block_reason)
            return self._pack(
                symbol,
                signal,
                strategy_name,
                breakdown,
                current_price,
                regime,
                ml_prob,
                block_reason,
            )

        _, h, l, c, v = _ohlcv_arrays(candles)
        price = float(c[-1])
        atr = _wilder_atr(h, l, c, 14)
        adx, pdi, mdi = _adx_and_di(h, l, c, 14)
        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        ef = float(ema12[-1]) if not math.isnan(ema12[-1]) else price
        es = float(ema26[-1]) if not math.isnan(ema26[-1]) else price
        regime = self._classify_regime(adx, ef, es, price)
        self.current_regime = regime

        rvol = _relative_volume(v, 20)
        ret5 = _log_ret_window(c, 5)
        ret20 = _log_ret_window(c, 20)

        # --- Anti “short the steamroller” filter ---
        # High RVOL + positive drift = aggressive bids; shorts face asymmetric squeeze risk.
        volume_heat = rvol > 1.75
        bull_drift = ret5 > 0.0008 and ret20 > 0.003
        ema_bull_stack = ef > es and price > ef
        if volume_heat and (bull_drift or ema_bull_stack):
            block_reason = "vol_uptrend"
        # Strong bullish DI dominance: trend participation is upward.
        elif pdi > mdi + 8.0 and adx > 22:
            block_reason = "bull_di"
        # Do not add to shorts if ADX says strong uptrend with price above fast EMA.
        elif regime == "trend_up" and adx > 26 and price > ef:
            block_reason = "adx_bull_trend"

        # Composite score ∈ [0,1]: bearish ML + structure + (optionally) bearish DI
        ml_component = max(0.0, (ml_down - 0.52) / 0.35)  # 0 at ~0.52, 1 by ~0.87
        ml_component = min(1.0, ml_component)
        struct_component = 0.0
        if mdi > pdi:
            struct_component += 0.35 * min((mdi - pdi) / 40.0, 1.0)
        if ef < es:
            struct_component += 0.25
        if regime == "trend_down":
            struct_component += 0.25
        if adx > 20:
            struct_component += 0.15 * min((adx - 20) / 30.0, 1.0)
        struct_component = min(1.0, struct_component)

        short_score = 0.55 * ml_component + 0.45 * struct_component

        # Entry: structure + high bearish ML tilt (config MIN_ML_CONFIDENCE ≈ M80+ tags).
        ml_gate = ml_down >= MIN_ML_CONFIDENCE
        struct_gate = short_score >= 0.42 and mdi >= pdi - 2.0
        allow = block_reason is None and struct_gate and ml_gate

        btc_candles = kwargs.get("btc_candles") or kwargs.get("btc_king_candles")
        eth_candles = kwargs.get("eth_candles")
        if allow and self.btc_eth_macro_risk_on(
            btc_candles if isinstance(btc_candles, list) else None,
            eth_candles if isinstance(eth_candles, list) else None,
        ):
            block_reason = "btc_king"
            allow = False

        if allow:
            signal = "short"
            strategy_name = self._generate_strategy_tag(regime, short_score, ml_down, None)
            # CRITICAL: bot.py _top_strategy() reads breakdown[*].strategy for "sell" arms
            breakdown = [
                {
                    "signal": "sell",
                    "strategy": strategy_name,
                    "alloc": float(max(0.05, min(1.0, short_score))),
                }
            ]
        else:
            signal = "none"
            br = block_reason or "filters"
            strategy_name = self._generate_strategy_tag(regime, short_score, ml_down, br)

        return self._pack(
            symbol,
            signal,
            strategy_name,
            breakdown,
            current_price,
            regime,
            ml_prob,
            block_reason,
            short_score=short_score,
            atr_pct=atr / max(price, 1e-12),
            adx=adx,
        )

    def _pack(
        self,
        symbol: str,
        signal: str,
        strategy_name: str,
        breakdown: list[dict[str, Any]],
        current_price: float,
        regime: str,
        ml_prob: float,
        block_reason: str | None,
        **extra: Any,
    ) -> dict[str, Any]:
        # V2: ML conviction × structure — feeds multiplicative SAC sizing in bot.py
        ss = float(extra.get("short_score", 0.0))
        cm = self.ml_conviction_size_mult(ml_prob, ss)
        return {
            "signal": signal,
            "strategy": strategy_name,
            "symbol": symbol,
            "price": current_price,
            "regime": regime,
            "on_fire": False,
            "position_size_mult": cm,
            "buy_weight": 0.0,
            "sell_weight": float(extra.get("short_score", 0.0)),
            "ml_prob": round(ml_prob, 6),
            "breakdown": breakdown,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "block_reason": block_reason,
            "meta": {k: extra[k] for k in ("atr_pct", "adx", "short_score") if k in extra},
        }
