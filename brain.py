"""
brain.py — The cognitive core of the trading system.

Orchestrates:
  1. Regime detection (ADX-based)
  2. XGBoost ML signal (probability of profitable entry)
  3. SAC actor position sizing (risk-adjusted, continuous action)
  4. Heuristic strategy ensemble (with softmax allocation)
  5. Genetic mutation & strategy lifecycle management
  6. Circuit breaker (max drawdown protection)
  7. Anti-correlation guard (avoid doubling into correlated pairs)
  8. Replay buffer review (reinforces winners, mutates chronic losers)

All CPU-bound work (ML inference, RL forward pass) is dispatched to
ProcessPoolExecutor workers by the bot loop — this class holds state only.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config import (
    KILL_THRESHOLD, MUTATION_PROB, MUTATION_SCALE, SOFTMAX_TEMPERATURE,
    ENSEMBLE_THRESHOLD, MAX_STRATEGIES, MIN_CORE_STRATEGIES,
    TRIAL_TRADES, YOLO_TRIAL_TRADES, TRIAL_MIN_WIN_RATE, TRIAL_MIN_PNL,
    GENERATE_EVERY, REPLAY_EVERY, MAX_DRAWDOWN_PCT, CB_RECOVERY_PCT,
    STOP_LOSS_ATR_MULT, TAKE_PROFIT_ATR_MULT, MAX_HOLD_CANDLES,
    ML_SIGNAL_THRESHOLD,
)
from db import (save_brain_key, load_brain_key, get_ml_prob,
                log_rl_experience)
from features import build_sac_state
from strategies import (
    StrategyParams, default_strategies, generate_random_strategy,
    STRATEGY_FNS, build_generated_strategy, adx_val, atr_val,
)

log = logging.getLogger(__name__)

REGIMES = ["trending_up", "trending_down", "ranging", "volatile"]

YOLO_MUTATION_PROB  = 0.50
YOLO_MUTATION_SCALE = 0.40


class Brain:
    """
    Central intelligence — stateful, single-instance per process.
    All heavy ML/RL inference happens in worker processes dispatched by bot.py.
    """

    def __init__(self) -> None:
        self.strategies:     list[StrategyParams] = []
        self.regime_weights: dict = {}
        self.replay_buffer:  list = []
        self.total_trades:   int  = 0
        self.current_regime: str  = "ranging"
        self.regime_history: list = []
        self.mutation_log:   list = []
        self.graveyard:      list = []
        self.generation_log: list = []
        self._generated_fns: dict = {}

        # Circuit breaker
        self.peak_equity:  float = 0.0
        self.circuit_open: bool  = False
        self.cb_log:       list  = []

        # ML prediction cache per symbol
        self._ml_probs:    dict[str, float] = {}

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE (SQLite-backed)
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = load_brain_key("brain_state")
        if data:
            self.strategies     = [self._deserialise(s) for s in data["strategies"]]
            self.regime_weights = data["regime_weights"]
            self.total_trades   = data["total_trades"]
            self.current_regime = data.get("current_regime", "ranging")
            self.regime_history = data.get("regime_history", [])
            self.mutation_log   = data.get("mutation_log", [])
            self.graveyard      = data.get("graveyard", [])
            self.generation_log = data.get("generation_log", [])
            self.peak_equity    = data.get("peak_equity", 0.0)
            self.circuit_open   = data.get("circuit_open", False)
            self.cb_log         = data.get("cb_log", [])

            for s in self.strategies:
                if getattr(s, "is_generated", False) and s.name not in STRATEGY_FNS:
                    bp = getattr(s, "blueprint", {})
                    if bp.get("entry") and bp.get("filter") and bp.get("logic"):
                        self._generated_fns[s.name] = build_generated_strategy(
                            bp["entry"], bp["filter"], bp["logic"], s.params, s.name)
        else:
            self.strategies     = default_strategies()
            self.regime_weights = {r: {s.name: 1.0 for s in self.strategies}
                                   for r in REGIMES}

        replay = load_brain_key("replay_buffer") or []
        self.replay_buffer = replay

    def save(self) -> None:
        save_brain_key("brain_state", {
            "strategies":     [self._serialise(s) for s in self.strategies],
            "regime_weights": self.regime_weights,
            "total_trades":   self.total_trades,
            "current_regime": self.current_regime,
            "regime_history": self.regime_history[-200:],
            "mutation_log":   self.mutation_log[-50:],
            "graveyard":      self.graveyard[-100:],
            "generation_log": self.generation_log[-50:],
            "peak_equity":    self.peak_equity,
            "circuit_open":   self.circuit_open,
            "cb_log":         self.cb_log[-20:],
        })
        save_brain_key("replay_buffer", self.replay_buffer[-500:])

    def _serialise(self, s: StrategyParams) -> dict:
        d = s.__dict__.copy()
        d["is_generated"] = getattr(s, "is_generated", False)
        d["is_yolo"]      = getattr(s, "is_yolo", False)
        d["blueprint"]    = getattr(s, "blueprint", {})
        return d

    def _deserialise(self, d: dict) -> StrategyParams:
        is_gen  = d.pop("is_generated", False)
        is_yolo = d.pop("is_yolo", False)
        bp      = d.pop("blueprint", {})
        d.setdefault("returns_buffer", [])
        d.setdefault("peak_score", 0.0)
        d.setdefault("max_drawdown", 0.0)
        sp = StrategyParams(**d)
        sp.is_generated = is_gen
        sp.is_yolo      = is_yolo
        sp.blueprint    = bp
        return sp

    # ─────────────────────────────────────────────────────────────────────────
    # CIRCUIT BREAKER
    # ─────────────────────────────────────────────────────────────────────────

    def check_circuit_breaker(self, current_equity: float) -> bool:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            if self.circuit_open:
                if current_equity >= self.peak_equity * (1 - CB_RECOVERY_PCT):
                    self.circuit_open = False
                    self.cb_log.append({"time": _now(), "event": "reset",
                                        "equity": current_equity})
                    log.warning("🟢 CIRCUIT BREAKER RESET — equity $%.2f", current_equity)

        if self.peak_equity > 0 and not self.circuit_open:
            dd = (self.peak_equity - current_equity) / self.peak_equity
            if dd >= MAX_DRAWDOWN_PCT:
                self.circuit_open = True
                self.cb_log.append({"time": _now(), "event": "open",
                                    "drawdown_pct": round(dd * 100, 2),
                                    "peak": self.peak_equity,
                                    "equity": current_equity})
                log.warning("🔴 CIRCUIT BREAKER OPEN — drawdown %.1f%% "
                            "(peak $%.2f → $%.2f)",
                            dd * 100, self.peak_equity, current_equity)

        return self.circuit_open

    # ─────────────────────────────────────────────────────────────────────────
    # STOP LOSS / TAKE PROFIT  (ATR-based)
    # ─────────────────────────────────────────────────────────────────────────

    def get_stop_take(self, entry_price: float, candles: list[dict],
                      is_yolo: bool = False) -> tuple[float, float]:
        atr = atr_val(candles, 14)
        if atr == 0:
            atr = entry_price * 0.01

        sl_mult = STOP_LOSS_ATR_MULT   * (0.6 if is_yolo else 1.0)
        tp_mult = TAKE_PROFIT_ATR_MULT * (1.5 if is_yolo else 1.0)

        stop   = entry_price - sl_mult * atr
        target = entry_price + tp_mult * atr
        return round(stop, 6), round(target, 6)

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME DETECTION  (ADX + realised vol)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_regime(self, candles: list[dict]) -> str:
        if len(candles) < 30:
            return "ranging"

        closes = [c["close"] for c in candles]
        adx, plus_di, minus_di = adx_val(candles, 14)

        rets = [(closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)
                for i in range(1, len(closes))]
        vol = math.sqrt(sum(r**2 for r in rets[-20:]) / 20) * 100

        if vol > 3.5:
            regime = "volatile"
        elif adx > 25 and plus_di > minus_di:
            regime = "trending_up"
        elif adx > 25 and minus_di > plus_di:
            regime = "trending_down"
        else:
            regime = "ranging"

        self.current_regime = regime
        self.regime_history.append({
            "time": _now(), "regime": regime,
            "adx": round(adx, 1), "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1), "vol": round(vol, 3),
        })
        return regime

    # ─────────────────────────────────────────────────────────────────────────
    # SOFTMAX ALLOCATION  (Sharpe-adjusted + regime weights)
    # ─────────────────────────────────────────────────────────────────────────

    def _softmax(self, scores: list[float],
                 temp: float = SOFTMAX_TEMPERATURE) -> list[float]:
        scaled = [s / temp for s in scores]
        max_s  = max(scaled)
        exps   = [math.exp(s - max_s) for s in scaled]
        total  = sum(exps) + 1e-9
        return [e / total for e in exps]

    def get_allocations(self, regime: str) -> dict[str, float]:
        weights = self.regime_weights.get(regime, {s.name: 1.0 for s in self.strategies})
        composite = []
        for s in self.strategies:
            rw      = weights.get(s.name, 1.0)
            score_w = s.score * 0.10
            sharpe_w = max(s.sharpe, 0) * 0.15
            composite.append(rw + score_w + sharpe_w)

        probs = self._softmax(composite)
        return {s.name: round(p, 4) for s, p in zip(self.strategies, probs)}

    # ─────────────────────────────────────────────────────────────────────────
    # ML + RL POSITION SIZING
    # ─────────────────────────────────────────────────────────────────────────

    def update_ml_prob(self, symbol: str, prob: float) -> None:
        """Store latest XGBoost probability for a symbol (updated async)."""
        self._ml_probs[symbol] = prob

    def get_ml_prob(self, symbol: str) -> float:
        return self._ml_probs.get(symbol, 0.5)

    def compute_sac_state(self, symbol: str, candles: list[dict],
                          cash: float, unrealised_pnl: float,
                          current_equity: float) -> np.ndarray:
        """Build the 8-dim state vector for the SAC actor."""
        ml_prob      = self.get_ml_prob(symbol)
        balance_ratio = cash / max(current_equity, 1.0)
        pnl_pct      = unrealised_pnl / max(current_equity, 1.0)
        dd_pct       = max(0.0, (self.peak_equity - current_equity)
                         / max(self.peak_equity, 1.0))

        closes = [c["close"] for c in candles]
        atr    = atr_val(candles, 14)
        atr_pct = atr / (closes[-1] + 1e-9)

        adx, _, _ = adx_val(candles, 14)

        rets = [(closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)
                for i in range(max(1, len(closes)-20), len(closes))]
        vol  = math.sqrt(sum(r**2 for r in rets) / max(len(rets), 1)) * 100

        return build_sac_state(
            ml_prob=ml_prob,
            balance_ratio=balance_ratio,
            unrealised_pnl_pct=pnl_pct,
            drawdown_pct=dd_pct,
            atr_pct=atr_pct,
            adx_norm=adx,
            regime=self.current_regime,
            vol_norm=min(vol / 5.0, 1.0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ENSEMBLE SIGNAL AGGREGATION
    # ─────────────────────────────────────────────────────────────────────────

    def get_ensemble_signal(self, symbol: str, candles: list[dict]) -> dict:
        regime      = self.detect_regime(candles)
        allocations = self.get_allocations(regime)
        ml_prob     = self.get_ml_prob(symbol)

        buy_weight   = 0.0
        sell_weight  = 0.0
        breakdown    = []
        on_fire      = False
        max_pos_mult = 1.0

        for strat in self.strategies:
            fn = STRATEGY_FNS.get(strat.name) or self._generated_fns.get(strat.name)
            if not fn:
                continue

            result = fn(candles, strat.params)
            alloc  = allocations.get(strat.name, 0.0)

            if result["signal"] == "buy":
                buy_weight  += alloc
            elif result["signal"] == "sell":
                sell_weight += alloc

            if result.get("on_fire"):
                on_fire = True
            mult = result.get("position_size_mult", 1.0)
            if mult > max_pos_mult:
                max_pos_mult = mult

            breakdown.append({
                "strategy": strat.name,
                "signal":   result["signal"],
                "alloc":    round(alloc * 100, 1),
                "score":    round(strat.score, 2),
                "win_rate": round(strat.win_rate * 100, 1),
                "sharpe":   round(strat.sharpe, 3),
                "on_fire":  result.get("on_fire", False),
                "meta":     result.get("meta", {}),
            })

        # ── ML GATE ──────────────────────────────────────────────────────────
        # XGBoost acts as a filter: heuristic signals only pass through
        # when ML confirms the directional thesis.
        # sell signals are allowed regardless of ML (defensive exits).
        if buy_weight > ENSEMBLE_THRESHOLD:
            if ml_prob >= ML_SIGNAL_THRESHOLD:
                signal = "buy"
            else:
                signal = "none"   # heuristics agree but ML disagrees → hold
        elif sell_weight > ENSEMBLE_THRESHOLD:
            signal = "sell"
        else:
            signal = "none"

        return {
            "signal":             signal,
            "symbol":             symbol,
            "price":              candles[-1]["close"],
            "regime":             regime,
            "buy_weight":         round(buy_weight,  3),
            "sell_weight":        round(sell_weight, 3),
            "ml_prob":            round(ml_prob,     3),
            "on_fire":            on_fire,
            "position_size_mult": max_pos_mult if signal != "none" else 1.0,
            "breakdown":          breakdown,
            "time":               _now(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # REINFORCEMENT LEARNING FEEDBACK
    # ─────────────────────────────────────────────────────────────────────────

    def reward(self, strategy_name: str, pnl: float, regime: str,
               state: np.ndarray | None, action: float,
               next_state: np.ndarray | None, trade_value: float = 1.0) -> None:
        """
        Called after every closed trade.
        Updates strategy scores, regime weights, replay buffer.
        Logs RL experience for offline SAC training.
        """
        strat = self._get_strategy(strategy_name)
        if strat:
            strat.record_trade(pnl, trade_value)

        self.total_trades += 1

        # Regime weight EMA update
        rw      = self.regime_weights.setdefault(regime, {})
        current = rw.get(strategy_name, 1.0)
        delta   = 1.0 if pnl > 0 else -0.5
        rw[strategy_name] = max(0.01, min(5.0, current * 0.9 + delta * 0.1))

        # Replay buffer
        self.replay_buffer.append({
            "strategy": strategy_name, "pnl": pnl,
            "trade_value": trade_value, "regime": regime,
            "time": _now(), "params": copy.deepcopy(strat.params) if strat else {},
        })

        # RL experience log (for offline SAC training)
        if state is not None and next_state is not None:
            # Differential Sharpe ratio reward
            rl_reward = _differential_sharpe_reward(pnl, trade_value)
            log_rl_experience(
                symbol="ALL", state=state.tolist(), action=action,
                reward=rl_reward, next_state=next_state.tolist(), done=False)

        # Trigger mutation if score crashes
        if strat and strat.score < KILL_THRESHOLD:
            self._mutate_strategy(strat, reason="low_score")

        # Trial evaluation for generated strategies
        if strat and (getattr(strat, "is_generated", False)
                      or getattr(strat, "is_yolo", False)):
            trial_len = (YOLO_TRIAL_TRADES if getattr(strat, "is_yolo", False)
                         else TRIAL_TRADES)
            self._evaluate_trial(strat, trial_len)

        if self.total_trades % REPLAY_EVERY == 0:
            self._replay_review()

        if self.total_trades % GENERATE_EVERY == 0:
            self._try_generate()

        self.save()

    # ─────────────────────────────────────────────────────────────────────────
    # GENETIC MUTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _mutate_strategy(self, strat: StrategyParams, reason: str = "scheduled") -> None:
        is_yolo   = getattr(strat, "is_yolo", False)
        mut_prob  = YOLO_MUTATION_PROB  if is_yolo else MUTATION_PROB
        mut_scale = YOLO_MUTATION_SCALE if is_yolo else MUTATION_SCALE

        old_params = copy.deepcopy(strat.params)
        new_params: dict = {}
        for key, val in strat.params.items():
            if random.random() < mut_prob:
                if isinstance(val, float):
                    noise = val * mut_scale * random.uniform(-1, 1)
                    new_params[key] = round(val + noise, 4)
                elif isinstance(val, int):
                    delta = max(1, int(val * mut_scale))
                    new_params[key] = max(2, val + random.randint(-delta, delta))
                else:
                    new_params[key] = val
            else:
                new_params[key] = val

        strat.params     = new_params
        strat.score      = 0.0
        strat.generation += 1

        self.mutation_log.append({
            "time": _now(), "strategy": strat.name, "reason": reason,
            "generation": strat.generation,
            "old_params": old_params, "new_params": new_params,
            "yolo": is_yolo,
        })
        emoji = "🔥" if is_yolo else "🧬"
        log.info("%s MUTATION [%s] gen %d — %s",
                 emoji, strat.name, strat.generation, reason)

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY GENERATION & LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def _try_generate(self) -> None:
        if len(self.strategies) >= MAX_STRATEGIES:
            gen_strats = [s for s in self.strategies
                          if getattr(s, "is_generated", False)
                          and s.total_trades >= TRIAL_TRADES]
            if not gen_strats:
                return
            worst = min(gen_strats, key=lambda s: s.score + s.sharpe)
            self._retire(worst, reason="replaced_by_new")

        existing_names = {s.name for s in self.strategies}
        sp, fn         = generate_random_strategy(existing_names)

        self.strategies.append(sp)
        self._generated_fns[sp.name] = fn
        for r in REGIMES:
            self.regime_weights.setdefault(r, {})[sp.name] = 1.0

        bp = sp.blueprint
        self.generation_log.append({
            "time": _now(), "name": sp.name,
            "entry": bp.get("entry", "?"), "filter": bp.get("filter", "?"),
            "logic": bp.get("logic", "?"), "status": "trial",
        })
        log.info("🌱 NEW STRATEGY: %s  Entry=%s Filter=%s Logic=%s",
                 sp.name, bp.get("entry"), bp.get("filter"), bp.get("logic"))

    def _evaluate_trial(self, strat: StrategyParams, trial_len: int) -> None:
        if strat.total_trades < trial_len:
            return

        is_core = not getattr(strat, "is_generated", False) and not getattr(strat, "is_yolo", False)
        passed  = strat.win_rate >= TRIAL_MIN_WIN_RATE and strat.total_pnl >= TRIAL_MIN_PNL

        if passed:
            verdict = "graduated"
            log.info("✅ GRADUATED: %s  wr=%.0f%%  pnl=%.2f  sharpe=%.2f",
                     strat.name, strat.win_rate * 100, strat.total_pnl, strat.sharpe)
        elif strat.win_rate < 0.30 or strat.total_pnl < TRIAL_MIN_PNL * 2:
            if not is_core:
                self._retire(strat, reason="failed_trial")
                return
            self._mutate_strategy(strat, reason="core_underperforming")
            verdict = "mutated"
        else:
            self._mutate_strategy(strat, reason="trial_improvement")
            verdict = "mutated_retry"

        for entry in reversed(self.generation_log):
            if entry["name"] == strat.name:
                entry["status"]   = verdict
                entry["win_rate"] = round(strat.win_rate * 100, 1)
                entry["pnl"]      = round(strat.total_pnl, 2)
                break

    def _retire(self, strat: StrategyParams, reason: str) -> None:
        self.graveyard.append({
            "time": _now(), "name": strat.name, "reason": reason,
            "win_rate": round(strat.win_rate * 100, 1),
            "total_pnl": round(strat.total_pnl, 2),
            "sharpe": round(strat.sharpe, 3),
            "generation": strat.generation,
            "blueprint": getattr(strat, "blueprint", {}),
        })
        self.strategies = [s for s in self.strategies if s.name != strat.name]
        self._generated_fns.pop(strat.name, None)
        for rw in self.regime_weights.values():
            rw.pop(strat.name, None)
        log.info("💀 RETIRED: %s  reason=%s", strat.name, reason)

    # ─────────────────────────────────────────────────────────────────────────
    # REPLAY REVIEW
    # ─────────────────────────────────────────────────────────────────────────

    def _replay_review(self) -> None:
        if len(self.replay_buffer) < 20:
            return
        recent    = self.replay_buffer[-100:]
        combo_pnl: dict[str, list] = {}
        for entry in recent:
            key = f"{entry['strategy']}|{entry['regime']}"
            combo_pnl.setdefault(key, []).append(entry["pnl"])

        best_combo, best_avg = None, -999.0
        for key, pnls in combo_pnl.items():
            sn, regime = key.split("|")
            avg = sum(pnls) / len(pnls)
            rw  = self.regime_weights.setdefault(regime, {})
            boost = 0.15 if avg > 0 else -0.08
            rw[sn] = max(0.01, min(5.0, rw.get(sn, 1.0) + boost))
            if avg > best_avg:
                best_avg, best_combo = avg, key

        log.info("📼 REPLAY — %d trades | best combo: %s avg=$%.2f",
                 len(recent), best_combo, best_avg)

        for regime in REGIMES:
            rw    = self.regime_weights.get(regime, {})
            worst = min(rw, key=lambda k: rw[k], default=None)
            if worst and rw[worst] < 0.3:
                s = self._get_strategy(worst)
                if s:
                    self._mutate_strategy(s, reason=f"replay_{regime}")

    # ─────────────────────────────────────────────────────────────────────────
    # ANTI-CORRELATION GUARD
    # ─────────────────────────────────────────────────────────────────────────

    def are_correlated(self, closes_a: list[float], closes_b: list[float],
                       window: int = 20) -> bool:
        if len(closes_a) < window + 1 or len(closes_b) < window + 1:
            return False
        ra = [(closes_a[i] - closes_a[i-1]) / (closes_a[i-1] + 1e-9)
              for i in range(-window, 0)]
        rb = [(closes_b[i] - closes_b[i-1]) / (closes_b[i-1] + 1e-9)
              for i in range(-window, 0)]
        n  = len(ra)
        ma, mb = sum(ra)/n, sum(rb)/n
        cov = sum((a-ma)*(b-mb) for a, b in zip(ra, rb)) / n
        sa  = math.sqrt(sum((a-ma)**2 for a in ra) / n + 1e-12)
        sb  = math.sqrt(sum((b-mb)**2 for b in rb) / n + 1e-12)
        return (cov / (sa * sb)) > 0.85

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_strategy(self, name: str) -> Optional[StrategyParams]:
        for s in self.strategies:
            if s.name == name:
                return s
        return None

    def get_summary(self) -> dict:
        return {
            "total_trades":   self.total_trades,
            "current_regime": self.current_regime,
            "circuit_open":   self.circuit_open,
            "peak_equity":    round(self.peak_equity, 2),
            "mutations":      len(self.mutation_log),
            "strategies": [{
                "name":         s.name,
                "score":        round(s.score, 2),
                "win_rate":     round(s.win_rate * 100, 1),
                "sharpe":       round(s.sharpe, 3),
                "total_trades": s.total_trades,
                "total_pnl":    round(s.total_pnl, 2),
                "generation":   s.generation,
                "params":       s.params,
                "is_generated": getattr(s, "is_generated", False),
                "is_yolo":      getattr(s, "is_yolo", False),
                "blueprint":    getattr(s, "blueprint", {}),
                "max_drawdown": round(s.max_drawdown, 2),
                "on_trial":     (
                    (getattr(s, "is_generated", False) or getattr(s, "is_yolo", False))
                    and s.total_trades < (YOLO_TRIAL_TRADES
                                          if getattr(s, "is_yolo", False) else TRIAL_TRADES)
                ),
                "trial_progress": (
                    round(s.total_trades / (YOLO_TRIAL_TRADES
                          if getattr(s, "is_yolo", False) else TRIAL_TRADES) * 100)
                    if (getattr(s, "is_generated", False) or getattr(s, "is_yolo", False))
                    else None
                ),
            } for s in self.strategies],
            "regime_history":  self.regime_history[-50:],
            "graveyard":       self.graveyard[-10:],
            "generation_log":  self.generation_log[-10:],
            "last_mutations":  self.mutation_log[-3:],
            "cb_log":          self.cb_log[-5:],
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _differential_sharpe_reward(pnl: float, trade_value: float,
                                 alpha: float = 0.01) -> float:
    """
    Differential Sharpe ratio approximation for RL reward.
    Penalises losses more than it rewards equivalent gains (risk-averse).
    """
    ret = pnl / max(abs(trade_value), 1.0)
    # Simple differential Sharpe: penalise losses asymmetrically
    if ret < 0:
        return ret * 2.0   # losses count double
    return ret
