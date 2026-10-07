"""
bot.py — Async WebSocket Trading Orchestrator.

SHORTING ADDITIONS:
  - "short" and "cover" actions now recognized and executed safely.
  - _tick_exit_check() correctly triggers inverse checks for shorts (highs hit stops, lows hit TPs).
  - Equity curves perfectly calculate total value using long + short margins/PnL combined.

FIXES (v3.1):
  Fix #1  — _tick_exit_check now correctly handles short positions (was bailing on shares==0).
  Fix #2  — _tick_exit_check acquires _cash_lock before _execute_trade (race condition).
  Fix #3  — websockets import moved to top-level (was re-importing on every reconnect).
  Fix #4  — import time moved to top-level; _now name collision in _evaluate_and_trade removed.
  Fix #5  — exit_monitor acquires _cash_lock before _execute_trade (race condition).
  Fix #6  — Equity calculation extracted into single _compute_total_equity() helper.
  Fix #7  — total_equity passed into _execute_trade as a parameter; no more double DB fetch.
  Fix #8  — SAC fallback sentinel changed to None; 0.5 no longer mis-treated as a valid signal.
  Fix #9  — ATR calculation upgraded to true Wilder's smoothed ATR (was a simple mean).
  Fix #10 — _exit_in_flight race closed; _trade_lock used as the atomicity guard.
  Fix #11 — Removed unused `import math`.
  Fix #12 — Removed unused STARTING_CASH and ENABLE_SHORTING imports.
  Fix #13 — Import block reorganised: stdlib -> third-party -> local -> module constants.
  Fix #14 — Removed unused get_short_position import from db.
  Fix #15 — _now() and helpers moved to top of file, above all functions that call them.

FIXES (v3.2) — Post-telemetry autopsy hotfixes:
  Fix #16 — _calc_trade_value now distinguishes three SAC states:
              None       -> genuine inference crash  -> warn + 2% fallback
              <= 0.001   -> deliberate AI VETO       -> info log + return 0.0 to abort trade
              > 0.001    -> live signal               -> size normally
            Previously, sac_fraction=0.0 was treated identically to a crash and forced a 2%
            fallback trade, overriding 181 explicit SAC vetoes and executing 247 unwanted trades.
  Fix #17 — _run_incremental_train now applies a class-balance gate before submitting training
            data. Batches with fewer than 30 samples or >85% single-class dominance are skipped,
            preventing the XGBoost learner.cc:782 zero-variance hallucination that caused the
            model to output 1.00 confidence 4,705 times during a one-directional market dump.
  Fix #18 — brain.reward() calls in sell/cover paths now pass a shaped reward that subtracts
            a per-candle opportunity cost and a round-trip fee estimate from raw PnL. This gives
            the SAC agent non-zero reward variance on scratch trades (previously $0.00 PnL ->
            flat 0.0 reward -> reward starvation -> no learning gradient).

FIXES (v4.1) — Audit-driven hardening (C1-C3):
  Fix #19 — exit_monitor's per-position _execute_trade call is now wrapped in try/except.
            Previously an unhandled exception there propagated straight through
            asyncio.gather in main() and crashed the entire 24/7 process — one bad tick
            on one symbol took down risk monitoring for all of them.
  Fix #20 — main()'s asyncio.gather now passes return_exceptions=True as a second line of
            defense, with any captured exception logged CRITICAL so a dead subsystem is
            loud, never silently swallowed.
  Fix #21 — Brain._peak_equity (the circuit breaker's drawdown high-water mark) is now
            persisted to brain_state on every new peak and restored at startup. It was
            previously RAM-only, so every restart (crash, deploy, OOM) silently re-armed
            MAX_DRAWDOWN_PCT from whatever equity existed at boot, discarding all prior
            drawdown memory.
  Fix #22 — open_short() now rejects re-shorting a symbol that already has an active short
            (raises ValueError) instead of silently overwriting the row via INSERT OR
            REPLACE, which used to orphan the first short's margin_reserved/stop/opened_ts
            tracking and permanently leak tracked cash. Guarded at both the entry-gating
            pre-check (_evaluate_and_trade) and the executor (defense in depth).
"""

from __future__ import annotations
import math
from dataclasses import dataclass

from config import (
    BINANCE_REST, BINANCE_WS_BASE, CANDLE_LIMIT, CHECK_EVERY_SECS,
    CORRELATION_THRESHOLD, ENABLE_HARD_MAX_HOLD_EXIT,
    GLOBAL_POSITION_NOTIONAL_CAP,
    HARD_STOP_LOSS_USD, MAX_HOLD_OPEN_SECONDS,
    INTERVAL, LOG_DIR, MAX_CORRELATED_OPEN_PEERS, MAX_HOLD_CANDLES,
    MAX_OPEN_POSITIONS, MIN_ML_CONFIDENCE, ML_UPDATE_EVERY,
    PROCESS_POOL_WORKERS,
    SLIPPAGE_PCT, SYMBOLS,
    SHORT_MARGIN_PCT, SHORT_MAX_OPEN, SYMBOL_COOLDOWN_SECS,
    DECAY_HALFLIFE_CANDLES, DECAY_MIN_CANDLES,
    DECAY_SL_TIGHTEN_STRENGTH, DECAY_TP_PULL_STRENGTH, EXPERIMENT_NAME,
    BE_TRIGGER_ATR_MULT, BE_MIN_PROFIT_ATR, BE_MIN_PROFIT_PCT,
    FEE_GATE_ROUND_TRIP, MICRO_WIN_REWARD_PENALTY, MICRO_WIN_USD,
    MIN_EXPECTED_TP_NET_USD,
    MAX_ORDER_EQUITY_FRAC, STALE_ENTRY_MAX_SECS, WS_BACKFILL_AFTER_SECS,
    STARTING_CASH,
)
from execution import OrderIntent, get_execution_adapter, make_client_order_id
from rl_agent import compute_position_fraction
from ml_engine import predict_signal, incremental_train, initial_train
from brain import Brain
# Fix O1: was a standalone implementation that had already drifted from
# brain.py's copy once (see Fix #9 below) before being manually re-aligned.
# Now the single shared implementation in indicators.py, imported under the
# original local name so every call site below is unchanged.
from indicators import wilder_atr_from_candles as _wilder_atr
from accounting_v2 import (
    entry_exit_fees_notional,
    net_realized_pnl,
    position_equity_components,
    shaped_reward_net,
)
from binance_margin import evaluate_margin_health, fetch_margin_account_snapshot
from risk_engine import (
    apply_dynamic_rr,
    correlation_blocks_entry,
    dynamic_conviction_size_mult,
    exposure_blocked,
)
from db import (
    close_position, delete_position, get_all_positions, get_candles, get_cash,
    get_entry_state, get_portfolio_stat, increment_candle_count,
    init_db, load_brain_key, log_trade, log_rl_experience,
    open_position, open_position_count,
    record_equity, save_brain_key, save_ml_cache, set_cash, set_portfolio_stat,
    upsert_candle, upsert_candles_bulk, open_short, close_short,
    get_short_position, get_all_short_positions, open_short_count, update_stop_price, update_tp_price, update_mfe_mae,
    get_open_orders, reduce_position, try_create_order, update_order,
)

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import numpy as np
import websockets
import requests

# Discord webhook for trade alerts. Sourced from the environment so the
# credential never lives in the repo (systemd: EnvironmentFile=/etc/quant-bot.env).
# Empty/unset disables alerts entirely.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def alert_sniper_shot(symbol, action, price, strategy):
    """Sends a formatted trade execution alert to Discord."""
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "":
        return

    # Format the alert with clear visual indicators
    direction_emoji = "🔴" if action.lower() == "short" else "🟢"

    payload = {
        "content": None,
        "embeds": [
            {
                "title": f"{direction_emoji} LIVE EXECUTION: {symbol}",
                "color": 16711680 if action.lower() == "short" else 65280,
                "fields": [
                    {"name": "Action", "value": action.upper(), "inline": True},
                    {"name": "Fill Price", "value": f"${price}", "inline": True},
                    {"name": "Strategy", "value": strategy, "inline": True}
                ],
                "footer": {"text": "QuantBot Engine • Risk Manager Cleared"}
            }
        ]
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")


# ── AI-driven sizing constants (replaces static TRADE_SIZE_PCT / MAX_TRADE_SIZE_PCT) ──
_SAC_SIZE_FLOOR = 0.01    # 1%   - minimum equity fraction the agent can deploy
# 35%  - hard ceiling per trade (prevents margin errors)
_SAC_SIZE_CEILING = 0.35
# 2%   - conservative fallback when sac_fraction is None (crash)
_SAC_FALLBACK_PCT = 0.02
# Fix #16: sac_fraction at or below this is a deliberate AI veto
_SAC_VETO_THRESH = -0.1501


def _authoritative_conviction_mult(ensemble: dict, ml_prob_fallback: float) -> float:
    """
    Re-bind brain.ml_conviction_size_mult at execution time so sizing cannot drift from
    whatever the ensemble dict carried (dashboards / future serializers may omit fields).
    """
    mp = float(ensemble.get("ml_prob", ml_prob_fallback))
    meta = ensemble.get("meta") or {}
    ss = float(meta.get("short_score", ensemble.get("sell_weight", 0.0)))
    return float(brain.ml_conviction_size_mult(mp, ss))


def _entry_proposed_equity_frac(sac: float, confidence_mult: float) -> float:
    """
    Equity fraction for new legs and the exposure pre-check.

    A single global floor on (sac × cm) wipes out brain's ml_conviction_size_mult when
    SAC is cautious — every symbol then lands on the same trade_pct. Scaling the
    floor by cm preserves relative sizing without changing ml_conviction_size_mult.
    """
    cm = max(0.0, float(confidence_mult))
    sf = max(0.0, float(sac))
    scaled_floor = _SAC_SIZE_FLOOR * cm
    raw = sf * cm
    return min(_SAC_SIZE_CEILING, max(scaled_floor, raw))


# ── SAC reward-shaping (V2 uses accounting_v2.shaped_reward_net) ─────────────
_REWARD_CANDLE_COST = 0.00005  # per-candle opportunity cost (tune in research)

# ── Logging ───────────────────────────────────────────────────────────────────
_log_file = LOG_DIR / "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            str(_log_file), maxBytes=5_000_000, backupCount=2),
    ],
)
log = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────────
brain = Brain()
_EDGE_PROFILE_DB_KEY = "adaptive_edge_profiles_v1"

# V4: execution backend (paper by default; testnet via EXECUTION_MODE).
# Selected once at import so a misconfigured mode fails at boot, not mid-trade.
_exec_adapter = get_execution_adapter()

# V4: WS-outage tracking for the reconnect backfill.
_ws_down_since: Optional[float] = None

_candles_cache:        dict[str, list[dict]] = {}
_closed_candle_counts: dict[str, int] = {s: 0 for s in SYMBOLS}
_executor: Optional[ProcessPoolExecutor] = None

_shutdown_event = asyncio.Event()
_trade_locks:       dict[str, asyncio.Lock] = {}
_cash_lock = asyncio.Lock()         # serialise cross-symbol cash read-modify-write
# One portfolio commit at a time: risk reads + SQLite fills cannot race (WS burst / ticks / monitor).
_strict_execution_lock = asyncio.Lock()
_symbol_last_trade: dict[str, float] = {}   # cooldown tracker
_exit_in_flight:    set[str] = set()

# V2: last margin / utilization snapshot for gating + dashboard
_margin_health_cache: dict = {}


def _persist_adaptive_edge_profiles() -> None:
    """
    Serialize brain._edge_profiles to JSON and write to brain_state table.

    Fix 2 — root-cause of the Amnesia bug:
      • Previously called set_portfolio_stat() which targets the `portfolio`
        table.  load_brain_key() reads from `brain_state`.  The two functions
        were writing to and reading from DIFFERENT tables, so brain_state was
        always empty and the edge profiles were invisibly lost on every restart.
      • Now explicitly calls save_brain_key() → brain_state table so persist and
        restore are symmetric.
      • Explicit json.dumps() with separators so the serialised string is compact
        and unambiguous even if export_edge_profiles() returns a nested dict.
      • Wrapped in try/except with a log.error so any serialisation or DB failure
        is immediately visible in the log instead of silently eating the write.
    """
    if not hasattr(brain, "export_edge_profiles"):
        return
    try:
        payload = brain.export_edge_profiles()          # dict[str, dict[str, float]]
        blob    = json.dumps(payload, separators=(",", ":"))  # compact JSON string
        save_brain_key(_EDGE_PROFILE_DB_KEY, json.loads(blob))  # store via brain_state
        log.debug(
            "Edge profiles persisted — %d buckets → brain_state[%s]",
            len(payload), _EDGE_PROFILE_DB_KEY,
        )
    except Exception as _persist_exc:
        log.error(
            "❌ _persist_adaptive_edge_profiles FAILED — memory NOT saved: %s",
            _persist_exc,
        )


def _restore_adaptive_edge_profiles() -> int:
    """
    Load edge profiles from brain_state at startup.

    Fix 3 — startup recovery fallback:
      • Reads from brain_state via load_brain_key() (mirrors the new persist path).
      • Returns 0 cleanly if the key is absent (fresh install or wiped DB) so
        the bot starts with default edge profiles without crashing.
      • Logs whether profiles were recovered or a clean slate was initialised.
    """
    if not hasattr(brain, "import_edge_profiles"):
        return 0
    try:
        payload = load_brain_key(_EDGE_PROFILE_DB_KEY, None)
    except Exception as _load_exc:
        log.warning("Edge profile DB read failed (%s) — starting with clean slate.", _load_exc)
        return 0

    if payload is None:
        log.info("No edge profiles found in brain_state — starting with clean slate.")
        return 0

    if not isinstance(payload, dict):
        log.warning(
            "brain_state[%s] is not a dict (got %s) — starting with clean slate.",
            _EDGE_PROFILE_DB_KEY, type(payload).__name__,
        )
        return 0

    try:
        count = int(brain.import_edge_profiles(payload))
    except Exception as _import_exc:
        log.warning("import_edge_profiles failed (%s) — starting with clean slate.", _import_exc)
        return 0

    log.info("Adaptive edge profiles restored: %d buckets from brain_state.", count)
    return count


def _bar_hours() -> float:
    """Convert INTERVAL string (e.g. '1m') to bar duration in hours."""
    s = str(INTERVAL).strip().lower()
    try:
        if s.endswith("m"):
            return int(s[:-1]) / 60.0
        if s.endswith("h"):
            return float(s[:-1])
    except ValueError:
        pass
    return 1.0 / 60.0


def _hold_hours_from_candles(n_candles: int) -> float:
    return max(0.0, float(n_candles) * _bar_hours())


async def _refresh_margin_health(
    session: aiohttp.ClientSession, cash: float, total_equity: float,
) -> dict:
    """Poll Binance margin account when keys exist; always merge synthetic utilization."""
    global _margin_health_cache
    live = await fetch_margin_account_snapshot(session)
    _margin_health_cache = evaluate_margin_health(
        live_json=live, cash=cash, total_equity=total_equity,
    )
    set_portfolio_stat("margin_health_json", json.dumps(_margin_health_cache))
    return _margin_health_cache


def _experiment_arm(symbol: str) -> Optional[str]:
    """Arm ("A"/"B") of the active experiment for a symbol, or None if off.

    Alternates in SYMBOLS order: balanced, deterministic, splits the majors.
    """
    if not EXPERIMENT_NAME:
        return None
    try:
        idx = SYMBOLS.index(symbol)
    except ValueError:
        idx = sum(symbol.encode())        # stable fallback for unknown symbols
    return "A" if idx % 2 == 0 else "B"


def _short_slots_full(symbol: str) -> bool:
    """SHORT_MAX_OPEN check; split evenly per arm while an experiment runs."""
    arm = _experiment_arm(symbol)
    if arm is None:
        return open_short_count() >= SHORT_MAX_OPEN
    per_arm = math.ceil(SHORT_MAX_OPEN / 2)
    same_arm = sum(1 for p in get_all_short_positions() if _experiment_arm(p["symbol"]) == arm)
    return same_arm >= per_arm


def _apply_designed_time_decay(sym: str, pos: dict, feats: dict) -> None:
    """
    Time decay as designed (experiment decay_v2, arm B): λ = 1 − e^(−t/τ) with
    t = wall-clock minutes since entry (INTERVAL is 1m, so τ in candles == τ
    in minutes), applied to the INITIAL stop/target recorded at entry. Being a
    pure function of age it is idempotent however often it is called; it only
    ever tightens, never loosens a level the trail / break-even already moved.
    """
    try:
        entry = float(pos.get("avg_cost") or 0.0)
        tp0 = float(feats["initial_tp"])
        sl0 = float(feats["initial_stop"])
        opened = datetime.fromisoformat(str(pos.get("opened_ts")).replace("Z", "+00:00"))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return
    age_min = (time.time() - opened.timestamp()) / 60.0
    if entry <= 0 or tp0 <= 0 or sl0 <= 0 or age_min < DECAY_MIN_CANDLES:
        return
    lam = 1.0 - math.exp(-age_min / max(DECAY_HALFLIFE_CANDLES, 1e-6))
    if lam < 0.03:
        return
    stop = float(pos.get("stop_price", 0.0) or 0.0)
    tp = float(pos.get("tp_price", 0.0) or 0.0)
    target_tp = entry + (tp0 - entry) * (1.0 - lam * DECAY_TP_PULL_STRENGTH)
    target_sl = sl0 + lam * DECAY_SL_TIGHTEN_STRENGTH * (entry - sl0)
    if pos.get("side", "long") == "short":
        if tp < target_tp < entry - 1e-9:
            update_tp_price(sym, target_tp)
        if entry + 1e-9 < target_sl < stop:
            update_stop_price(sym, target_sl)
    else:
        if entry + 1e-9 < target_tp < tp:
            update_tp_price(sym, target_tp)
        if stop < target_sl < entry - 1e-9:
            update_stop_price(sym, target_sl)


def _apply_ml_time_decay(sym: str, pos: dict, candles: list[dict]) -> None:
    """
    Exponential time-decay on stagnant risk targets: λ = 1 − e^(−t/τ).

    Pulls TP toward entry (sooner monetisation) and tightens stop — variance
    collapses when edge does not materialise (optional stopping / real-options view).

    Positions entered in arm B of experiment decay_v2 use the as-designed
    implementation; everything else (arm A, pre-experiment positions) keeps
    the legacy path below, which compounds on every call — see config.py.
    """
    try:
        feats = json.loads(pos.get("entry_features") or "{}")
    except (TypeError, ValueError):
        feats = {}
    if (feats.get("experiment") == "decay_v2" and feats.get("arm") == "B"
            and feats.get("initial_tp") and feats.get("initial_stop")):
        _apply_designed_time_decay(sym, pos, feats)
        return

    count = int(pos.get("candle_count", 0))
    # Do not apply any decay until the trade has had DECAY_MIN_CANDLES bars to
    # develop.  Without this guard, decay starts within ~8 candles (λ≥0.03 at
    # halflife=240) and makes the agent impatient on entries that simply haven't
    # had time to reach their target yet.
    if count < DECAY_MIN_CANDLES:
        return
    lam = 1.0 - math.exp(-count / max(DECAY_HALFLIFE_CANDLES, 1e-6))
    if lam < 0.03:
        return
    is_short = pos.get("side", "long") == "short"
    entry = float(pos.get("avg_cost", 0.0))
    stop = float(pos.get("stop_price", 0.0) or 0.0)
    tp = float(pos.get("tp_price", 0.0) or 0.0)
    if entry <= 0 or stop <= 0 or tp <= 0:
        return

    if not is_short:
        new_tp = entry + (tp - entry) * (1.0 - lam * DECAY_TP_PULL_STRENGTH)
        new_sl = stop + lam * DECAY_SL_TIGHTEN_STRENGTH * (entry - stop)
        if new_tp < tp - 1e-12 and new_tp > entry + 1e-9:
            update_tp_price(sym, new_tp)
        if new_sl > stop + 1e-12 and new_sl < entry - 1e-9:
            update_stop_price(sym, new_sl)
    else:
        new_tp = entry - (entry - tp) * (1.0 - lam * DECAY_TP_PULL_STRENGTH)
        new_sl = stop - lam * DECAY_SL_TIGHTEN_STRENGTH * (stop - entry)
        if new_tp > tp + 1e-12 and new_tp < entry - 1e-9:
            update_tp_price(sym, new_tp)
        if new_sl < stop - 1e-12 and new_sl > entry + 1e-9:
            update_stop_price(sym, new_sl)


# ── Pure helpers (Fix #15: defined before first use) ─────────────────────────

def _now() -> str:
    """UTC ISO-8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _last_candle_ts(candles: list[dict]) -> Optional[int]:
    """Open-time (ms) of the newest candle, or None if unavailable."""
    if not candles:
        return None
    try:
        return int(candles[-1].get("time"))
    except (TypeError, ValueError):
        return None


def _entry_data_stale_reason(candles: list[dict]) -> Optional[str]:
    """
    V4 stale-data gate — NEW ENTRIES only, exits are never gated.

    Returns a block reason when the newest candle's open time is older than
    STALE_ENTRY_MAX_SECS (default 180s = the live bar plus two closed 1m bars),
    or when no timestamp is available at all. A dead feed must not be allowed
    to open positions at prices that no longer exist.
    """
    ts = _last_candle_ts(candles)
    if ts is None:
        return "stale_data (no candle timestamp)"
    age = time.time() - ts / 1000.0
    if age > STALE_ENTRY_MAX_SECS:
        return f"stale_data (last candle {age:.0f}s old > {STALE_ENTRY_MAX_SECS:.0f}s)"
    return None


def _trade_lock(symbol: str) -> asyncio.Lock:
    if symbol not in _trade_locks:
        _trade_locks[symbol] = asyncio.Lock()
    return _trade_locks[symbol]


def _compute_total_equity(cash: float, open_positions: list[dict]) -> float:
    """
    Strict absolute accounting:
      equity = cash + sum(position_value + unrealised_pnl) over open positions.

    Position decomposition is centralized in accounting_v2.position_equity_components.
    """
    total_eq = float(cash)
    for p in open_positions:
        c_price = _candles_cache.get(
            p["symbol"], [{"close": p["avg_cost"]}])[-1]["close"]
        _, _, contrib = position_equity_components(
            side=p.get("side", "long"),
            avg_cost=p.get("avg_cost", 0.0),
            quantity=p.get("shares", 0.0),
            mark_price=c_price,
            margin_reserved=p.get("margin_reserved", 0.0),
        )
        total_eq += contrib
    return total_eq


def _top_strategy(ensemble: dict, direction: str) -> str:
    """
    Map the execution action back to the strategy's internal vote vocabulary
    so brain.reward() is attributed to the real strategy, not "ENSEMBLE".

    buy/cover executions  <-- strategies that voted "buy"
    sell/short executions <-- strategies that voted "sell"
    """
    target_signal = "buy" if direction in ("buy", "cover") else "sell"
    bd = ensemble.get("breakdown", [])
    top = max(
        (b for b in bd if b["signal"] == target_signal),
        key=lambda x: x["alloc"],
        default=None,
    )
    # No strategy voted -> pure ML Gate A override -> label XGBOOST
    return top["strategy"] if top else None


def _handle_signal(signame: str) -> None:
    log.info("Received %s - initiating graceful shutdown", signame)
    _shutdown_event.set()


def _shaped_reward_v2(gross_pnl: float, hold_candles: int, trade_value: float) -> float:
    """V2: SAC learns on *net* PnL after fees + time opportunity cost."""
    hh = _hold_hours_from_candles(hold_candles)
    return shaped_reward_net(
        gross_pnl, hold_candles, trade_value, hh, candle_cost=_REWARD_CANDLE_COST,
    )


def _is_trainable(
    labels:          list[int],
    min_samples:     int = 30,
    max_imbalance:   float = 0.98,
) -> bool:
    """
    Fix #17: class-balance gate for XGBoost incremental training.

    Returns False (skip update) when:
      - fewer than `min_samples` examples are present, OR
      - one class makes up more than `max_imbalance` of the batch.

    A stale model is far safer than one overfit to a one-directional market dump.
    The learner.cc:782 zero-variance warning and 100% confidence hallucination
    are both caused by feeding a degenerate single-class batch to the trainer.
    """
    if len(labels) < min_samples:
        log.warning(
            "XGB training skipped — only %d samples (min=%d)", len(
                labels), min_samples
        )
        return False
    pos_rate = sum(labels) / len(labels)
    if pos_rate > max_imbalance or pos_rate < (1.0 - max_imbalance):
        log.warning(
            "XGB training skipped — class imbalance %.1f%% (max=%.0f%%)",
            pos_rate * 100, max_imbalance * 100,
        )
        return False
    return True


# ── WebSocket / HTTP ──────────────────────────────────────────────────────────

async def fetch_candle_history(session: aiohttp.ClientSession, symbol: str) -> list[dict]:
    url = f"{BINANCE_REST}/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": CANDLE_LIMIT}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
            r.raise_for_status()
            raw = await r.json()
        candles = [
            {"time": c[0], "open": float(c[1]), "high":  float(c[2]),
             "low":  float(c[3]), "close": float(c[4]), "volume": float(c[5])}
            for c in raw
        ]
        log.info("Fetched %d historical candles for %s", len(candles), symbol)
        return candles
    except Exception as exc:
        log.error("History fetch failed for %s: %s", symbol, exc)
        return []


def _ws_url() -> str:
    streams = "/".join(f"{s.lower()}@kline_{INTERVAL}" for s in SYMBOLS)
    return f"{BINANCE_WS_BASE}?streams={streams}"


async def websocket_listener(loop: asyncio.AbstractEventLoop) -> None:
    # Fix #3: websockets imported at module level, not inside the reconnect loop.
    global _ws_down_since
    url = _ws_url()
    backoff = 1.0

    while not _shutdown_event.is_set():
        log.info("Connecting WebSocket -> %s", url)
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1.0
                # V4: after a long outage the in-memory candle cache has a gap
                # that would poison every indicator — REST-backfill first.
                # (New entries were already frozen by the stale-data gate.)
                if _ws_down_since is not None:
                    gap = time.time() - _ws_down_since
                    if gap > WS_BACKFILL_AFTER_SECS:
                        log.warning(
                            "WS was down %.0fs (> %.0fs) — backfilling candle "
                            "history before consuming the stream",
                            gap, WS_BACKFILL_AFTER_SECS,
                        )
                        try:
                            refreshed = await _load_history_for_all_symbols()
                            log.info("Backfill complete for %d symbols", len(refreshed))
                        except Exception as _bf_exc:
                            log.error("Backfill failed (%s) — stale gate still protects entries", _bf_exc)
                    _ws_down_since = None
                log.info("WebSocket connected")
                async for raw_msg in ws:
                    if _shutdown_event.is_set():
                        return
                    await _handle_ws_message(raw_msg, loop)

        except Exception as exc:
            if _shutdown_event.is_set():
                return
            if _ws_down_since is None:
                _ws_down_since = time.time()
            log.error("WebSocket error: %s - reconnecting in %.1fs", exc, backoff)
            try:
                await asyncio.wait_for(_shutdown_event.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 60.0)


# ── Fee-gate helpers (pure, no I/O, safe to call from async context) ─────────

def _classify_exit_reason(
    reason: Optional[str], avg_cost: float, stop: float, is_short: bool,
    initial_stop: Optional[float] = None,
) -> str:
    """
    Map the free-text exit trigger onto a canonical code for the trade journal
    (so exits can be GROUP BY'd instead of regex'd).

    Codes: TAKE_PROFIT, STOP_LOSS (initial protective stop, never moved),
    TIGHTENED_STOP (still on the loss side of entry but pulled in from the
    initial stop by time decay or the trail), BREAKEVEN_STOP (stop ratcheted
    to entry), TRAILING_STOP (stop ratcheted into profit), HARD_STOP (USD
    survival kill-switch), TIME_HOLD_EXIT, MAX_HOLD, CIRCUIT_BREAKER, OTHER.

    A stop-triggered exit is split by where the stop sat relative to entry —
    and to the initial stop, when known — at the moment it fired: the
    free-text reason alone cannot tell an initial stop from a moved one.
    Without initial_stop (positions opened before it was recorded) a moved
    loss-side stop is indistinguishable from the original: STOP_LOSS.
    """
    r = (reason or "").lower()
    if "circuit" in r:
        return "CIRCUIT_BREAKER"
    if "hard_stop" in r:
        return "HARD_STOP"
    if "time_hold" in r:
        return "TIME_HOLD_EXIT"
    if "max_hold" in r:
        return "MAX_HOLD"
    if "take_profit" in r:
        return "TAKE_PROFIT"
    if "stop_loss" in r:
        if avg_cost > 0 and stop > 0:
            if abs(stop - avg_cost) <= avg_cost * 1e-9:
                return "BREAKEVEN_STOP"
            in_profit = (stop < avg_cost) if is_short else (stop > avg_cost)
            if in_profit:
                return "TRAILING_STOP"
            if initial_stop and abs(stop - float(initial_stop)) > avg_cost * 1e-6:
                return "TIGHTENED_STOP"
        return "STOP_LOSS"
    return "OTHER"


def _entry_snapshot(
    ensemble: dict, action: str, regime: str, strategy_name: str,
    sac_fraction: Optional[float], total_equity: float,
) -> dict:
    """
    Signal-time decision snapshot for the trade journal, taken BEFORE any
    entry gate runs so skipped entries (ML gate, stale data, SAC veto, sizing,
    margin, duplicates) are journaled with the same features as fills.
    Sizing and stop/target levels are added by _prepare_trade as computed.

    Telemetry only: read-only (no edge-profile side effects) and never raises
    into the money path.
    """
    try:
        meta = ensemble.get("meta") or {}
        side = "short" if action == "short" else "long"
        raw_p = ensemble.get("ml_prob")
        ml_p = None if raw_p is None else float(raw_p)
        conviction = None if ml_p is None else ((1.0 - ml_p) if side == "short" else ml_p)
        ep = brain._edge_profiles.get(brain._edge_key(side, regime)) or {}
        snap: dict = {
            "ml_prob": None if ml_p is None else round(ml_p, 6),
            "ml_down": None if ml_p is None else round(1.0 - ml_p, 6),
            "ml_tier": None if conviction is None else int(conviction * 100) // 5 * 5,
            "regime": regime,
            "strategy": strategy_name,
            "sac_fraction": None if sac_fraction is None else round(float(sac_fraction), 6),
            "edge_size_mult": round(float(ep.get("size_mult", 1.0)), 4),
            "edge_rr_mult": round(float(ep.get("rr_mult", 1.0)), 4),
            "equity": round(float(total_equity), 2),
        }
        arm = _experiment_arm(str(ensemble.get("symbol", "")))
        if arm is not None:
            snap["experiment"] = EXPERIMENT_NAME
            snap["arm"] = arm
        for k in ("short_score", "adx", "pdi", "mdi", "rvol", "ret5", "ret20",
                  "ema_spread", "ml_component", "struct_component"):
            snap[k] = meta.get(k)
        return snap
    except Exception as exc:   # pragma: no cover - defensive
        log.debug("entry snapshot failed (%s) — journaling without features", exc)
        return {"snapshot_error": str(exc)}


def _initial_stop(pos: dict) -> Optional[float]:
    """Initial stop recorded in the position's entry_features, or None."""
    try:
        v = json.loads(pos.get("entry_features") or "{}").get("initial_stop")
        return float(v) if v else None
    except (ValueError, TypeError, AttributeError):
        return None


def _exit_journal_fields(pos: dict, is_short: bool, trade_rec: dict) -> dict:
    """
    Trade-journal columns for a closing leg, derived from the position row
    (opened_ts, entry_features, mfe/mae marks) plus the exit_reason/exit_detail
    that the exit path attached to trade_rec.

    peak_profit_pct is clamped >= 0 and max_drawdown_pct <= 0 (a loss is
    negative, same sign as min_unrealized_pnl). Excursions are candle-extreme
    based (update_mfe_mae), seeded at the entry price. Fields that cannot be
    determined are None, never a fabricated 0.
    """
    out: dict = {
        "exit_reason": trade_rec.get("exit_reason"),
        "exit_detail": trade_rec.get("exit_detail"),
        "entry_features": pos.get("entry_features"),
        "peak_profit_pct": None,
        "max_drawdown_pct": None,
        "hold_time_seconds": None,
    }
    avg = float(pos.get("avg_cost") or 0.0)
    mfe, mae = pos.get("mfe_price"), pos.get("mae_price")
    if avg > 0 and mfe is not None and mae is not None:
        sign = 1.0 if is_short else -1.0          # short profits when price falls
        out["peak_profit_pct"] = round(max(0.0, sign * (avg - float(mfe)) / avg * 100.0), 4)
        out["max_drawdown_pct"] = round(min(0.0, sign * (avg - float(mae)) / avg * 100.0), 4)
    opened = pos.get("opened_ts")
    if opened:
        try:
            _dt = datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
            if _dt.tzinfo is None:
                _dt = _dt.replace(tzinfo=timezone.utc)
            out["hold_time_seconds"] = max(0, int(time.time() - _dt.timestamp()))
        except (ValueError, TypeError):
            pass
    return out


def _profit_clears_fees(
    avg_cost: float, shares: float, exit_price: float, is_short: bool
) -> bool:
    """
    Returns True if closing at exit_price would BOOK a positive net PnL —
    projected exactly as _prepare_trade/_commit_trade will book it: exit fill
    slipped by SLIPPAGE_PCT against us, then net_realized_pnl()'s round-trip
    fee. (The old check compared raw profit to the fee alone and ignored the
    exit slippage, so ~half of all take-profit exits booked a net LOSS.)

    Gates take_profit exits only. Stop exits of every kind must never be
    routed through here — a stop is risk control, not a profit decision.

    Returns True (allow) on degenerate inputs so a missing field never blocks
    a legitimate exit.
    """
    if avg_cost <= 0 or shares <= 0 or exit_price <= 0:
        return True
    fill = exit_price * (1 + SLIPPAGE_PCT) if is_short else exit_price * (1 - SLIPPAGE_PCT)
    entry_notional = shares * avg_cost
    exit_notional = shares * fill
    gross = entry_notional - exit_notional if is_short else exit_notional - entry_notional
    _, net = net_realized_pnl(gross, entry_notional, exit_notional)
    return net > 0.0


def _hard_stop_open_pnl(pos: dict, mark_price: float) -> float:
    """
    Unrealised open PnL in USD (negative = loss).

    Used exclusively by the HARD_STOP_LOSS_USD survival kill-switch — intentionally
    simple so it cannot throw on bad data.  Always returns 0.0 on degenerate input.
    """
    avg  = float(pos.get("avg_cost", 0.0) or 0.0)
    qty  = float(pos.get("shares",   0.0) or 0.0)
    mark = float(mark_price or 0.0)
    if avg <= 0 or qty <= 0 or mark <= 0:
        return 0.0
    if pos.get("side", "long") == "short":
        return (avg - mark) * qty
    return (mark - avg) * qty


# ── Tick-level exit check ─────────────────────────────────────────────────────

async def _tick_exit_check(sym: str, candle: dict, loop: asyncio.AbstractEventLoop) -> None:
    # Fix #10: _trade_lock.locked() is the correct atomicity guard.
    # The old bare-set pattern had a TOCTOU window between the membership check
    # and the add; two concurrent tasks for the same symbol could both pass.
    if _trade_lock(sym).locked():
        return

    pos_list = get_all_positions()
    pos = next((p for p in pos_list if p["symbol"] == sym), None)
    if not pos:
        return

    is_short = pos.get("side", "long") == "short"

    # Fix #1: correct existence check per side.
    # Shorts carry their value in margin_reserved, not shares.
    if is_short and pos.get("margin_reserved", 0.0) <= 0:
        return
    if not is_short and pos.get("shares", 0) <= 0:
        return

    stop = pos.get("stop_price", 0)
    tp = pos.get("tp_price",   0)

    candle_high = candle["high"]
    candle_low = candle["low"]
    reason:     Optional[str] = None
    exit_price: float = 0.0

    # Fix 1: ratchet MFE/MAE on every 1-minute tick so the columns are
    # current when _execute_trade reads the position on exit.  The 30-second
    # exit_monitor also calls this, but a fast exit (e.g. TP on first bar)
    # would close before the monitor cycle runs without this guard.
    try:
        update_mfe_mae(sym, float(candle_high), float(candle_low), is_short)
    except Exception as _mfe_tick_exc:
        log.debug("MFE/MAE tick update skipped for %s: %s", sym, _mfe_tick_exc)

    # Fix #9: use Wilder's smoothed ATR from the shared helper
    _recent = _candles_cache.get(sym, [])
    _atr = _wilder_atr(_recent) if len(_recent) >= 2 else 0.0

    avg_cost = pos.get("avg_cost", 0.0)

    # ── SURVIVAL KILL-SWITCH (Upgrade 1) ──────────────────────────────────────
    # Evaluate open PnL from the candle close price BEFORE all other logic so the
    # check cannot be silenced by the fee gate, BE ratchet, or lock re-entry.
    _mark_close = float(candle.get("close", avg_cost))
    _open_pnl   = _hard_stop_open_pnl(pos, _mark_close)
    _is_survival = _open_pnl < HARD_STOP_LOSS_USD
    if _is_survival:
        log.critical(
            "⚠ SURVIVAL KILL-SWITCH (tick) %s — open_pnl=$%.2f < hard floor=$%.2f — "
            "fee gate bypassed",
            sym, _open_pnl, HARD_STOP_LOSS_USD,
        )
    # ── END SURVIVAL KILL-SWITCH ──────────────────────────────────────────────

    # ── TRAILING STOP ─────────────────────────────────────────────────────────
    # Long  : if candle high pushes ABOVE tp_price (original target), trail
    #         the stop up to (candle_high - 1xATR), locking in profit beyond tp.
    # Short : if candle low drops BELOW tp_price (original short target), trail
    #         the stop down to (candle_low + 1xATR), locking in profit.
    # The directional guard means the stop only ever ratchets favourably.
    if _atr > 0 and not is_short:
        if tp > 0 and candle_high > tp:
            new_stop = candle_high - _atr
            if new_stop > stop:
                update_stop_price(sym, new_stop)
                stop = new_stop
                log.info(
                    "TRAIL STOP %s -> $%.4f  (high=$%.4f  atr=$%.4f)",
                    sym, new_stop, candle_high, _atr,
                )
    elif _atr > 0 and is_short:
        if tp > 0 and candle_low < tp:
            new_stop = candle_low + _atr
            if new_stop < stop:
                update_stop_price(sym, new_stop)
                stop = new_stop
                log.info(
                    "TRAIL STOP SHORT %s -> $%.4f  (low=$%.4f  atr=$%.4f)",
                    sym, new_stop, candle_low, _atr,
                )
    # ── END TRAILING STOP ─────────────────────────────────────────────────────

    # ── PURE BREAK-EVEN STOP (V2.2 + Suffocation Fix) ────────────────────────
    # Move stop to exact entry only after a deep, *confirmed* favorable excursion.
    #
    # Two-stage guard:
    #   Stage 1 (wick check) : candle high/low must reach BE_TRIGGER_ATR_MULT × ATR
    #                          from entry — same as before (3 ATR default).
    #   Stage 2 (close check): the *close* price must also be in profit by at least
    #                          BE_MIN_PROFIT_ATR × ATR (1.5 ATR default) or
    #                          BE_MIN_PROFIT_PCT of entry when ATR is unavailable.
    #   Rationale: in choppy markets, a wick can briefly reach +3 ATR while the
    #   candle closes flat.  Moving BE on the wick locks the stop at entry; price
    #   retraces 1 tick and exits at break-even → a fee-loss every single time.
    if _atr > 0 and avg_cost > 0:
        _close_now = float(candle.get("close", avg_cost))
        _min_be_profit = BE_MIN_PROFIT_ATR * _atr   # ATR-based floor always available here
        if is_short:
            if candle_low <= avg_cost - BE_TRIGGER_ATR_MULT * _atr:
                if avg_cost - _close_now >= _min_be_profit:
                    be_stop = avg_cost
                    if be_stop < stop:
                        update_stop_price(sym, be_stop)
                        stop = be_stop
                        log.info(
                            "BREAK-EVEN STOP (short) %s -> $%.4f  (entry=$%.4f  close_profit=$%.4f)",
                            sym, be_stop, avg_cost, avg_cost - _close_now,
                        )
                else:
                    log.info(
                        "BE GUARD (short) %s — close profit $%.4f < min $%.4f (%.1f ATR); BE deferred",
                        sym, avg_cost - _close_now, _min_be_profit, BE_MIN_PROFIT_ATR,
                    )
        else:
            if candle_high >= avg_cost + BE_TRIGGER_ATR_MULT * _atr:
                if _close_now - avg_cost >= _min_be_profit:
                    be_stop = avg_cost
                    if be_stop > stop:
                        update_stop_price(sym, be_stop)
                        stop = be_stop
                        log.info(
                            "BREAK-EVEN STOP (long) %s -> $%.4f  (entry=$%.4f  close_profit=$%.4f)",
                            sym, be_stop, avg_cost, _close_now - avg_cost,
                        )
                else:
                    log.info(
                        "BE GUARD (long) %s — close profit $%.4f < min $%.4f (%.1f ATR); BE deferred",
                        sym, _close_now - avg_cost, _min_be_profit, BE_MIN_PROFIT_ATR,
                    )
    elif avg_cost > 0:
        # ATR unavailable: fall back to absolute % threshold for the close check only.
        _close_now = float(candle.get("close", avg_cost))
        _min_be_profit_abs = avg_cost * BE_MIN_PROFIT_PCT
        if is_short:
            if avg_cost - _close_now < _min_be_profit_abs:
                log.info(
                    "BE GUARD (short/no-ATR) %s — close profit $%.4f < %.1f%% min; BE deferred",
                    sym, avg_cost - _close_now, BE_MIN_PROFIT_PCT * 100,
                )
        else:
            if _close_now - avg_cost < _min_be_profit_abs:
                log.info(
                    "BE GUARD (long/no-ATR) %s — close profit $%.4f < %.1f%% min; BE deferred",
                    sym, _close_now - avg_cost, BE_MIN_PROFIT_PCT * 100,
                )
    # ── ML TIME-DECAY (also on tick so targets move with each bar) ────────────
    if _recent and len(_recent) > 20:
        _apply_ml_time_decay(sym, pos, _recent)
    # ── END BREAK-EVEN / DECAY ────────────────────────────────────────────────

    # Fill model: a take-profit is a resting limit -> fills AT tp. A stop is a
    # stop-market -> fills at the stop only if price is still near it; if the
    # market is already beyond the stop (gap, fast tape, or a stop that sat
    # disabled), it fills at the current price. Filling every stop at its
    # trigger level understated losses, e.g. a stop 5.7% under market booked ~$0.
    _mark_now = float(candle.get("close", 0.0) or 0.0)
    if not is_short:
        if tp > 0 and candle_high >= tp:
            reason = f"take_profit_tick (high=${candle_high:,.4f} >= tp=${tp:,.4f})"
            exit_price = tp
        elif stop > 0 and candle_low <= stop:
            reason = f"stop_loss_tick (low=${candle_low:,.4f} <= stop=${stop:,.4f})"
            exit_price = min(stop, _mark_now) if _mark_now > 0 else stop
    else:
        if tp > 0 and candle_low <= tp:
            reason = f"take_profit_tick_short (low=${candle_low:,.4f} <= tp=${tp:,.4f})"
            exit_price = tp
        elif stop > 0 and candle_high >= stop:
            reason = f"stop_loss_tick_short (high=${candle_high:,.4f} >= stop=${stop:,.4f})"
            exit_price = max(stop, _mark_now)

    # Survival override: no standard SL/TP triggered yet but hard floor crossed.
    # Force an immediate market exit at the current close price.
    if reason is None and _is_survival:
        reason     = f"hard_stop_survival_tick (pnl=${_open_pnl:.2f})"
        exit_price = _mark_close

    if reason is None:
        return

    # ── FEE-BLEED GATE (tick path) ────────────────────────────────────────────
    # Hold a take_profit exit until it would book a positive NET (after exit
    # slippage + round-trip fee). Take-profit only: stop exits of every kind —
    # including trailed / break-even stops — always fire. Gating trailed stops
    # used to disable them outright once price crossed back through entry
    # (profit can never clear fees there), leaving only the $ hard stop.
    _shares_tick = float(pos.get("shares", 0.0))
    if not _is_survival and "take_profit" in reason:
        if not _profit_clears_fees(avg_cost, _shares_tick, exit_price, is_short):
            log.info(
                "FEE GATE blocked %s for %s — net after %.2f%% fees + %.2f%% exit slippage would be <= 0",
                reason, sym, FEE_GATE_ROUND_TRIP * 100, SLIPPAGE_PCT * 100,
            )
            return
    # ── END FEE-BLEED GATE ────────────────────────────────────────────────────

    try:
        _exit_in_flight.add(sym)
        log.info("TICK EXIT %s -- %s", sym, reason)

        candles = _candles_cache.get(sym, [])
        fake_ensemble = {
            "signal":             "cover" if is_short else "sell",
            "exit_reason":        _classify_exit_reason(
                reason, float(avg_cost), float(stop or 0.0), is_short, _initial_stop(pos)),
            "exit_detail":        reason,
            "symbol":             sym,
            "price":              exit_price,
            "regime":             brain.current_regime,
            "on_fire":            False,
            "position_size_mult": 1.0,
            "buy_weight":         0.0,
            "sell_weight":        0.0,
            "breakdown":          [],
            "time":               _now(),
        }

        # Fix #2 + #10: strict global commit ordering, then per-symbol + cash.
        # Fix C4: _trade_lock(sym) is now the OUTERMOST lock, spanning the
        # whole prepare -> network -> commit sequence, so a same-symbol
        # operation can't sneak in during the network call. _strict_execution_lock
        # is only held for the two local bookkeeping phases and is released
        # while the exchange call is in flight, so one slow order never blocks
        # the other 46 symbols' exit checks.
        async with _trade_lock(sym):
            prepared = None
            async with _strict_execution_lock:
                pos_list2 = get_all_positions()
                still_open = next(
                    (p for p in pos_list2 if p["symbol"] == sym), None)
                if not still_open:
                    return
                if is_short and still_open.get("margin_reserved", 0) <= 0:
                    return
                if not is_short and still_open.get("shares", 0) <= 0:
                    return

                cash = get_cash()
                open_pos = get_all_positions()
                total_eq = _compute_total_equity(cash, open_pos)

                prepared = await loop.run_in_executor(
                    None,
                    _prepare_trade,
                    fake_ensemble,
                    pos.get("strategy", "TICK_EXIT"),
                    candles,
                    None,   # sac_fraction=None -> conservative 2% fallback
                    np.zeros(13, dtype=np.float32),
                    total_eq,
                    None,
                )
            if prepared is None:
                return
            fill = await loop.run_in_executor(None, _exec_adapter.execute, prepared.order_intent)
            async with _strict_execution_lock:
                async with _cash_lock:
                    await loop.run_in_executor(None, _commit_trade, prepared, fill)
    finally:
        _exit_in_flight.discard(sym)


# ── WebSocket message handler ─────────────────────────────────────────────────

async def _handle_ws_message(raw_msg: str, loop: asyncio.AbstractEventLoop) -> None:
    try:
        msg = json.loads(raw_msg)
    except json.JSONDecodeError:
        return

    data = msg.get("data", {})
    kline = data.get("k", {})
    sym = data.get("s", "")

    if not kline or sym not in SYMBOLS:
        return

    try:
        candle = {
            "time":   int(kline["t"]),
            "open":   float(kline["o"]),
            "high":   float(kline["h"]),
            "low":    float(kline["l"]),
            "close":  float(kline["c"]),
            "volume": float(kline["v"]),
        }
    except (KeyError, ValueError) as exc:
        log.warning("Malformed kline for %s: %s", sym, exc)
        return

    # Fix #23 (W8): every tick used to hit SQLite twice here (upsert_candle +
    # set_portfolio_stat last_price) for up to 47 symbols -- ~90 writes/sec on
    # the event-loop thread for data already held in _candles_cache. The
    # in-memory cache still updates on every tick (that's the whole point of
    # tick-level reactivity for stops); SQLite is now only touched once per
    # symbol per closed candle, below.
    if sym in _candles_cache:
        cache = _candles_cache[sym]
        if cache and cache[-1]["time"] == candle["time"]:
            cache[-1] = candle
        else:
            cache.append(candle)
            if len(cache) > CANDLE_LIMIT:
                cache.pop(0)

    asyncio.create_task(_tick_exit_check(sym, candle, loop))

    is_closed = bool(kline.get("x", False))
    if not is_closed:
        return

    upsert_candle(sym, candle)
    set_portfolio_stat(f"last_price_{sym}", candle["close"])

    _closed_candle_counts[sym] = _closed_candle_counts.get(sym, 0) + 1
    count = _closed_candle_counts[sym]
    candles = _candles_cache.get(sym, [])
    if len(candles) < 60:
        return

    try:
        ml_prob = await loop.run_in_executor(_executor, predict_signal, list(candles))
    except Exception as exc:
        log.warning("ML predict error for %s: %s", sym, exc)
        ml_prob = 0.5

    brain.update_ml_prob(sym, ml_prob)
    save_ml_cache(sym, [], ml_prob)

    if count % ML_UPDATE_EVERY == 0:
        log.info("Scheduling incremental XGB update for %s", sym)
        asyncio.create_task(_run_incremental_train(loop, sym, list(candles)))

    asyncio.create_task(_evaluate_and_trade(sym, list(candles), ml_prob, loop))


async def _run_incremental_train(loop: asyncio.AbstractEventLoop, sym: str, candles: list[dict]) -> None:
    # Fix #17: check class balance before submitting to the trainer.
    # incremental_train internally derives labels from price direction; we mirror
    # that here by checking whether recent closes are monotonically one-directional.
    closes = [c["close"]
              for c in candles[-60:]]  # last 60 candles is sufficient
    if len(closes) >= 2:
        labels = [1 if closes[i] > closes[i - 1]
                  else 0 for i in range(1, len(closes))]
        if not _is_trainable(labels):
            return   # skip — degenerate batch would cause learner.cc:782 hallucination

    try:
        result = await loop.run_in_executor(_executor, incremental_train, candles)
        if result:
            log.info("Incremental XGB update complete for %s", sym)
    except Exception as exc:
        log.error("Incremental train error: %s", exc)


# ── Evaluate and trade ────────────────────────────────────────────────────────

async def _evaluate_and_trade(
    symbol:   str,
    candles:  list[dict],
    ml_prob:  float,
    loop:     asyncio.AbstractEventLoop,
) -> None:
    if _shutdown_event.is_set():
        return

    # Fix #4: use module-level `time` import; rename local var to _ts to avoid
    # shadowing the module-level _now() function.
    _ts = time.time()
    if _ts - _symbol_last_trade.get(symbol, 0) < SYMBOL_COOLDOWN_SECS:
        return
    _symbol_last_trade[symbol] = _ts

    open_positions = get_all_positions()
    pos = next((p for p in open_positions if p["symbol"] == symbol), None)
    pos_side = pos.get("side", "long") if pos else None

    ensemble = brain.get_ensemble_signal(
        symbol,
        candles,
        position_side=pos_side,
        ml_prob=ml_prob,
        btc_candles=_candles_cache.get("BTCUSDT"),
        eth_candles=_candles_cache.get("ETHUSDT"),
    )
    direction = ensemble["signal"]

    if direction not in ("buy", "sell", "short", "cover"):
        return

    # V4: cheap early stale-data skip for entries (executor re-checks — this
    # just avoids burning SAC/margin work on a dead feed).
    if direction in ("buy", "short"):
        _stale_early = _entry_data_stale_reason(candles)
        if _stale_early:
            log.info("%s %s skipped — %s", symbol, direction.upper(), _stale_early)
            return

    # ── Belt-and-suspenders: refuse any inbound entry ensemble whose ml_prob
    # field is missing or out of range. Pin ml_prob to the freshly-computed
    # XGBoost value so downstream gates see a single, trusted source.
    if direction in ("buy", "short"):
        _emb_p = ensemble.get("ml_prob")
        try:
            _emb_p_f = float(_emb_p) if _emb_p is not None else None
        except (TypeError, ValueError):
            _emb_p_f = None
        if _emb_p_f is None or not (0.0 <= _emb_p_f <= 1.0):
            log.warning(
                "%s %s REFUSED — ensemble missing/invalid ml_prob (%r)",
                symbol, direction.upper(), _emb_p,
            )
            return
        try:
            _live_p = float(ml_prob)
        except (TypeError, ValueError):
            log.warning(
                "%s %s REFUSED — live ml_prob unusable (%r)",
                symbol, direction.upper(), ml_prob,
            )
            return
        if not (0.0 <= _live_p <= 1.0):
            log.warning(
                "%s %s REFUSED — live ml_prob out of range (%.6f)",
                symbol, direction.upper(), _live_p,
            )
            return
        _live_conf = (1.0 - _live_p) if direction == "short" else _live_p
        if _live_conf < MIN_ML_CONFIDENCE:
            log.info(
                "%s %s REFUSED — live ML confidence %.4f < %.4f (ml_prob=%.4f)",
                symbol, direction.upper(), _live_conf, MIN_ML_CONFIDENCE, _live_p,
            )
            return

    cm_authoritative = _authoritative_conviction_mult(ensemble, ml_prob)
    ensemble = dict(ensemble)
    ensemble["position_size_mult"] = cm_authoritative
    ensemble["ml_prob"] = float(ml_prob)

    if direction == "buy" and brain.circuit_open:
        log.info("%s BUY blocked - circuit breaker open", symbol)
        return

    if direction == "buy" and open_position_count() >= MAX_OPEN_POSITIONS:
        return
    if direction == "short" and _short_slots_full(symbol):
        return
    # Fix #22: a short already open on this symbol must never be re-opened —
    # open_short()'s INSERT OR REPLACE would silently orphan the first short's
    # margin/stop/opened_ts tracking and leak its reserved margin from cash
    # forever. brain.get_ensemble_signal() does not consider position_side,
    # so this gate is the only thing standing between a persistent trend and
    # a duplicate entry.
    if direction == "short" and get_short_position(symbol) is not None:
        log.warning(
            "%s SHORT blocked — a short is already open on this symbol", symbol,
        )
        return

    last = candles[-1]
    if last["low"] > 0 and (last["high"] - last["low"]) / last["low"] < 0.0005:
        return

    cash = get_cash()
    unrealised_pnl = 0.0
    for p in open_positions:
        c_price = _candles_cache.get(
            p["symbol"], [{"close": p["avg_cost"]}])[-1]["close"]
        if p.get("side", "long") == "long":
            unrealised_pnl += (c_price - p["avg_cost"]) * p["shares"]
        else:
            unrealised_pnl += (p["avg_cost"] - c_price) * p["shares"]

    total_equity = _compute_total_equity(cash, open_positions)
    state_vec = brain.compute_sac_state(
        symbol, candles, cash, unrealised_pnl, total_equity)

    try:
        sac_fraction = await loop.run_in_executor(
            _executor, compute_position_fraction, state_vec
        )
    except Exception as exc:
        log.warning("SAC inference error: %s - using fallback", exc)
        sac_fraction = None

    strategy_name = _top_strategy(ensemble, direction)

    # V2 — entries: one global commit frame (ρ / exposure / DB) at a time; SAC stays outside.
    # Fix C4: _trade_lock(symbol) now spans the whole sequence (outermost),
    # while _strict_execution_lock is released around the exchange call so a
    # slow order on this symbol can't block the other 46 symbols' exit checks
    # or entries. Margin-health/correlation/exposure checks below are network-
    # and CPU-bound but NOT the exchange order call itself, so they stay under
    # the global lock as before (out of scope for this pass — see summary).
    if direction in ("buy", "short"):
        prepared = None
        async with _trade_lock(symbol):
            async with _strict_execution_lock:
                open_positions = get_all_positions()
                if direction == "buy" and open_position_count() >= MAX_OPEN_POSITIONS:
                    return
                if direction == "short" and _short_slots_full(symbol):
                    return
                # Fix #22 (re-check): close the TOCTOU window between the early
                # gate above and acquiring this commit frame's lock.
                if direction == "short" and get_short_position(symbol) is not None:
                    log.warning(
                        "%s SHORT blocked at commit frame — a short opened on this "
                        "symbol while sizing was in flight", symbol,
                    )
                    return

                cash_e = get_cash()
                unrealised_e = 0.0
                for p in open_positions:
                    c_price = _candles_cache.get(
                        p["symbol"], [{"close": p["avg_cost"]}])[-1]["close"]
                    if p.get("side", "long") == "long":
                        unrealised_e += (c_price - p["avg_cost"]) * p["shares"]
                    else:
                        unrealised_e += (p["avg_cost"] - c_price) * p["shares"]
                total_equity = _compute_total_equity(cash_e, open_positions)

                async with aiohttp.ClientSession() as _sess:
                    await _refresh_margin_health(_sess, cash_e, total_equity)
                if _margin_health_cache.get("halt_new_entries"):
                    log.info(
                        "%s entry blocked — margin %s",
                        symbol, _margin_health_cache.get("halt_reason"),
                    )
                    return
                peer_syms = [p["symbol"] for p in open_positions if p["symbol"] != symbol]
                if peer_syms:
                    blocked_corr, worst_rho = correlation_blocks_entry(
                        candles,
                        peer_syms,
                        _candles_cache,
                        rho_threshold=CORRELATION_THRESHOLD,
                        max_high_corr_peers=MAX_CORRELATED_OPEN_PEERS,
                    )
                    if blocked_corr:
                        log.info("%s entry blocked — ρ-cluster (worst |ρ|=%.3f)", symbol, worst_rho)
                        return
                sf = sac_fraction if sac_fraction is not None else _SAC_FALLBACK_PCT
                proposed_frac = _entry_proposed_equity_frac(float(sf), cm_authoritative)
                marks = {
                    p["symbol"]: float(
                        (_candles_cache.get(p["symbol"]) or [{"close": p["avg_cost"]}])[-1]["close"]
                    )
                    for p in open_positions
                }
                blocked_exp, cur_exp = exposure_blocked(
                    open_positions,
                    marks,
                    total_equity,
                    proposed_frac,
                    global_cap=GLOBAL_POSITION_NOTIONAL_CAP,
                )
                if blocked_exp:
                    log.info(
                        "%s entry blocked — exposure %.1f%% + proposed %.1f%% > cap %.0f%%",
                        symbol,
                        cur_exp * 100,
                        proposed_frac * 100,
                        GLOBAL_POSITION_NOTIONAL_CAP * 100,
                    )
                    return

                prepared = await loop.run_in_executor(
                    None,
                    _prepare_trade,
                    ensemble, strategy_name, candles, sac_fraction, state_vec, total_equity,
                    cm_authoritative,
                )
            if prepared is None:
                return
            fill = await loop.run_in_executor(None, _exec_adapter.execute, prepared.order_intent)
            async with _strict_execution_lock:
                async with _cash_lock:
                    await loop.run_in_executor(None, _commit_trade, prepared, fill)
        return

    prepared = None
    async with _trade_lock(symbol):
        async with _strict_execution_lock:
            prepared = await loop.run_in_executor(
                None,
                _prepare_trade,
                ensemble, strategy_name, candles, sac_fraction, state_vec, total_equity,
                None,
            )
        if prepared is None:
            return
        fill = await loop.run_in_executor(None, _exec_adapter.execute, prepared.order_intent)
        async with _strict_execution_lock:
            async with _cash_lock:
                await loop.run_in_executor(None, _commit_trade, prepared, fill)


# ── Core trade executor ───────────────────────────────────────────────────────
#
# Fix C4: split into _prepare_trade (gates/sizing/order-journal claim) and
# _commit_trade (booking) around the exchange network call. Previously the
# whole thing ran as one function under the caller's global lock, so a slow
# testnet order for one symbol held _strict_execution_lock for the entire
# round-trip, blocking every other symbol's stop-loss/take-profit evaluation.
# Async callers now: acquire the global lock -> _prepare_trade -> release ->
# call the adapter unlocked (still under that symbol's _trade_lock) ->
# reacquire the global lock -> _commit_trade. _execute_trade itself remains a
# synchronous wrapper around all three steps, preserving the original direct-
# call interface so existing callers/tests are unaffected.

@dataclass
class _PreparedTrade:
    """Carries state from _prepare_trade to _commit_trade across the
    (now-unlocked) network call. Fields not relevant to a given action stay
    at their defaults."""
    action: str
    symbol: str
    order_intent: OrderIntent
    trade_rec: dict
    candles: list
    regime: str
    strategy_name: str
    on_fire: bool
    pre_state: Optional[np.ndarray]
    sac_fraction: Optional[float]
    total_equity: float
    # buy / short only
    is_yolo: bool = False
    stop_price: float = 0.0
    tp_price: float = 0.0
    atr_now: float = 0.0
    trade_pct: float = 0.0
    confidence_multiplier: float = 1.0
    entry_features: Optional[dict] = None   # short entry: decision snapshot for the trade journal
    # sell / cover only
    pos: Optional[dict] = None
    pos_shares: float = 0.0
    avg_cost: float = 0.0
    journaled: bool = True
    margin_res_total: float = 0.0  # cover only


def _prepare_trade(
    ensemble:      dict,
    strategy_name: str,
    candles:       list[dict],
    # Fix #8: None = SAC failed, use 2% fallback
    sac_fraction:  Optional[float],
    pre_state:     np.ndarray,
    total_equity:  float,            # Fix #7: passed in, not re-fetched from DB
    conviction_mult: Optional[float] = None,
) -> Optional[_PreparedTrade]:
    """
    Phase 1 of 2 (Fix C4): every gate, the sizing decision, and the
    order-journal claim -- everything that must happen before any money
    moves. Runs under _strict_execution_lock. Returns None if the trade is
    rejected (already logged via log_trade); otherwise a _PreparedTrade ready
    for the exchange call in _commit_trade.
    """
    action = ensemble["signal"]
    symbol = ensemble["symbol"]
    raw_price = ensemble["price"]
    regime = ensemble.get("regime", "ranging")
    on_fire = ensemble.get("on_fire", False)

    exec_price = (
        raw_price * (1 + SLIPPAGE_PCT) if action in ("buy", "cover")
        else raw_price * (1 - SLIPPAGE_PCT)
    )

    trade_rec = {
        "symbol":      symbol,         "action":     action,
        "strategy":    strategy_name,  "regime":     regime,
        "price":       raw_price,      "exec_price": exec_price,
        "on_fire":     on_fire,
        "timestamp":   _now(),
        "side":        "long" if action in ("buy", "sell") else "short",
        "slippage":    round(abs(exec_price - raw_price), 6),
        "status":      "skipped",
        # Fix 4 — NULL VIRUS: seed with explicit zero defaults so early-return
        # paths (no_position, insufficient_funds, gate blocks, etc.) always log
        # a non-NULL value.  Branch-specific update() calls overwrite these.
        "shares":      0.0,
        "trade_value": 0.0,
    }
    # Trade journal: the exit path stamps the canonical trigger on the ensemble.
    if ensemble.get("exit_reason"):
        trade_rec["exit_reason"] = ensemble["exit_reason"]
        trade_rec["exit_detail"] = ensemble.get("exit_detail")
    # Entries: journal the decision snapshot on every outcome, skips included.
    if action in ("buy", "short"):
        trade_rec["entry_features"] = _entry_snapshot(
            ensemble, action, regime, strategy_name, sac_fraction, total_equity,
        )

    # ── HARD ML CONFIDENCE GATE (defense-in-depth) ────────────────────────────
    # Single source of truth for entries. brain.get_ensemble_signal already
    # gates shorts on ml_down >= MIN_ML_CONFIDENCE, but we re-validate at the
    # money-pulling layer so no caller — present or future, real or synthetic —
    # can open a position without strict ML conviction.
    #
    # ml_prob = P(profitable LONG move) from XGBoost (see ml_engine.predict_signal).
    #   long  conviction = ml_prob
    #   short conviction = 1 - ml_prob
    #
    # Exits ("sell" / "cover") intentionally bypass this gate — they are
    # stop-loss / take-profit driven and must always be allowed to fire.
    if action in ("buy", "short"):
        try:
            _ml_prob = float(ensemble.get("ml_prob"))
        except (TypeError, ValueError):
            _ml_prob = None
        if _ml_prob is None or not (0.0 <= _ml_prob <= 1.0):
            trade_rec["reason"] = "ml_gate_missing_prob"
            log.warning(
                "%s %s BLOCKED at executor — ml_prob missing/invalid in ensemble",
                symbol, action.upper(),
            )
            log_trade(trade_rec)
            return None
        ml_conf = (1.0 - _ml_prob) if action == "short" else _ml_prob
        if ml_conf < MIN_ML_CONFIDENCE:
            trade_rec["reason"] = (
                f"ml_gate_blocked (conf={ml_conf:.4f} < min={MIN_ML_CONFIDENCE:.4f})"
            )
            log.warning(
                "%s %s BLOCKED at executor — ML confidence %.4f < %.4f "
                "(ml_prob=%.4f)",
                symbol, action.upper(), ml_conf, MIN_ML_CONFIDENCE, _ml_prob,
            )
            log_trade(trade_rec)
            return None

        # ── V4 STALE-DATA GATE (entries only) ─────────────────────────────
        # A stalled feed (WS outage, reconnect gap) must never open a new
        # position at a price that may no longer exist. Exits bypass this —
        # closing on the freshest price we have beats staying exposed.
        _stale = _entry_data_stale_reason(candles)
        if _stale:
            trade_rec["reason"] = _stale
            log.warning("%s %s BLOCKED at executor — %s", symbol, action.upper(), _stale)
            log_trade(trade_rec)
            return None

    # ── Shared SAC-driven sizing (used by BUY and SHORT) ──────────────────────
    # Set when sizing deliberately returns zero, so the skip row records the
    # real cause instead of a misleading insufficient_funds/margin.
    size_veto_reason: Optional[str] = None

    def _calc_trade_value() -> tuple[float, float, float]:
        nonlocal size_veto_reason
        """
        Returns (trade_value, trade_pct, confidence_multiplier).

        Three SAC states:
          sac_fraction is None   -> inference crash -> warn + 2% fallback
          sac_fraction <= veto   -> AI veto         -> return (0,0,cm)
          sac_fraction > veto    -> live signal      -> size normally

        Sizing pipeline (in order):
          1. base_pct  = _entry_proposed_equity_frac(sac, cm_eff × dyn_size_mult)
             SAC × ML-conviction only; edge_position_mult is intentionally excluded
             here — its conf_scale blending amplified size_mult until the 35%
             ceiling absorbed every AI penalty.
          2. final_pct = base_pct × raw_size_mult (direct edge-profile penalty)
             raw_size_mult is read straight from brain._edge_profile(), giving the
             exact "BASE_ALLOCATION_PCT × size_mult" semantics the adaptive AI
             targets.
          3. If final trade_value < BINANCE_MIN_NOTIONAL, the AI penalty has vetoed
             the trade by making it unexecutable → blocked with a dedicated log.
        """
        cm_eff = (
            float(conviction_mult)
            if conviction_mult is not None
            else float(ensemble.get("position_size_mult", 1.0))
        )
        edge_side = "short" if action == "short" else "long"

        # Read size_mult directly from the edge profile — no conf_scale blending,
        # no EDGE_SIZE_MIN_MULT floor clamp that was masking the penalty.
        _ep = brain._edge_profile(edge_side, regime)
        raw_size_mult = float(_ep.get("size_mult", 1.0))

        try:
            _ml_p_sz = float(ensemble.get("ml_prob", 0.5))
        except (TypeError, ValueError):
            _ml_p_sz = 0.5
        dyn_size_mult = dynamic_conviction_size_mult(_ml_p_sz, edge_side)
        # edge_mult removed from this product — applied separately below as raw_size_mult
        confidence_multiplier = cm_eff * dyn_size_mult

        if sac_fraction is None:
            log.warning(
                "SAC inference crash for %s - falling back to %.0f%% equity",
                symbol, _SAC_FALLBACK_PCT * 100,
            )
            trade_pct = _entry_proposed_equity_frac(_SAC_FALLBACK_PCT, confidence_multiplier)

        elif sac_fraction <= _SAC_VETO_THRESH:
            log.info(
                "SAC agent VETOED trade on %s (sac_fraction=%.4f, regime=%s)",
                symbol, sac_fraction, ensemble.get("regime", "unknown"),
            )
            size_veto_reason = "sac_veto"
            return 0.0, 0.0, confidence_multiplier

        else:
            trade_pct = _entry_proposed_equity_frac(float(sac_fraction), confidence_multiplier)

        # ── Apply edge-profile size_mult as a direct final multiplier ──────────
        # final_allocation = base_allocation * size_mult
        # This is the link that was missing: size_mult now unambiguously scales
        # whatever the SAC+ML engine proposed, regardless of the ceiling.
        trade_pct = trade_pct * raw_size_mult
        log.debug(
            "SIZING %s [%s:%s] base_pct=%.3f size_mult=%.4f final_pct=%.3f (%.1f%%)",
            symbol, edge_side, regime,
            trade_pct / max(raw_size_mult, 1e-9),  # show base before penalty
            raw_size_mult, trade_pct, trade_pct * 100,
        )

        trade_value = total_equity * trade_pct

        # ── V4 PER-ORDER CEILING (defense in depth) ────────────────────────
        # Current sizing maths tops out at 0.35 × 1.35 = 47.25% of equity, so
        # at the default 50% this clamp never fires — it exists to stop a
        # future sizing bug from deploying the whole account in one order.
        _order_cap_value = total_equity * MAX_ORDER_EQUITY_FRAC
        if trade_value > _order_cap_value:
            log.warning(
                "ORDER CAP: %s %s trade_value $%.2f exceeds %.0f%% of equity — "
                "clamped to $%.2f (sizing bug upstream?)",
                symbol, action.upper(), trade_value,
                MAX_ORDER_EQUITY_FRAC * 100, _order_cap_value,
            )
            trade_value = _order_cap_value
            trade_pct = MAX_ORDER_EQUITY_FRAC

        # Binance requires ≥ $10 notional per order; a heavily penalised
        # size_mult can push a small account below this floor — veto cleanly.
        _BINANCE_MIN_NOTIONAL = 10.0
        if 0 < trade_value < _BINANCE_MIN_NOTIONAL:
            log.info(
                "Trade vetoed by AI penalty: %s [%s:%s] size_mult=%.4f → "
                "trade_value=$%.2f < min_notional=$%.2f",
                symbol, edge_side, regime, raw_size_mult,
                trade_value, _BINANCE_MIN_NOTIONAL,
            )
            size_veto_reason = "size_below_min_notional"
            return 0.0, 0.0, confidence_multiplier

        return trade_value, trade_pct, confidence_multiplier
    # ── End sizing helper ─────────────────────────────────────────────────────

    if action == "buy":
        cash = get_cash()
        trade_value, trade_pct, confidence_multiplier = _calc_trade_value()
        trade_rec["entry_features"].update({
            "conviction_mult": round(float(confidence_multiplier), 4),
            "trade_pct": round(float(trade_pct), 6),
        })

        if size_veto_reason:
            trade_rec["reason"] = size_veto_reason
            log_trade(trade_rec)
            return None
        if cash < trade_value or trade_value < 1.0:
            trade_rec["reason"] = "insufficient_funds"
            log_trade(trade_rec)
            return None

        shares = trade_value / exec_price
        is_yolo = strategy_name == "YOLO_FIRE"
        stop_price, tp_price = brain.get_stop_take(
            exec_price, candles, is_yolo, side="long", regime=regime,
            confidence_mult=confidence_multiplier,
        )
        atr_now = _wilder_atr(candles) if candles else 0.0
        stop_price, tp_price = apply_dynamic_rr(
            entry=exec_price, stop=stop_price, take=tp_price,
            atr=atr_now, side="long",
        )
        tp_gross = max(0.0, shares * max(0.0, tp_price - exec_price))
        tp_friction = entry_exit_fees_notional(trade_value, trade_value)
        tp_net_est = tp_gross - tp_friction
        if tp_net_est < MIN_EXPECTED_TP_NET_USD:
            trade_rec["reason"] = "edge_too_small"
            log_trade(trade_rec)
            log.info(
                "BUY %s blocked — projected TP net edge $%.4f < min $%.2f",
                symbol, tp_net_est, MIN_EXPECTED_TP_NET_USD,
            )
            return None

        # ── V4 DUPLICATE-ORDER GUARD ───────────────────────────────────────
        # Claim the deterministic (symbol, action, candle) order id BEFORE any
        # money moves. A second identical signal — retry, WS replay, racing
        # task — fails the claim and cannot double the position.
        _candle_ts = _last_candle_ts(candles)
        _coid = make_client_order_id(symbol, action, _candle_ts)
        if not try_create_order(_coid, symbol, action, "long", _candle_ts,
                                shares, exec_price, _exec_adapter.name):
            trade_rec["reason"] = "duplicate_order_blocked"
            log.warning(
                "BUY %s BLOCKED — duplicate order for this candle (id=%s)",
                symbol, _coid,
            )
            log_trade(trade_rec)
            return None

        return _PreparedTrade(
            action=action, symbol=symbol,
            order_intent=OrderIntent(
                client_order_id=_coid, symbol=symbol, action=action, side="long",
                qty=shares, ref_price=raw_price, limit_price=exec_price,
                candle_ts=_candle_ts,
            ),
            trade_rec=trade_rec, candles=candles, regime=regime,
            strategy_name=strategy_name, on_fire=on_fire, pre_state=pre_state,
            sac_fraction=sac_fraction, total_equity=total_equity,
            is_yolo=is_yolo, stop_price=stop_price, tp_price=tp_price,
            atr_now=atr_now, trade_pct=trade_pct,
            confidence_multiplier=confidence_multiplier,
        )

    elif action == "sell":
        pos = next((p for p in get_all_positions() if p["symbol"] == symbol), None)
        if not pos or pos.get("shares", 0) <= 0:
            trade_rec["reason"] = "no_position"
            log_trade(trade_rec)
            return None

        pos_shares = float(pos["shares"])
        avg_cost   = float(pos["avg_cost"])

        # ── V4 EXECUTION (exit — fail-open on the journal, position-safe) ──
        # Exits must never be blocked by bookkeeping: if the journal claim
        # fails (e.g. a crashed prior attempt already holds the id for this
        # candle) we proceed anyway.
        _candle_ts = _last_candle_ts(candles)
        _coid = make_client_order_id(symbol, action, _candle_ts)
        _journaled = try_create_order(_coid, symbol, action, "long", _candle_ts,
                                      pos_shares, exec_price, _exec_adapter.name)

        return _PreparedTrade(
            action=action, symbol=symbol,
            order_intent=OrderIntent(
                client_order_id=_coid, symbol=symbol, action=action, side="long",
                qty=pos_shares, ref_price=raw_price, limit_price=exec_price,
                candle_ts=_candle_ts,
            ),
            trade_rec=trade_rec, candles=candles, regime=regime,
            strategy_name=strategy_name, on_fire=on_fire, pre_state=pre_state,
            sac_fraction=sac_fraction, total_equity=total_equity,
            pos=pos, pos_shares=pos_shares, avg_cost=avg_cost, journaled=_journaled,
        )

    elif action == "short":
        cash = get_cash()
        trade_value, trade_pct, confidence_multiplier = _calc_trade_value()
        trade_rec["entry_features"].update({
            "conviction_mult": round(float(confidence_multiplier), 4),
            "trade_pct": round(float(trade_pct), 6),
        })
        margin_reserved = trade_value * SHORT_MARGIN_PCT

        if size_veto_reason:
            trade_rec["reason"] = size_veto_reason
            log_trade(trade_rec)
            return None
        if cash < margin_reserved or margin_reserved < 1.0:
            trade_rec["reason"] = "insufficient_margin"
            log_trade(trade_rec)
            return None

        shares = trade_value / exec_price
        is_yolo = strategy_name == "YOLO_FIRE"
        stop_price, tp_price = brain.get_stop_take(
            exec_price, candles, is_yolo, side="short", regime=regime,
            confidence_mult=confidence_multiplier,
        )
        atr_now = _wilder_atr(candles) if candles else 0.0
        stop_price, tp_price = apply_dynamic_rr(
            entry=exec_price, stop=stop_price, take=tp_price,
            atr=atr_now, side="short",
        )
        tp_gross = max(0.0, shares * max(0.0, exec_price - tp_price))
        tp_friction = entry_exit_fees_notional(trade_value, trade_value)
        tp_net_est = tp_gross - tp_friction
        if tp_net_est < MIN_EXPECTED_TP_NET_USD:
            trade_rec["reason"] = "edge_too_small"
            log_trade(trade_rec)
            log.info(
                "SHORT %s blocked — projected TP net edge $%.4f < min $%.2f",
                symbol, tp_net_est, MIN_EXPECTED_TP_NET_USD,
            )
            return None

        # ── V4 DUPLICATE-ORDER GUARD (short) ───────────────────────────────
        _candle_ts = _last_candle_ts(candles)
        _coid = make_client_order_id(symbol, action, _candle_ts)
        if not try_create_order(_coid, symbol, action, "short", _candle_ts,
                                shares, exec_price, _exec_adapter.name):
            trade_rec["reason"] = "duplicate_order_blocked"
            log.warning(
                "SHORT %s BLOCKED — duplicate order for this candle (id=%s)",
                symbol, _coid,
            )
            log_trade(trade_rec)
            return None

        _entry_features = trade_rec["entry_features"]
        _entry_features.update({
            "atr_pct": round(atr_now / exec_price, 6) if exec_price else None,
            "stop_pct": round((stop_price - exec_price) / exec_price * 100.0, 4),
            "tp_pct": round((exec_price - tp_price) / exec_price * 100.0, 4),
            "is_yolo": bool(is_yolo),
        })

        return _PreparedTrade(
            action=action, symbol=symbol,
            entry_features=_entry_features,
            order_intent=OrderIntent(
                client_order_id=_coid, symbol=symbol, action=action, side="short",
                qty=shares, ref_price=raw_price, limit_price=exec_price,
                candle_ts=_candle_ts,
            ),
            trade_rec=trade_rec, candles=candles, regime=regime,
            strategy_name=strategy_name, on_fire=on_fire, pre_state=pre_state,
            sac_fraction=sac_fraction, total_equity=total_equity,
            is_yolo=is_yolo, stop_price=stop_price, tp_price=tp_price,
            atr_now=atr_now, trade_pct=trade_pct,
            confidence_multiplier=confidence_multiplier,
        )

    elif action == "cover":
        pos = next((p for p in get_all_positions()
                    if p["symbol"] == symbol), None)
        if not pos or pos.get("shares", 0) <= 0:
            trade_rec["reason"] = "no_position"
            log_trade(trade_rec)
            return None

        pos_shares = float(pos["shares"])
        avg_cost   = float(pos["avg_cost"])
        margin_res_total = float(pos.get("margin_reserved") or 0.0)

        # ── V4 EXECUTION (cover — fail-open journal, position-safe) ────────
        _candle_ts = _last_candle_ts(candles)
        _coid = make_client_order_id(symbol, action, _candle_ts)
        _journaled = try_create_order(_coid, symbol, action, "short", _candle_ts,
                                      pos_shares, exec_price, _exec_adapter.name)

        return _PreparedTrade(
            action=action, symbol=symbol,
            order_intent=OrderIntent(
                client_order_id=_coid, symbol=symbol, action=action, side="short",
                qty=pos_shares, ref_price=raw_price, limit_price=exec_price,
                candle_ts=_candle_ts,
            ),
            trade_rec=trade_rec, candles=candles, regime=regime,
            strategy_name=strategy_name, on_fire=on_fire, pre_state=pre_state,
            sac_fraction=sac_fraction, total_equity=total_equity,
            pos=pos, pos_shares=pos_shares, avg_cost=avg_cost, journaled=_journaled,
            margin_res_total=margin_res_total,
        )

    # Unreachable in practice — callers only ever pass buy/sell/short/cover.
    trade_rec["reason"] = f"unknown_action:{action}"
    log_trade(trade_rec)
    return None


def _commit_trade(prepared: _PreparedTrade, fill) -> None:
    """
    Phase 2 of 2 (Fix C4): booking. Runs under _strict_execution_lock again,
    AFTER the exchange call has already completed with the global lock
    released. Cash is re-read fresh here (not carried over from the prepare
    phase) for buy/short specifically — sell/cover already did this in the
    original code — since another symbol's trade may have moved it during
    this order's network round-trip.
    """
    action        = prepared.action
    symbol        = prepared.symbol
    trade_rec     = prepared.trade_rec
    candles       = prepared.candles
    regime        = prepared.regime
    strategy_name = prepared.strategy_name
    on_fire       = prepared.on_fire
    pre_state     = prepared.pre_state
    sac_fraction  = prepared.sac_fraction
    total_equity  = prepared.total_equity
    _coid         = prepared.order_intent.client_order_id
    raw_price     = prepared.order_intent.ref_price
    exec_price    = prepared.order_intent.limit_price

    if action == "buy":
        if not fill.executed:
            update_order(_coid, state=fill.status if fill.status in
                         ("REJECTED", "FAILED") else "FAILED", note=fill.note)
            trade_rec["reason"] = f"execution_{fill.status.lower()}"
            log.warning("BUY %s not executed (%s: %s)", symbol, fill.status, fill.note)
            log_trade(trade_rec)
            return

        stop_price, tp_price = prepared.stop_price, prepared.tp_price
        if abs(fill.price - exec_price) > exec_price * 1e-9:
            stop_price, tp_price = brain.get_stop_take(
                fill.price, candles, prepared.is_yolo, side="long", regime=regime,
                confidence_mult=prepared.confidence_multiplier,
            )
            stop_price, tp_price = apply_dynamic_rr(
                entry=fill.price, stop=stop_price, take=tp_price,
                atr=prepared.atr_now, side="long",
            )
        exec_price = fill.price
        shares = fill.qty
        trade_value = shares * exec_price
        trade_rec["exec_price"] = exec_price

        # Fix C4: fresh cash read at commit time, not a prepare-phase
        # snapshot — the network call above ran with the global lock
        # released, so other symbols' trades may have already spent cash.
        cash = get_cash()
        if cash < trade_value:
            log.warning(
                "BUY %s committing with cash=$%.2f < trade_value=$%.2f — the "
                "fill already executed and cannot be undone; other symbols' "
                "trades likely consumed cash during this order's round-trip",
                symbol, cash, trade_value,
            )
        set_cash(cash - trade_value)
        open_position(
            symbol, shares, exec_price, strategy_name,
            stop_price, tp_price, on_fire,
            entry_state=pre_state.tolist() if pre_state is not None else None,
        )
        update_order(_coid, state=fill.status, filled_qty=shares,
                     avg_fill_price=exec_price,
                     exchange_order_id=fill.exchange_order_id, note=fill.note)
        set_portfolio_stat("total_trades", int(get_portfolio_stat("total_trades", "0")) + 1)

        trade_rec.update({
            "status":      "filled",
            "shares":      round(float(shares or 0.0), 8),
            "trade_value": round(float(trade_value or 0.0), 2),
            "stop_price":  stop_price,
            "tp_price":    tp_price,
            "allocated_equity_pct": round(100.0 * float(trade_value or 0.0) / max(total_equity, 1e-9), 4),
        })

        alert_sniper_shot(symbol, "buy", exec_price, strategy_name)

        log.info(
            "BUY %s @ $%.4f val=$%.0f (%.1f%% of eq=$%.0f) [%s] sac=%s cm=%.2f stop=$%.4f tp=$%.4f",
            symbol, exec_price, trade_value, prepared.trade_pct * 100, get_cash(),
            strategy_name,
            f"{sac_fraction:.3f}" if sac_fraction is not None else "FALLBACK",
            prepared.confidence_multiplier, stop_price, tp_price
        )
        log_trade(trade_rec)
        return

    if action == "sell":
        pos = prepared.pos
        pos_shares = prepared.pos_shares
        avg_cost = prepared.avg_cost
        _journaled = prepared.journaled

        if not fill.executed:
            if _journaled:
                update_order(_coid, state="FAILED", note=fill.note)
            trade_rec["reason"] = f"execution_{fill.status.lower()}"
            log.critical(
                "SELL %s DID NOT EXECUTE (%s: %s) — position stays open for retry",
                symbol, fill.status, fill.note,
            )
            log_trade(trade_rec)
            return

        exec_price = fill.price
        shares     = fill.qty                      # actual, possibly partial
        trade_rec["exec_price"] = exec_price
        proceeds  = shares * exec_price
        cost      = shares * avg_cost
        pnl_gross = proceeds - cost
        hold_candles = pos.get("candle_count", 1)
        hh = _hold_hours_from_candles(hold_candles)
        fee_total, net_pnl = net_realized_pnl(pnl_gross, cost, proceeds, hh)

        # Fix 1 — DYNAMIC MARGIN REFUND (long side): already read cash fresh
        # here in the original code — unaffected by the C4 split.
        returned_margin_long = cost
        set_cash(get_cash() + returned_margin_long + net_pnl)
        set_portfolio_stat("realised_pnl", float(get_portfolio_stat("realised_pnl", "0.0")) + pnl_gross)
        set_portfolio_stat(
            "realised_pnl_net",
            float(get_portfolio_stat("realised_pnl_net", "0.0")) + net_pnl,
        )
        if shares >= pos_shares * (1.0 - 1e-9):
            close_position(symbol)
        else:
            reduce_position(symbol, shares)
            log.warning(
                "SELL %s PARTIAL fill %.8f of %.8f — %.8f remains open",
                symbol, shares, pos_shares, pos_shares - shares,
            )
        if _journaled:
            update_order(_coid, state=fill.status, filled_qty=shares,
                         avg_fill_price=exec_price,
                         exchange_order_id=fill.exchange_order_id, note=fill.note)
        set_portfolio_stat("total_trades", int(get_portfolio_stat("total_trades", "0")) + 1)

        stored_entry_state = get_entry_state(symbol)
        entry_state_arr = np.array(stored_entry_state, dtype=np.float32) if stored_entry_state is not None else pre_state

        new_cash = get_cash()
        exit_state = brain.compute_sac_state(symbol, candles, new_cash, 0.0, new_cash)

        shaped_reward = _shaped_reward_v2(pnl_gross, hold_candles, cost)
        if 0.0 < net_pnl < MICRO_WIN_USD:
            shaped_reward -= MICRO_WIN_REWARD_PENALTY

        brain.reward(
            strategy_name=pos.get("strategy", strategy_name), pnl=shaped_reward, regime=regime,
            state=entry_state_arr,
            action=sac_fraction if sac_fraction is not None else 0.0,
            next_state=exit_state, trade_value=cost, side="long",
        )
        _persist_adaptive_edge_profiles()

        _avg_cost_s  = float(pos.get("avg_cost", 0.0))
        _mfe_price_s = float(pos.get("mfe_price") or _avg_cost_s)
        _mae_price_s = float(pos.get("mae_price") or _avg_cost_s)
        _max_unreal  = round((_mfe_price_s - _avg_cost_s) * shares, 4)
        _min_unreal  = round((_mae_price_s - _avg_cost_s) * shares, 4)

        trade_rec.update({
            "status":              "filled",
            "shares":              round(float(shares or 0.0), 8),
            "trade_value":         round(float(cost or 0.0), 2),
            "proceeds":            round(float(proceeds or 0.0), 2),
            "pnl":                 round(float(net_pnl or 0.0), 2),
            "gross_pnl":           round(float(pnl_gross or 0.0), 2),
            "fee_total":           round(float(fee_total or 0.0), 6),
            "net_pnl":             round(float(net_pnl or 0.0), 2),
            "max_unrealized_pnl":  _max_unreal,
            "min_unrealized_pnl":  _min_unreal,
            **_exit_journal_fields(pos, False, trade_rec),
        })

        alert_sniper_shot(
            symbol, f"sell (Net: ${net_pnl:.2f})", exec_price, pos.get("strategy", strategy_name)
        )

        log.info(
            "SELL %s @ $%.4f gross=$%+.2f net=$%+.2f fees=$%.4f shaped=$%+.4f [%s]",
            symbol, exec_price, pnl_gross, net_pnl, fee_total, shaped_reward, pos.get("strategy", "?")
        )
        log_trade(trade_rec)

        try:
            log_rl_experience(
                symbol=symbol,
                state=entry_state_arr.tolist() if entry_state_arr is not None else [0.0] * 13,
                action=float(sac_fraction) if sac_fraction is not None else 0.0,
                reward=float(shaped_reward),
                next_state=exit_state.tolist() if exit_state is not None else [0.0] * 13,
                done=True,
            )
        except Exception as _rl_exc:
            log.warning("log_rl_experience failed for SELL %s: %s", symbol, _rl_exc)
        return

    if action == "short":
        if not fill.executed:
            update_order(_coid, state=fill.status if fill.status in
                         ("REJECTED", "FAILED") else "FAILED", note=fill.note)
            trade_rec["reason"] = f"execution_{fill.status.lower()}"
            log.warning("SHORT %s not executed (%s: %s)", symbol, fill.status, fill.note)
            log_trade(trade_rec)
            return

        stop_price, tp_price = prepared.stop_price, prepared.tp_price
        if abs(fill.price - exec_price) > exec_price * 1e-9:
            stop_price, tp_price = brain.get_stop_take(
                fill.price, candles, prepared.is_yolo, side="short", regime=regime,
                confidence_mult=prepared.confidence_multiplier,
            )
            stop_price, tp_price = apply_dynamic_rr(
                entry=fill.price, stop=stop_price, take=tp_price,
                atr=prepared.atr_now, side="short",
            )
        exec_price = fill.price
        shares = fill.qty
        trade_value = shares * exec_price
        margin_reserved = trade_value * SHORT_MARGIN_PCT
        trade_rec["exec_price"] = exec_price
        if prepared.entry_features is not None and exec_price > 0:
            # Final levels as booked (re-derived above if the fill moved), so
            # the exit classifier can tell a moved stop from the original.
            prepared.entry_features.update({
                "initial_stop": stop_price,
                "initial_tp": tp_price,
                "stop_pct": round((stop_price - exec_price) / exec_price * 100.0, 4),
                "tp_pct": round((exec_price - tp_price) / exec_price * 100.0, 4),
            })

        # Fix #22: open_short() before set_cash(), and guarded. open_short()
        # raises ValueError instead of silently overwriting an already-active
        # short. Calling it before set_cash() means a rejection here never
        # leaves cash debited with nothing booked against it.
        try:
            open_short(
                symbol, shares, exec_price, strategy_name,
                stop_price, tp_price, margin_reserved, on_fire,
                entry_state=pre_state.tolist() if pre_state is not None else None,
                entry_features=prepared.entry_features,
            )
        except ValueError as _dup_short_exc:
            trade_rec["reason"] = "short_already_open"
            log.critical(
                "⚠ SHORT %s FILLED but BLOCKED from booking — %s — "
                "no cash moved, manual review required",
                symbol, _dup_short_exc,
            )
            update_order(
                _coid, state=fill.status, filled_qty=shares,
                avg_fill_price=exec_price, exchange_order_id=fill.exchange_order_id,
                note=f"BLOCKED: {_dup_short_exc}",
            )
            log_trade(trade_rec)
            return

        # Fix C4: fresh cash read at commit time (see BUY above).
        cash = get_cash()
        if cash < margin_reserved:
            log.warning(
                "SHORT %s committing with cash=$%.2f < margin=$%.2f — the "
                "fill already executed and cannot be undone; other symbols' "
                "trades likely consumed cash during this order's round-trip",
                symbol, cash, margin_reserved,
            )
        set_cash(cash - margin_reserved)
        update_order(_coid, state=fill.status, filled_qty=shares,
                     avg_fill_price=exec_price,
                     exchange_order_id=fill.exchange_order_id, note=fill.note)
        set_portfolio_stat("total_trades", int(
            get_portfolio_stat("total_trades", "0")) + 1)

        trade_rec.update({
            "status":      "filled",
            "shares":      round(float(shares or 0.0), 8),
            "trade_value": round(float(trade_value or 0.0), 2),
            "margin":      round(float(margin_reserved or 0.0), 2),
            "stop_price":  stop_price,
            "tp_price":    tp_price,
            "allocated_equity_pct": round(100.0 * float(trade_value or 0.0) / max(total_equity, 1e-9), 4),
            "entry_features": (
                json.dumps(prepared.entry_features, default=str, sort_keys=True)
                if prepared.entry_features is not None else None
            ),
        })
        alert_sniper_shot(symbol, "short", exec_price, strategy_name)
        log.info(
            "SHORT %s @ $%.4f  val=$%.0f (%.1f%% of eq=$%.0f) (margin=$%.0f) [%s]  sac=%s  cm=%.2f  stop=$%.4f  tp=$%.4f",
            symbol, exec_price, trade_value, prepared.trade_pct * 100, total_equity,
            margin_reserved, strategy_name,
            f"{sac_fraction:.3f}" if sac_fraction is not None else "FALLBACK",
            prepared.confidence_multiplier, stop_price, tp_price,
        )
        log_trade(trade_rec)
        return

    if action == "cover":
        pos = prepared.pos
        pos_shares = prepared.pos_shares
        avg_cost = prepared.avg_cost
        margin_res_total = prepared.margin_res_total
        _journaled = prepared.journaled

        if not fill.executed:
            if _journaled:
                update_order(_coid, state="FAILED", note=fill.note)
            trade_rec["reason"] = f"execution_{fill.status.lower()}"
            log.critical(
                "COVER %s DID NOT EXECUTE (%s: %s) — position stays open for retry",
                symbol, fill.status, fill.note,
            )
            log_trade(trade_rec)
            return

        exec_price = fill.price
        shares     = fill.qty                      # actual, possibly partial
        trade_rec["exec_price"] = exec_price
        entry_cost = shares * avg_cost
        cover_cost = shares * exec_price
        pnl_gross  = entry_cost - cover_cost
        _cover_frac = min(1.0, shares / pos_shares) if pos_shares > 0 else 1.0
        margin_res = margin_res_total * _cover_frac
        hold_candles_c = pos.get("candle_count", 1)
        hh = _hold_hours_from_candles(hold_candles_c)
        fee_total, net_pnl = net_realized_pnl(pnl_gross, entry_cost, cover_cost, hh)

        # Fix 1 — DYNAMIC MARGIN REFUND (short side): already read cash fresh
        # here in the original code — unaffected by the C4 split.
        set_cash(get_cash() + margin_res + net_pnl)
        set_portfolio_stat(
            "realised_pnl", float(get_portfolio_stat("realised_pnl", "0.0")) + pnl_gross
        )
        set_portfolio_stat(
            "realised_pnl_net",
            float(get_portfolio_stat("realised_pnl_net", "0.0")) + net_pnl,
        )
        if shares >= pos_shares * (1.0 - 1e-9):
            close_short(symbol)
        else:
            reduce_position(symbol, shares, margin_released=margin_res)
            log.warning(
                "COVER %s PARTIAL fill %.8f of %.8f — %.8f remains open",
                symbol, shares, pos_shares, pos_shares - shares,
            )
        if _journaled:
            update_order(_coid, state=fill.status, filled_qty=shares,
                         avg_fill_price=exec_price,
                         exchange_order_id=fill.exchange_order_id, note=fill.note)
        # Fix (Wave 3): cover previously fell through without incrementing
        # total_trades, unlike buy/sell/short -- undercounting the lifetime
        # trade stat for every closed short.
        set_portfolio_stat("total_trades", int(get_portfolio_stat("total_trades", "0")) + 1)
        alert_sniper_shot(
            symbol, f"cover (Net: ${net_pnl:.2f})", exec_price, pos.get("strategy", strategy_name)
        )

        stored_entry_state = get_entry_state(symbol)
        entry_state_arr = (
            np.array(stored_entry_state, dtype=np.float32)
            if stored_entry_state is not None else pre_state
        )
        new_cash = get_cash()
        exit_state = brain.compute_sac_state(symbol, candles, new_cash, 0.0, new_cash)

        hold_candles = pos.get("candle_count", 1)
        shaped_reward = _shaped_reward_v2(pnl_gross, hold_candles, entry_cost)
        if 0.0 < net_pnl < MICRO_WIN_USD:
            shaped_reward -= MICRO_WIN_REWARD_PENALTY

        brain.reward(
            strategy_name=pos.get("strategy", strategy_name), pnl=shaped_reward, regime=regime,
            state=entry_state_arr,
            action=sac_fraction if sac_fraction is not None else 0.0,
            next_state=exit_state, trade_value=entry_cost, side="short",
        )
        _persist_adaptive_edge_profiles()

        # For shorts: mfe_price is the LOWEST low seen (favourable); mae_price
        # is the HIGHEST high seen (adverse squeeze). Convert to USD PnL sign.
        _avg_cost_c  = float(pos.get("avg_cost", 0.0))
        _mfe_price_c = float(pos.get("mfe_price") or _avg_cost_c)
        _mae_price_c = float(pos.get("mae_price") or _avg_cost_c)
        _max_unreal  = round((_avg_cost_c - _mfe_price_c) * shares, 4)  # short profit
        _min_unreal  = round((_avg_cost_c - _mae_price_c) * shares, 4)  # short loss

        trade_rec.update({
            "status":              "filled",
            "shares":              round(float(shares or 0.0), 8),
            "trade_value":         round(float(entry_cost or 0.0), 2),
            "proceeds":            round(float(cover_cost or 0.0), 2),
            "pnl":                 round(float(net_pnl or 0.0), 2),
            "gross_pnl":           round(float(pnl_gross or 0.0), 2),
            "fee_total":           round(float(fee_total or 0.0), 6),
            "net_pnl":             round(float(net_pnl or 0.0), 2),
            "max_unrealized_pnl":  _max_unreal,
            "min_unrealized_pnl":  _min_unreal,
            **_exit_journal_fields(pos, True, trade_rec),
        })
        log.info(
            "COVER %s @ $%.4f  gross=$%+.2f net=$%+.2f fees=$%.4f shaped=$%+.4f  [%s]",
            symbol, exec_price, pnl_gross, net_pnl, fee_total, shaped_reward, pos.get("strategy", "?")
        )
        log_trade(trade_rec)

        try:
            log_rl_experience(
                symbol=symbol,
                state=entry_state_arr.tolist() if entry_state_arr is not None else [0.0] * 13,
                action=float(sac_fraction) if sac_fraction is not None else 0.0,
                reward=float(shaped_reward),
                next_state=exit_state.tolist() if exit_state is not None else [0.0] * 13,
                done=True,
            )
        except Exception as _rl_exc:
            log.warning("log_rl_experience failed for COVER %s: %s", symbol, _rl_exc)
        return


def _execute_trade(
    ensemble:      dict,
    strategy_name: str,
    candles:       list[dict],
    sac_fraction:  Optional[float],
    pre_state:     np.ndarray,
    total_equity:  float,
    conviction_mult: Optional[float] = None,
) -> None:
    """
    Synchronous convenience wrapper preserving the original single-call
    interface (used directly by tests/test_executor.py, and by any caller
    that doesn't need the async lock-scope split). Async callers use
    _prepare_trade / _exec_adapter.execute / _commit_trade directly so the
    exchange call can run with the global lock released (Fix C4) — see
    _tick_exit_check, _evaluate_and_trade, and exit_monitor.
    """
    prepared = _prepare_trade(
        ensemble, strategy_name, candles, sac_fraction, pre_state,
        total_equity, conviction_mult,
    )
    if prepared is None:
        return
    fill = _exec_adapter.execute(prepared.order_intent)
    _commit_trade(prepared, fill)


# ── Periodic exit monitor ─────────────────────────────────────────────────────

async def exit_monitor(loop: asyncio.AbstractEventLoop) -> None:
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=CHECK_EVERY_SECS)
            return
        except asyncio.TimeoutError:
            pass

        positions = get_all_positions()
        cash_m = get_cash()
        te_m = _compute_total_equity(cash_m, positions)
        try:
            async with aiohttp.ClientSession() as _s:
                await _refresh_margin_health(_s, cash_m, te_m)
        except Exception as exc:
            log.warning("margin refresh: %s", exc)

        for pos in positions:
            sym = pos["symbol"]

            # ── FIX 2: ZOMBIE SHELL GUARD ─────────────────────────────────────
            # _seed_portfolio() inserts placeholder rows with shares=0 for every
            # symbol at first boot.  If any of those rows slip through the
            # get_all_positions() WHERE filter (float precision, race condition,
            # or schema migration edge case), they must be nuked here before any
            # math runs on them — bad shares values cause division-by-zero and
            # phantom PnL that corrupts the cash balance.
            _pos_shares = float(pos.get("shares") or 0.0)
            if _pos_shares <= 0.0:
                log.warning(
                    "ZOMBIE SHELL detected for %s (shares=%s, margin=%s) — hard-deleting row",
                    sym, pos.get("shares"), pos.get("margin_reserved"),
                )
                try:
                    delete_position(sym)
                except Exception as _z_exc:
                    log.error("Failed to delete zombie position %s: %s", sym, _z_exc)
                continue
            # ── END ZOMBIE GUARD ──────────────────────────────────────────────

            candles = _candles_cache.get(sym)
            if not candles:
                continue

            _apply_ml_time_decay(sym, pos, candles)

            current_price = candles[-1]["close"]
            count = increment_candle_count(sym)
            stop = pos.get("stop_price", 0)
            tp = pos.get("tp_price",   float("inf"))

            reason:   Optional[str] = None
            is_short = pos.get("side", "long") == "short"

            # ── MFE / MAE RATCHET (every monitor cycle, ~30 s) ────────────────
            # Uses the last full candle's high/low rather than tick close, so
            # intra-bar excursions are captured even if no exit fires this cycle.
            try:
                _last_c = candles[-1]
                update_mfe_mae(
                    sym,
                    float(_last_c.get("high", current_price)),
                    float(_last_c.get("low",  current_price)),
                    is_short,
                )
            except Exception as _mfe_exc:
                log.debug("MFE/MAE update skipped for %s: %s", sym, _mfe_exc)
            # ── END MFE / MAE RATCHET ─────────────────────────────────────────

            # ── SURVIVAL KILL-SWITCH (Upgrade 1) ──────────────────────────────
            # Evaluated first — bypasses every downstream gate including the fee
            # gate.  _exit_in_flight and _trade_lock are still acquired below to
            # preserve DB atomicity; we just don't skip on a hard-loss position.
            _open_pnl_m = _hard_stop_open_pnl(pos, current_price)
            if _open_pnl_m < HARD_STOP_LOSS_USD:
                reason = (
                    f"hard_stop_survival "
                    f"(open_pnl=${_open_pnl_m:.2f} < floor=${HARD_STOP_LOSS_USD:.2f})"
                )
                log.critical(
                    "⚠ SURVIVAL KILL-SWITCH (monitor) %s — open_pnl=$%.2f < $%.2f — "
                    "bypassing all gates",
                    sym, _open_pnl_m, HARD_STOP_LOSS_USD,
                )
            # ── END SURVIVAL KILL-SWITCH ──────────────────────────────────────

            if not reason:
                if not is_short:
                    if stop > 0 and current_price <= stop:
                        reason = f"stop_loss (${current_price:,.4f} <= ${stop:,.4f})"
                    elif tp < float("inf") and current_price >= tp:
                        reason = f"take_profit (${current_price:,.4f} >= ${tp:,.4f})"
                else:
                    if stop > 0 and current_price >= stop:
                        reason = f"stop_loss_short (${current_price:,.4f} >= ${stop:,.4f})"
                    elif tp < float("inf") and current_price <= tp:
                        reason = f"take_profit_short (${current_price:,.4f} <= ${tp:,.4f})"

                # ── TIME-FREEZE FIX (Upgrade 2) ───────────────────────────────
                # candle_count freezes when the WS drops and no bars close.
                # Use wall-clock time from opened_ts as a second independent timer.
                # Only closes losing positions — winners are left to trail/TP.
                if not reason:
                    _opened_ts_raw = pos.get("opened_ts") or ""
                    _open_secs = 0.0
                    if _opened_ts_raw:
                        try:
                            _opened_dt = datetime.fromisoformat(
                                _opened_ts_raw.replace("Z", "+00:00")
                            )
                            _open_secs = time.time() - _opened_dt.timestamp()
                        except Exception:
                            _open_secs = 0.0
                    if _open_secs >= MAX_HOLD_OPEN_SECONDS and _open_pnl_m < 0:
                        reason = (
                            f"time_hold_exit "
                            f"({_open_secs / 3600:.1f}h open, in_red=${_open_pnl_m:.2f})"
                        )
                        log.warning(
                            "TIME-FREEZE EXIT %s — held %.1fh, open PnL=$%.2f < 0",
                            sym, _open_secs / 3600, _open_pnl_m,
                        )
                # ── END TIME-FREEZE FIX ───────────────────────────────────────

                if not reason and ENABLE_HARD_MAX_HOLD_EXIT and count >= MAX_HOLD_CANDLES:
                    reason = f"max_hold_time ({count} candles)"

            # ── FEE-BLEED GATE (monitor path) ─────────────────────────────────
            # Take-profit exits are held until they would book a positive net
            # (exit slippage + round-trip fee). Stops of every kind always fire
            # — see the tick-path gate for why trailed stops are not gated.
            _fee_gate_applies = reason and not any(
                k in reason for k in (
                    "hard_stop_survival", "time_hold_exit", "max_hold_time",
                )
            )
            if _fee_gate_applies:
                _avg_cost_m = float(pos.get("avg_cost", 0.0))
                _shares_m   = float(pos.get("shares", 0.0))
                if "take_profit" in reason:
                    if not _profit_clears_fees(_avg_cost_m, _shares_m, current_price, is_short):
                        log.info(
                            "FEE GATE blocked %s for %s — net after %.2f%% fees + %.2f%% exit slippage would be <= 0",
                            reason, sym, FEE_GATE_ROUND_TRIP * 100, SLIPPAGE_PCT * 100,
                        )
                        reason = None
            # ── END FEE-BLEED GATE ────────────────────────────────────────────

            if reason:
                log.info("EXIT %s - %s", sym, reason)
                fake_ensemble = {
                    "signal":             "cover" if is_short else "sell",
                    "exit_reason":        _classify_exit_reason(
                        reason, float(pos.get("avg_cost") or 0.0), float(stop or 0.0), is_short,
                        _initial_stop(pos)),
                    "exit_detail":        reason,
                    "symbol":             sym,
                    "price":              current_price,
                    "regime":             brain.current_regime,
                    "on_fire":            False,
                    "position_size_mult": 1.0,
                    "buy_weight":         0.0,
                    "sell_weight":        0.0,
                    "breakdown":          [],
                    "time":               _now(),
                }
                # Fix #5: strict commit ordering + cash lock (matches tick exit / evaluate).
                # Fix #19: one bad exit must never take down the whole 24/7 process.
                # exit_monitor is awaited directly inside main()'s asyncio.gather (not
                # via create_task like the tick/evaluate paths), so an uncaught exception
                # here previously propagated all the way out and killed every symbol's
                # risk monitoring, not just this one.
                # Fix C4: _trade_lock(sym) outermost, _strict_execution_lock released
                # around the exchange call — see _tick_exit_check for the full rationale.
                try:
                    prepared = None
                    async with _trade_lock(sym):
                        async with _strict_execution_lock:
                            cash = get_cash()
                            open_pos = get_all_positions()
                            total_eq = _compute_total_equity(cash, open_pos)
                            prepared = await loop.run_in_executor(
                                None, _prepare_trade, fake_ensemble,
                                pos.get("strategy", "EXIT"), candles,
                                # sac_fraction=None -> 2% fallback (exit sizing irrelevant)
                                None,
                                np.zeros(13, dtype=np.float32),
                                total_eq,
                                None,
                            )
                        if prepared is not None:
                            fill = await loop.run_in_executor(
                                None, _exec_adapter.execute, prepared.order_intent,
                            )
                            async with _strict_execution_lock:
                                async with _cash_lock:
                                    await loop.run_in_executor(None, _commit_trade, prepared, fill)
                except Exception:
                    log.error(
                        "exit_monitor: trade execution failed for %s (%s) — "
                        "position left open, will retry next cycle",
                        sym, reason, exc_info=True,
                    )
                    continue

        # Fix #6: use shared helper for equity snapshot
        cash = get_cash()
        all_pos = get_all_positions()
        eq = _compute_total_equity(cash, all_pos)
        record_equity(round(eq, 2))
        brain.check_circuit_breaker(eq)


# ── Startup ───────────────────────────────────────────────────────────────────

def _reconcile_orders_on_boot() -> None:
    """
    V4 restart protection: resolve journal rows left non-terminal by a crash.

    Paper mode: a non-terminal row means the process died between claiming the
    intent and committing the fill — nothing was booked, so the order is
    marked ORPHANED (never executed).
    Testnet mode: the exchange is queried by client id; a fill that exists on
    the exchange but not in our books is logged CRITICAL for manual review —
    deliberately NOT auto-booked, since blind booking could double-count.
    """
    stuck = get_open_orders()
    if not stuck:
        log.info("Order journal clean — no non-terminal orders to reconcile")
        return
    log.warning("Reconciling %d non-terminal order(s) from previous run", len(stuck))
    for row in stuck:
        coid, sym = row["client_order_id"], row["symbol"]
        try:
            if hasattr(_exec_adapter, "reconcile_with_symbol"):
                fill = _exec_adapter.reconcile_with_symbol(sym, coid)
            else:
                fill = _exec_adapter.reconcile(coid)
            if fill is None:
                update_order(coid, state="ORPHANED", note="reconcile: status unknown")
                log.warning("Order %s (%s) status unknown — marked ORPHANED", coid, sym)
            elif fill.executed:
                update_order(coid, state=fill.status, filled_qty=fill.qty,
                             avg_fill_price=fill.price,
                             note="reconcile: executed on exchange but UNBOOKED — review")
                log.critical(
                    "⚠ RECONCILE: order %s (%s) EXECUTED on exchange (qty=%.8f @ %.8f) "
                    "but is not in our books — manual review required",
                    coid, sym, fill.qty, fill.price,
                )
            else:
                update_order(coid, state="ORPHANED",
                             note=f"reconcile: not executed ({fill.note})")
                log.info("Order %s (%s) never executed — marked ORPHANED", coid, sym)
        except Exception as exc:
            log.error("Reconcile failed for %s (%s): %s — left for next boot", coid, sym, exc)


def _report_cash_invariant() -> None:
    """
    V4 restart protection: verify the books still balance after a restart.

    Invariant: cash + Σ(long cost basis) + Σ(short margin reserved)
               == STARTING_CASH + realised_pnl_net
    Report-only — drift is surfaced loudly, never auto-'healed' (heal_equity.py
    exists for deliberate, backed-up repair).
    """
    try:
        cash = get_cash()
        deployed_cost = 0.0
        for p in get_all_positions():
            if p.get("side", "long") == "short":
                deployed_cost += float(p.get("margin_reserved") or 0.0)
            else:
                deployed_cost += float(p.get("shares") or 0.0) * float(p.get("avg_cost") or 0.0)
        realised_net = float(get_portfolio_stat("realised_pnl_net", "0.0") or 0.0)
        expected = STARTING_CASH + realised_net - deployed_cost
        drift = cash - expected
        msg = (
            f"Cash invariant: cash=${cash:,.2f} deployed=${deployed_cost:,.2f} "
            f"realised_net=${realised_net:,.2f} -> expected cash=${expected:,.2f} "
            f"drift=${drift:+.2f}"
        )
        if abs(drift) > 1.00:
            log.warning("⚠ %s — books drifted (heal_equity.py --confirm after review)", msg)
        else:
            log.info("✅ %s", msg)
    except Exception as exc:
        log.error("Cash invariant check failed: %s", exc)


async def _load_history_for_all_symbols() -> dict[str, list[dict]]:
    """REST-fetch full candle history for every symbol into DB + cache."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_candle_history(session, sym) for sym in SYMBOLS]
        results = await asyncio.gather(*tasks)

    all_candles: dict[str, list[dict]] = {}
    for sym, candles in zip(SYMBOLS, results):
        if candles:
            upsert_candles_bulk(sym, candles)
            _candles_cache[sym] = candles
            all_candles[sym] = candles
    return all_candles


async def startup() -> None:
    log.info("Startup - fetching %d candles per symbol", CANDLE_LIMIT)
    log.info("Execution mode: %s", _exec_adapter.name)
    _reconcile_orders_on_boot()
    _report_cash_invariant()
    restored = _restore_adaptive_edge_profiles()
    if restored > 0:
        log.info("Adaptive edge profiles restored: %d buckets", restored)

    # Fix #21: restore the circuit breaker's peak-equity high-water mark so a
    # restart (crash, deploy, OOM) cannot silently re-arm MAX_DRAWDOWN_PCT from
    # whatever equity happens to exist at boot. Defaults to boot-time equity
    # when nothing was ever persisted (fresh install / wiped DB).
    _boot_cash = get_cash()
    _boot_equity = _compute_total_equity(_boot_cash, get_all_positions())
    _restored_peak = brain.restore_peak_equity(default=_boot_equity)
    log.info(
        "Circuit breaker peak-equity restored: $%.2f (boot equity $%.2f)",
        _restored_peak, _boot_equity,
    )

    all_candles = await _load_history_for_all_symbols()
    log.info("Historical candles loaded for %d symbols", len(all_candles))

    loop = asyncio.get_event_loop()
    trained = await loop.run_in_executor(_executor, initial_train, all_candles)
    if trained:
        log.info("Initial XGBoost model trained on startup history")

    for sym, candles in all_candles.items():
        try:
            prob = await loop.run_in_executor(_executor, predict_signal, candles)
            brain.update_ml_prob(sym, prob)
        except Exception as exc:
            log.warning("Initial ML inference failed for %s: %s", sym, exc)


# ── Housekeeping ──────────────────────────────────────────────────────────────
async def housekeeping_loop() -> None:
    """
    Runs every 60 seconds.  Persists:
      • Adaptive edge profiles  → brain_state table  (Fix 2 route)
      • Regime / circuit-breaker → portfolio table   (dashboard reads these)

    Each sub-task is wrapped individually so one failure cannot silently abort
    the others and cannot kill the loop.  Confirmed in asyncio.gather at startup.
    """
    # V4: honour the shutdown event (previously `while True` — asyncio.gather
    # could never finish, so every systemd stop timed out into SIGKILL).
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=60)
            return
        except asyncio.TimeoutError:
            pass

        # ── Edge profiles (brain long-term memory) ────────────────────────────
        try:
            brain.save()   # no-op in current Brain stub; safe to call
            _persist_adaptive_edge_profiles()
        except Exception as _hk_edge_exc:
            log.error("❌ Housekeeping — edge profile save FAILED: %s", _hk_edge_exc)

        # ── Regime / circuit-breaker state for dashboard ──────────────────────
        try:
            set_portfolio_stat("current_regime", brain.current_regime or "ranging")
            set_portfolio_stat("circuit_open", "1" if brain.circuit_open else "0")
        except Exception as _hk_reg_exc:
            log.error("❌ Housekeeping — regime persist FAILED: %s", _hk_reg_exc)

        # ── Confirmation log (only emitted when both succeed) ─────────────────
        n_buckets = len(getattr(brain, "_edge_profiles", {}))
        log.info(
            "🧹 Housekeeping complete — %d edge buckets → brain_state, regime=%s cb=%s",
            n_buckets,
            getattr(brain, "current_regime", "?"),
            "OPEN" if getattr(brain, "circuit_open", False) else "clear",
        )

# ── Entry point ───────────────────────────────────────────────────────────────
# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    global _executor

    # V4: the dashboard is managed by its own systemd unit (quant-dashboard).
    # The old subprocess.Popen(dashboard.py) here always died on the port-8000
    # conflict with that unit and left a zombie child on every boot.
    log.info("Quant Bot V4 — execution adapter (%s) + order journal — booting",
             _exec_adapter.name)
    init_db()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig.name)

    _executor = ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS)

    try:
        await startup()
        # Fix #20: return_exceptions=True is a second line of defense behind Fix #19
        # — even an exception nobody anticipated in one of these three coroutines can
        # no longer cancel its siblings and crash the process. It CAN still leave that
        # one subsystem dead for the rest of the run, so a captured exception is
        # always logged CRITICAL below rather than silently discarded.
        _results = await asyncio.gather(
            websocket_listener(loop),
            exit_monitor(loop),
            housekeeping_loop(),
            return_exceptions=True,
        )
        for _name, _result in zip(
            ("websocket_listener", "exit_monitor", "housekeeping_loop"), _results
        ):
            if isinstance(_result, BaseException):
                log.critical(
                    "⚠ %s terminated with an unhandled exception — this subsystem "
                    "is DEAD until the next restart: %r",
                    _name, _result, exc_info=_result,
                )
    finally:
        log.info("Shutting down ProcessPoolExecutor...")
        _executor.shutdown(wait=True, cancel_futures=True)
        log.info("Bot stopped cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
