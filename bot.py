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
"""

from __future__ import annotations
import math

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
    DECAY_SL_TIGHTEN_STRENGTH, DECAY_TP_PULL_STRENGTH,
    BE_TRIGGER_ATR_MULT, BE_MIN_PROFIT_ATR, BE_MIN_PROFIT_PCT,
    FEE_GATE_ROUND_TRIP, MICRO_WIN_REWARD_PENALTY, MICRO_WIN_USD,
    MIN_EXPECTED_TP_NET_USD,
)
from rl_agent import compute_position_fraction
from ml_engine import predict_signal, incremental_train, initial_train
from brain import Brain
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
    open_short_count, update_stop_price, update_tp_price, update_mfe_mae,
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


def _apply_ml_time_decay(sym: str, pos: dict, candles: list[dict]) -> None:
    """
    Exponential time-decay on stagnant risk targets: λ = 1 − e^(−t/τ).

    Pulls TP toward entry (sooner monetisation) and tightens stop — variance
    collapses when edge does not materialise (optional stopping / real-options view).
    """
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


def _wilder_atr(candles: list[dict], window: int = 14) -> float:
    """
    Fix #9: proper Wilder's smoothed ATR.
    The previous simple-mean approximation produced a different volatility
    estimate than brain.py, causing trailing/BE stops to use inconsistent
    distances from entry stops/TPs.

    Seeds from the mean of the first `window` true ranges, then applies
    Wilder's exponential smoothing: ATR = (prev * (n-1) + TR) / n
    """
    if len(candles) < 2:
        return 0.0

    seed_end = min(window + 1, len(candles))
    true_ranges = [
        max(
            candles[i]["high"] - candles[i]["low"],
            abs(candles[i]["high"] - candles[i - 1]["close"]),
            abs(candles[i]["low"] - candles[i - 1]["close"]),
        )
        for i in range(1, seed_end)
    ]
    atr = sum(true_ranges) / len(true_ranges)

    for i in range(seed_end, len(candles)):
        tr = max(
            candles[i]["high"] - candles[i]["low"],
            abs(candles[i]["high"] - candles[i - 1]["close"]),
            abs(candles[i]["low"] - candles[i - 1]["close"]),
        )
        atr = (atr * (window - 1) + tr) / window

    return atr


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
    url = _ws_url()
    backoff = 1.0

    while not _shutdown_event.is_set():
        log.info("Connecting WebSocket -> %s", url)
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1.0
                log.info("WebSocket connected")
                async for raw_msg in ws:
                    if _shutdown_event.is_set():
                        return
                    await _handle_ws_message(raw_msg, loop)

        except Exception as exc:
            if _shutdown_event.is_set():
                return
            log.error("WebSocket error: %s - reconnecting in %.1fs", exc, backoff)
            try:
                await asyncio.wait_for(_shutdown_event.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 60.0)


# ── Fee-gate helpers (pure, no I/O, safe to call from async context) ─────────

def _is_profitable_stop(avg_cost: float, stop: float, is_short: bool) -> bool:
    """
    Returns True if the current stop price represents a locked-profit level
    (i.e. it has been trailed to break-even or better), meaning the fee-bleed
    gate must be applied before allowing it to fire.

    Long : stop >= avg_cost  (BE or above)
    Short: stop <= avg_cost  (BE or below)

    Protective loss stops (stop < entry for longs, stop > entry for shorts)
    return False — they bypass the fee gate unconditionally.
    """
    if avg_cost <= 0 or stop <= 0:
        return False
    return (stop >= avg_cost) if not is_short else (stop <= avg_cost)


def _profit_clears_fees(
    avg_cost: float, shares: float, exit_price: float, is_short: bool
) -> bool:
    """
    Returns True if the unrealised profit at exit_price strictly exceeds the
    Binance round-trip taker fee (FEE_GATE_ROUND_TRIP × position notional).

    This gates take_profit and profitable trailing-stop exits so the bot cannot
    systematically close positions for less than the cost of two fills.
    Hard stop_loss exits must bypass this function entirely.

    Returns True (allow) on degenerate inputs so a missing field never blocks
    a legitimate exit.
    """
    if avg_cost <= 0 or shares <= 0:
        return True
    notional = shares * avg_cost
    profit = (
        shares * (avg_cost - exit_price) if is_short
        else shares * (exit_price - avg_cost)
    )
    return profit > notional * FEE_GATE_ROUND_TRIP


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

    if not is_short:
        if tp > 0 and candle_high >= tp:
            reason = f"take_profit_tick (high=${candle_high:,.4f} >= tp=${tp:,.4f})"
            exit_price = tp
        elif stop > 0 and candle_low <= stop:
            reason = f"stop_loss_tick (low=${candle_low:,.4f} <= stop=${stop:,.4f})"
            exit_price = stop
    else:
        if tp > 0 and candle_low <= tp:
            reason = f"take_profit_tick_short (low=${candle_low:,.4f} <= tp=${tp:,.4f})"
            exit_price = tp
        elif stop > 0 and candle_high >= stop:
            reason = f"stop_loss_tick_short (high=${candle_high:,.4f} >= stop=${stop:,.4f})"
            exit_price = stop

    # Survival override: no standard SL/TP triggered yet but hard floor crossed.
    # Force an immediate market exit at the current close price.
    if reason is None and _is_survival:
        reason     = f"hard_stop_survival_tick (pnl=${_open_pnl:.2f})"
        exit_price = _mark_close

    if reason is None:
        return

    # ── FEE-BLEED GATE (tick path) ────────────────────────────────────────────
    # Block take_profit and profitable trailing-stop (BE or better) exits whose
    # unrealised profit does not yet clear the Binance round-trip taker fee
    # (≈ 0.12% of notional).  Hard stop_loss exits that are still below entry
    # bypass this gate unconditionally — never delay a loss-protection exit.
    # Survival kill-switch exits ALSO bypass: a $-15 loss must close regardless.
    _shares_tick = float(pos.get("shares", 0.0))
    if not _is_survival and ("take_profit" in reason or _is_profitable_stop(avg_cost, stop, is_short)):
        if not _profit_clears_fees(avg_cost, _shares_tick, exit_price, is_short):
            log.info(
                "FEE GATE blocked %s for %s — profit does not clear %.2f%% round-trip fees",
                reason, sym, FEE_GATE_ROUND_TRIP * 100,
            )
            return
    # ── END FEE-BLEED GATE ────────────────────────────────────────────────────

    try:
        _exit_in_flight.add(sym)
        log.info("TICK EXIT %s -- %s", sym, reason)

        candles = _candles_cache.get(sym, [])
        fake_ensemble = {
            "signal":             "cover" if is_short else "sell",
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
        async with _strict_execution_lock:
            async with _trade_lock(sym):
                async with _cash_lock:
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

                    await loop.run_in_executor(
                        None,
                        _execute_trade,
                        fake_ensemble,
                        pos.get("strategy", "TICK_EXIT"),
                        candles,
                        None,   # sac_fraction=None -> conservative 2% fallback
                        np.zeros(13, dtype=np.float32),
                        total_eq,
                        None,
                    )
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

    upsert_candle(sym, candle)

    set_portfolio_stat(f"last_price_{sym}", candle["close"])
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
    if direction == "short" and open_short_count() >= SHORT_MAX_OPEN:
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
    if direction in ("buy", "short"):
        async with _strict_execution_lock:
            open_positions = get_all_positions()
            if direction == "buy" and open_position_count() >= MAX_OPEN_POSITIONS:
                return
            if direction == "short" and open_short_count() >= SHORT_MAX_OPEN:
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

            async with _trade_lock(symbol):
                async with _cash_lock:
                    await loop.run_in_executor(
                        None,
                        _execute_trade,
                        ensemble, strategy_name, candles, sac_fraction, state_vec, total_equity,
                        cm_authoritative,
                    )
        return

    async with _strict_execution_lock:
        async with _trade_lock(symbol):
            async with _cash_lock:
                await loop.run_in_executor(
                    None,
                    _execute_trade,
                    ensemble, strategy_name, candles, sac_fraction, state_vec, total_equity,
                    None,
                )


# ── Core trade executor ───────────────────────────────────────────────────────

def _execute_trade(
    ensemble:      dict,
    strategy_name: str,
    candles:       list[dict],
    # Fix #8: None = SAC failed, use 2% fallback
    sac_fraction:  Optional[float],
    pre_state:     np.ndarray,
    total_equity:  float,            # Fix #7: passed in, not re-fetched from DB
    conviction_mult: Optional[float] = None,
) -> None:
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
            return
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
            return

    # ── Shared SAC-driven sizing (used by BUY and SHORT) ──────────────────────
    def _calc_trade_value() -> tuple[float, float, float]:
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
            return 0.0, 0.0, confidence_multiplier

        return trade_value, trade_pct, confidence_multiplier
    # ── End sizing helper ─────────────────────────────────────────────────────

    if action == "buy":
        cash = get_cash()
        trade_value, trade_pct, confidence_multiplier = _calc_trade_value()

        if cash < trade_value or trade_value < 1.0:
            trade_rec["reason"] = "insufficient_funds"
            log_trade(trade_rec)
            return

        shares = trade_value / exec_price
        is_yolo = strategy_name == "YOLO_FIRE"
        stop_price, tp_price = brain.get_stop_take(
            exec_price, candles, is_yolo, side="long", regime=regime,
            confidence_mult=confidence_multiplier,
        )
        _atr_now = _wilder_atr(candles) if candles else 0.0
        stop_price, tp_price = apply_dynamic_rr(
            entry=exec_price, stop=stop_price, take=tp_price,
            atr=_atr_now, side="long",
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
            return

        set_cash(cash - trade_value)
        open_position(
            symbol, shares, exec_price, strategy_name,
            stop_price, tp_price, on_fire,
            entry_state=pre_state.tolist() if pre_state is not None else None,
        )
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
            symbol, exec_price, trade_value, trade_pct * 100, get_cash(),
            strategy_name,
            f"{sac_fraction:.3f}" if sac_fraction is not None else "FALLBACK",
            confidence_multiplier, stop_price, tp_price
        )
        log_trade(trade_rec)
        return

    elif action == "sell":
        pos = next((p for p in get_all_positions() if p["symbol"] == symbol), None)
        if not pos or pos.get("shares", 0) <= 0:
            trade_rec["reason"] = "no_position"
            log_trade(trade_rec)
            return

        shares    = float(pos["shares"])
        avg_cost  = float(pos["avg_cost"])
        proceeds  = shares * exec_price
        cost      = shares * avg_cost
        pnl_gross = proceeds - cost
        hold_candles = pos.get("candle_count", 1)
        hh = _hold_hours_from_candles(hold_candles)
        fee_total, net_pnl = net_realized_pnl(pnl_gross, cost, proceeds, hh)

        # Fix 1 — DYNAMIC MARGIN REFUND (long side):
        # Return the original capital outlay (shares × avg_cost) plus the net
        # realised PnL (after fees).  Using net_pnl instead of pnl_gross means
        # the fee_total is actually deducted from the live cash balance so the
        # equity calculation stays accurate.
        #   correct: cash += returned_margin + net_pnl
        #   broken:  cash += proceeds            ← overstates by fee_total
        returned_margin_long = cost  # original capital spent to open the long
        set_cash(get_cash() + returned_margin_long + net_pnl)
        set_portfolio_stat("realised_pnl", float(get_portfolio_stat("realised_pnl", "0.0")) + pnl_gross)
        set_portfolio_stat(
            "realised_pnl_net",
            float(get_portfolio_stat("realised_pnl_net", "0.0")) + net_pnl,
        )
        close_position(symbol)
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

        # ── MFE / MAE → USD conversion for the trade record ──────────────
        _avg_cost_s  = float(pos.get("avg_cost", 0.0))
        _mfe_price_s = float(pos.get("mfe_price") or _avg_cost_s)
        _mae_price_s = float(pos.get("mae_price") or _avg_cost_s)
        _max_unreal  = round((_mfe_price_s - _avg_cost_s) * shares, 4)
        _min_unreal  = round((_mae_price_s - _avg_cost_s) * shares, 4)

        trade_rec.update({
            "status":              "filled",
            "shares":              round(float(shares or 0.0), 8),
            # trade_value for a sell = the original cost basis (capital returned)
            "trade_value":         round(float(cost or 0.0), 2),
            "proceeds":            round(float(proceeds or 0.0), 2),
            "pnl":                 round(float(net_pnl or 0.0), 2),
            "gross_pnl":           round(float(pnl_gross or 0.0), 2),
            "fee_total":           round(float(fee_total or 0.0), 6),
            "net_pnl":             round(float(net_pnl or 0.0), 2),
            "max_unrealized_pnl":  _max_unreal,
            "min_unrealized_pnl":  _min_unreal,
        })

        alert_sniper_shot(
            symbol, f"sell (Net: ${net_pnl:.2f})", exec_price, pos.get("strategy", strategy_name)
        )

        log.info(
            "SELL %s @ $%.4f gross=$%+.2f net=$%+.2f fees=$%.4f shaped=$%+.4f [%s]",
            symbol, exec_price, pnl_gross, net_pnl, fee_total, shaped_reward, pos.get("strategy", "?")
        )
        log_trade(trade_rec)

        # ── SAC ONLINE LEARNING PIPELINE (Upgrade 4) ──────────────────────────
        # Persist the complete (s, a, r, s', done) transition so offline_trainer.py
        # can pull real live experience from rl_experience on the next training run.
        # entry_state_arr = SAC state at position open (loaded from positions.entry_state)
        # action          = SAC fraction used at entry; None on crashes/exits → 0.0
        # reward          = shaped net PnL (already includes candle opportunity cost)
        # exit_state      = SAC state computed immediately after position closes
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
        # ── END SAC PIPELINE ──────────────────────────────────────────────────
        return

    elif action == "short":
        cash = get_cash()
        trade_value, trade_pct, confidence_multiplier = _calc_trade_value()
        margin_reserved = trade_value * SHORT_MARGIN_PCT

        if cash < margin_reserved or margin_reserved < 1.0:
            trade_rec["reason"] = "insufficient_margin"
            log_trade(trade_rec)
            return

        shares = trade_value / exec_price
        is_yolo = strategy_name == "YOLO_FIRE"
        stop_price, tp_price = brain.get_stop_take(
            exec_price, candles, is_yolo, side="short", regime=regime,
            confidence_mult=confidence_multiplier,
        )
        _atr_now = _wilder_atr(candles) if candles else 0.0
        stop_price, tp_price = apply_dynamic_rr(
            entry=exec_price, stop=stop_price, take=tp_price,
            atr=_atr_now, side="short",
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
            return

        set_cash(cash - margin_reserved)
        open_short(
            symbol, shares, exec_price, strategy_name,
            stop_price, tp_price, margin_reserved, on_fire,
            entry_state=pre_state.tolist() if pre_state is not None else None,
        )
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
        })
        alert_sniper_shot(symbol, "short", exec_price, strategy_name)
        log.info(
            "SHORT %s @ $%.4f  val=$%.0f (%.1f%% of eq=$%.0f) (margin=$%.0f) [%s]  sac=%s  cm=%.2f  stop=$%.4f  tp=$%.4f",
            symbol, exec_price, trade_value, trade_pct * 100, total_equity,
            margin_reserved, strategy_name,
            f"{sac_fraction:.3f}" if sac_fraction is not None else "FALLBACK",
            confidence_multiplier, stop_price, tp_price,
        )
        log_trade(trade_rec)
        return

    elif action == "cover":
        pos = next((p for p in get_all_positions()
                    if p["symbol"] == symbol), None)
        if not pos or pos.get("shares", 0) <= 0:
            trade_rec["reason"] = "no_position"
            log_trade(trade_rec)
            return

        shares     = float(pos["shares"])
        avg_cost   = float(pos["avg_cost"])
        entry_cost = shares * avg_cost
        cover_cost = shares * exec_price
        pnl_gross  = entry_cost - cover_cost
        margin_res = float(pos.get("margin_reserved") or 0.0)
        hold_candles_c = pos.get("candle_count", 1)
        hh = _hold_hours_from_candles(hold_candles_c)
        fee_total, net_pnl = net_realized_pnl(pnl_gross, entry_cost, cover_cost, hh)

        # Fix 1 — DYNAMIC MARGIN REFUND (short side):
        # Return the collateral (margin_reserved) plus net_pnl (after fees).
        # Using pnl_gross here overstated cash by fee_total on every cover.
        #   correct: cash += margin_reserved + net_pnl
        #   broken:  cash += margin_reserved + pnl_gross
        set_cash(get_cash() + margin_res + net_pnl)
        set_portfolio_stat(
            "realised_pnl", float(get_portfolio_stat("realised_pnl", "0.0")) + pnl_gross
        )
        set_portfolio_stat(
            "realised_pnl_net",
            float(get_portfolio_stat("realised_pnl_net", "0.0")) + net_pnl,
        )
        close_short(symbol)
        alert_sniper_shot(
            symbol, f"cover (Net: ${net_pnl:.2f})", exec_price, pos.get("strategy", strategy_name)
        )

    stored_entry_state = get_entry_state(symbol)
    entry_state_arr: Optional[np.ndarray] = (
        np.array(stored_entry_state, dtype=np.float32)
        if stored_entry_state is not None else pre_state
    )
    new_cash = get_cash()
    exit_state = brain.compute_sac_state(
        symbol, candles, new_cash, 0.0, new_cash)

    if "pos" not in locals() or pos is None:
        pos = {}
    if "pnl_gross" not in locals():
        pnl_gross = locals().get("pnl", 0.0)
    if "entry_cost" not in locals():
        entry_cost = 0.0
    if "fee_total" not in locals():
        fee_total = 0.0
    if "net_pnl" not in locals():
        net_pnl = pnl_gross
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

    # ── MFE / MAE → USD conversion for the cover trade record ────────────
    # For shorts: mfe_price is the LOWEST low seen (favourable); mae_price is
    # the HIGHEST high seen (adverse squeeze).  Convert to USD PnL sign.
    _avg_cost_c  = float(pos.get("avg_cost", 0.0))
    _mfe_price_c = float(pos.get("mfe_price") or _avg_cost_c)
    _mae_price_c = float(pos.get("mae_price") or _avg_cost_c)
    _max_unreal  = round((_avg_cost_c - _mfe_price_c) * shares, 4)  # short profit
    _min_unreal  = round((_avg_cost_c - _mae_price_c) * shares, 4)  # short loss

    _cover_cost_val = float(locals().get("cover_cost") or locals().get("sell_value") or 0.0)
    _entry_cost_val = float(locals().get("entry_cost") or 0.0)
    trade_rec.update({
        "status":              "filled",
        "shares":              round(float(shares or 0.0), 8),
        # trade_value for a cover = the entry notional (margin collateral basis)
        "trade_value":         round(_entry_cost_val, 2),
        "proceeds":            round(_cover_cost_val, 2),
        "pnl":                 round(float(net_pnl or 0.0), 2),
        "gross_pnl":           round(float(pnl_gross or 0.0), 2),
        "fee_total":           round(float(fee_total or 0.0), 6),
        "net_pnl":             round(float(net_pnl or 0.0), 2),
        "max_unrealized_pnl":  _max_unreal,
        "min_unrealized_pnl":  _min_unreal,
    })
    log.info(
        "COVER %s @ $%.4f  gross=$%+.2f net=$%+.2f fees=$%.4f shaped=$%+.4f  [%s]",
        symbol, exec_price, pnl_gross, net_pnl, fee_total, shaped_reward, pos.get("strategy", "?")
    )

    log_trade(trade_rec)

    # ── SAC ONLINE LEARNING PIPELINE (Upgrade 4) ──────────────────────────────
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
    # ── END SAC PIPELINE ──────────────────────────────────────────────────────


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
            # Take-profit and profitable trailing-stop exits are held until
            # unrealised profit clears round-trip fees.
            # Bypassed for: hard_stop_survival, time_hold_exit, max_hold_time,
            # and all raw stop_loss exits (loss-protection must never be delayed).
            _fee_gate_applies = reason and not any(
                k in reason for k in (
                    "hard_stop_survival", "time_hold_exit", "max_hold_time",
                )
            )
            if _fee_gate_applies:
                _avg_cost_m = float(pos.get("avg_cost", 0.0))
                _shares_m   = float(pos.get("shares", 0.0))
                if "take_profit" in reason or _is_profitable_stop(_avg_cost_m, stop, is_short):
                    if not _profit_clears_fees(_avg_cost_m, _shares_m, current_price, is_short):
                        log.info(
                            "FEE GATE blocked %s for %s — profit does not clear %.2f%% round-trip fees",
                            reason, sym, FEE_GATE_ROUND_TRIP * 100,
                        )
                        reason = None
            # ── END FEE-BLEED GATE ────────────────────────────────────────────

            if reason:
                log.info("EXIT %s - %s", sym, reason)
                fake_ensemble = {
                    "signal":             "cover" if is_short else "sell",
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
                async with _strict_execution_lock:
                    async with _trade_lock(sym):
                        async with _cash_lock:
                            cash = get_cash()
                            open_pos = get_all_positions()
                            total_eq = _compute_total_equity(cash, open_pos)
                            await loop.run_in_executor(
                                None, _execute_trade, fake_ensemble,
                                pos.get("strategy", "EXIT"), candles,
                                # sac_fraction=None -> 2% fallback (exit sizing irrelevant)
                                None,
                                np.zeros(13, dtype=np.float32),
                                total_eq,
                                None,
                            )

        # Fix #6: use shared helper for equity snapshot
        cash = get_cash()
        all_pos = get_all_positions()
        eq = _compute_total_equity(cash, all_pos)
        record_equity(round(eq, 2))
        brain.check_circuit_breaker(eq)


# ── Startup ───────────────────────────────────────────────────────────────────

async def startup() -> None:
    log.info("Startup - fetching %d candles per symbol", CANDLE_LIMIT)
    restored = _restore_adaptive_edge_profiles()
    if restored > 0:
        log.info("Adaptive edge profiles restored: %d buckets", restored)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_candle_history(session, sym) for sym in SYMBOLS]
        results = await asyncio.gather(*tasks)

    all_candles: dict[str, list[dict]] = {}
    for sym, candles in zip(SYMBOLS, results):
        if candles:
            upsert_candles_bulk(sym, candles)
            _candles_cache[sym] = candles
            all_candles[sym] = candles

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
    while True:
        await asyncio.sleep(60)

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
    import subprocess
    import sys

    log.info("🚀 Booting UI Command Center...")
    dash_proc = subprocess.Popen([sys.executable, "/home/admin/trading_bot/dashboard.py"])

    log.info("Quant Bot V2 — risk engine + net accounting — booting")
    init_db()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig.name)

    _executor = ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS)

    try:
        await startup()
        await asyncio.gather(
            websocket_listener(loop),
            exit_monitor(loop),
            housekeeping_loop(),
        )
    finally:
        log.info("Shutting down Dashboard server...")
        dash_proc.terminate()
        log.info("Shutting down ProcessPoolExecutor...")
        _executor.shutdown(wait=True, cancel_futures=True)
        log.info("Bot stopped cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
