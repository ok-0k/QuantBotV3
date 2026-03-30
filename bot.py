"""
bot.py — Phase 1: Async WebSocket Trading Orchestrator.

Replaces the blocking REST + time.sleep() loop with a fully non-blocking
asyncio event loop driven by Binance server-push WebSocket messages.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  asyncio event loop (single OS thread)                      │
  │                                                             │
  │  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
  │  │ WS Listener  │   │ Exit Monitor │   │ ML Cron Task   │  │
  │  │ (coroutine)  │   │ (coroutine)  │   │ (coroutine)    │  │
  │  └──────┬───────┘   └──────────────┘   └────────────────┘  │
  │         │ CPU-bound work dispatched via run_in_executor     │
  │         ▼                                                   │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │  ProcessPoolExecutor (2 workers)                     │   │
  │  │  Worker 1: XGBoost predict / incremental_train       │   │
  │  │  Worker 2: SAC actor forward pass / feature compute  │   │
  │  └──────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────┘

SQLite (WAL mode) is the shared state store — the FastAPI dashboard
process reads from the same file concurrently without any locking.

Run: python bot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from config import (
    BINANCE_REST, BINANCE_WS_BASE, CANDLE_LIMIT, CHECK_EVERY_SECS,
    LOG_DIR, MAX_HOLD_CANDLES, MAX_OPEN_POSITIONS, MAX_TRADE_SIZE_PCT,
    ML_UPDATE_EVERY, PROCESS_POOL_WORKERS, SLIPPAGE_PCT, STARTING_CASH,
    SYMBOLS, INTERVAL, TRADE_SIZE_PCT, YOLO_TRADE_SIZE_PCT,
)
from db import (
    close_position, get_all_positions, get_candle_count, get_candles,
    get_cash, get_ml_prob, get_portfolio_stat, increment_candle_count,
    init_db, log_trade, open_position, open_position_count, record_equity,
    set_cash, set_portfolio_stat, upsert_candle, upsert_candles_bulk,
    save_ml_cache,
)
from brain import Brain
from ml_engine import predict_signal, incremental_train, initial_train
from rl_agent import compute_position_fraction
from features import build_sac_state

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING  (to tmpfs — zero SD card wear on Pi 5)
# ─────────────────────────────────────────────────────────────────────────────
CHECK_EVERY_SECS = 30   # exit monitor cadence

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

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
brain           = Brain()
_candles_cache: dict[str, list[dict]] = {}   # in-memory recent candles
_closed_candle_counts: dict[str, int]  = {s: 0 for s in SYMBOLS}
_executor: Optional[ProcessPoolExecutor] = None
_bot_running = True


# ─────────────────────────────────────────────────────────────────────────────
# BINANCE REST — startup history fetch
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_candle_history(session: aiohttp.ClientSession, symbol: str) -> list[dict]:
    """Fetch CANDLE_LIMIT historical candles via REST on startup."""
    url    = f"{BINANCE_REST}/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": CANDLE_LIMIT}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
            r.raise_for_status()
            raw = await r.json()
        candles = [
            {"time": c[0], "open": float(c[1]), "high": float(c[2]),
             "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
            for c in raw
        ]
        log.info("Fetched %d historical candles for %s", len(candles), symbol)
        return candles
    except Exception as exc:
        log.error("History fetch failed for %s: %s", symbol, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET STREAM LISTENER
# ─────────────────────────────────────────────────────────────────────────────

def _build_ws_url() -> str:
    """Build the combined multi-stream WebSocket URL for all symbols."""
    streams = "/".join(f"{s.lower()}@kline_{INTERVAL}" for s in SYMBOLS)
    return f"{BINANCE_WS_BASE}?streams={streams}"


async def websocket_listener(loop: asyncio.AbstractEventLoop) -> None:
    """
    Persistent WebSocket listener with exponential-backoff reconnection.

    The Binance server PUSHES kline updates — we never poll.
    Each incoming message triggers:
      1. Candle upsert to SQLite
      2. On closed candle: async ML inference job (ProcessPoolExecutor)
      3. On closed candle: ensemble signal evaluation
      4. If actionable signal: paper trade execution
    """
    url = _build_ws_url()
    backoff = 1.0

    while _bot_running:
        log.info("Connecting WebSocket → %s", url)
        try:
            import websockets  # type: ignore
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1.0  # reset on successful connection
                log.info("✅ WebSocket connected")
                async for raw_msg in ws:
                    if not _bot_running:
                        break
                    await _handle_ws_message(raw_msg, loop)

        except Exception as exc:
            log.error("WebSocket error: %s — reconnecting in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)   # cap at 60 seconds


async def _handle_ws_message(raw_msg: str, loop: asyncio.AbstractEventLoop) -> None:
    """Parse a kline WebSocket message and dispatch work."""
    try:
        msg = json.loads(raw_msg)
    except json.JSONDecodeError:
        return

    data  = msg.get("data", {})
    kline = data.get("k", {})
    sym   = data.get("s", "")

    if not kline or sym not in SYMBOLS:
        return

    candle = {
        "time":   int(kline["t"]),
        "open":   float(kline["o"]),
        "high":   float(kline["h"]),
        "low":    float(kline["l"]),
        "close":  float(kline["c"]),
        "volume": float(kline["v"]),
    }

    # Always upsert (partial candles update price; closed candles finalise)
    upsert_candle(sym, candle)

    # Update in-memory cache
    if sym not in _candles_cache:
        _candles_cache[sym] = get_candles(sym, CANDLE_LIMIT)
    else:
        # Replace or append latest candle
        cache = _candles_cache[sym]
        if cache and cache[-1]["time"] == candle["time"]:
            cache[-1] = candle
        else:
            cache.append(candle)
            if len(cache) > CANDLE_LIMIT:
                cache.pop(0)

    is_closed = bool(kline.get("x", False))
    if not is_closed:
        return   # skip processing until candle closes

    # ── CLOSED CANDLE — dispatch CPU-bound work to worker process ────────────
    _closed_candle_counts[sym] = _closed_candle_counts.get(sym, 0) + 1
    count = _closed_candle_counts[sym]

    candles = _candles_cache.get(sym, [])
    if len(candles) < 60:
        return

    # Run ML inference in worker process — never blocks the event loop
    ml_prob = await loop.run_in_executor(
        _executor, predict_signal, list(candles))
    brain.update_ml_prob(sym, ml_prob)
    save_ml_cache(sym, [], ml_prob)

    # Scheduled incremental model update
    if count % ML_UPDATE_EVERY == 0:
        log.info("⚙️  Scheduling incremental XGB update for %s", sym)
        asyncio.create_task(
            _run_incremental_train(loop, sym, list(candles)))

    # Evaluate ensemble + execute
    asyncio.create_task(_evaluate_and_trade(sym, candles, ml_prob, loop))


async def _run_incremental_train(loop: asyncio.AbstractEventLoop,
                                  sym: str, candles: list[dict]) -> None:
    """Offload incremental XGBoost training to a worker process."""
    try:
        result = await loop.run_in_executor(_executor, incremental_train, candles)
        if result:
            log.info("🎓 Incremental XGB update complete for %s", sym)
    except Exception as exc:
        log.error("Incremental train error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE EVALUATION & EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

async def _evaluate_and_trade(symbol: str, candles: list[dict],
                               ml_prob: float, loop: asyncio.AbstractEventLoop) -> None:
    """
    Full evaluation pipeline for one symbol on each closed candle.
    Runs within the event loop but dispatches RL inference to worker.
    """
    ensemble = brain.get_ensemble_signal(symbol, candles)

    if ensemble["signal"] not in ("buy", "sell"):
        return

    direction = ensemble["signal"]

    # ── CIRCUIT BREAKER ──────────────────────────────────────────────────────
    if direction == "buy" and brain.circuit_open:
        log.info("🔴 %s BUY blocked — circuit breaker open", symbol)
        return

    # ── MAX POSITIONS ────────────────────────────────────────────────────────
    if direction == "buy" and open_position_count() >= MAX_OPEN_POSITIONS:
        return

    # ── ANTI-CORRELATION ─────────────────────────────────────────────────────
    if direction == "buy":
        pos_list = get_all_positions()
        for pos in pos_list:
            other_sym = pos["symbol"]
            if other_sym == symbol:
                continue
            other_candles = _candles_cache.get(other_sym, [])
            if other_candles and brain.are_correlated(
                    [c["close"] for c in candles],
                    [c["close"] for c in other_candles]):
                log.info("   %s: BUY blocked — correlated with %s", symbol, other_sym)
                return

    # ── CANDLE VOLATILITY FILTER ─────────────────────────────────────────────
    last = candles[-1]
    if last["low"] > 0:
        candle_pct = (last["high"] - last["low"]) / last["low"]
        if candle_pct < 0.0005:
            return  # market too quiet for meaningful scalping

    # ── SAC POSITION SIZING ──────────────────────────────────────────────────
    cash         = get_cash()
    positions    = get_all_positions()
    unrealised   = sum(
        (candles[-1]["close"] - p["avg_cost"]) * p["shares"]
        for p in positions
        if p.get("symbol") in _candles_cache
    )
    current_eq   = cash + unrealised + sum(
        _candles_cache.get(p["symbol"], [{}])[-1].get("close", p["avg_cost"]) * p["shares"]
        for p in positions
    )

    state_vec    = brain.compute_sac_state(
        symbol, candles, cash, unrealised, current_eq)

    # SAC actor forward pass in worker process
    position_fraction = await loop.run_in_executor(
        _executor, compute_position_fraction, state_vec)

    # ── EXECUTE ───────────────────────────────────────────────────────────────
    strategy_name = _top_strategy(ensemble, direction)
    _execute_trade(ensemble, strategy_name, candles,
                   position_fraction, state_vec, loop)


def _execute_trade(ensemble: dict, strategy_name: str,
                   candles: list[dict], sac_fraction: float,
                   pre_state: "np.ndarray", loop: asyncio.AbstractEventLoop) -> None:
    """Paper trade execution with ATR stops and full audit trail."""
    action    = ensemble["signal"]
    symbol    = ensemble["symbol"]
    raw_price = ensemble["price"]
    regime    = ensemble.get("regime", "ranging")
    on_fire   = ensemble.get("on_fire", False)

    # Slippage model
    if action == "buy":
        exec_price = raw_price * (1 + SLIPPAGE_PCT)
    else:
        exec_price = raw_price * (1 - SLIPPAGE_PCT)

    trade_rec: dict = {
        "symbol": symbol, "action": action, "strategy": strategy_name,
        "regime": regime, "price": raw_price, "exec_price": exec_price,
        "on_fire": on_fire, "timestamp": _now(),
        "slippage": round(abs(exec_price - raw_price), 6),
        "status": "skipped",
    }

    if action == "buy":
        cash = get_cash()
        # SAC fraction modulated by ensemble conviction and on_fire
        base_pct    = YOLO_TRADE_SIZE_PCT if strategy_name == "YOLO_FIRE" else TRADE_SIZE_PCT
        regime_mult = {"trending_up": 1.2, "trending_down": 1.1,
                       "ranging": 0.85, "volatile": 0.55}.get(regime, 1.0)
        fire_mult   = ensemble.get("position_size_mult", 1.0)

        # Combine: SAC drives the size but is capped by hard ceiling
        trade_pct   = base_pct * regime_mult * fire_mult * max(0.5, sac_fraction * 2)
        trade_pct   = min(trade_pct, MAX_TRADE_SIZE_PCT)
        trade_value = cash * trade_pct

        if trade_value < 1.0:
            trade_rec["reason"] = "insufficient_cash"
            log_trade(trade_rec)
            return

        shares = trade_value / exec_price
        is_yolo = strategy_name == "YOLO_FIRE"
        stop_price, tp_price = brain.get_stop_take(exec_price, candles, is_yolo)

        set_cash(cash - trade_value)
        open_position(symbol, shares, exec_price, strategy_name,
                      stop_price, tp_price, on_fire)
        set_portfolio_stat("total_trades",
                           int(get_portfolio_stat("total_trades", "0")) + 1)

        trade_rec.update({
            "status": "filled", "shares": round(shares, 8),
            "trade_value": round(trade_value, 2),
            "stop_price": stop_price, "tp_price": tp_price,
        })
        log.info("📈 BUY  %s @ $%.4f  val=$%.0f  [%s]  sac=%.2f  stop=$%.4f  tp=$%.4f",
                 symbol, exec_price, trade_value, strategy_name,
                 sac_fraction, stop_price, tp_price)

    elif action == "sell":
        pos = next((p for p in get_all_positions() if p["symbol"] == symbol), None)
        if not pos or pos.get("shares", 0) <= 0:
            trade_rec["reason"] = "no_position"
            log_trade(trade_rec)
            return

        shares   = pos["shares"]
        proceeds = shares * exec_price
        cost     = shares * pos["avg_cost"]
        pnl      = proceeds - cost

        set_cash(get_cash() + proceeds)
        set_portfolio_stat(
            "realised_pnl",
            float(get_portfolio_stat("realised_pnl", "0.0")) + pnl)
        close_position(symbol)
        set_portfolio_stat("total_trades",
                           int(get_portfolio_stat("total_trades", "0")) + 1)

        # Feedback to brain for RL and strategy evolution
        next_state = brain.compute_sac_state(
            symbol, candles, get_cash(), 0.0, get_cash())
        brain.reward(
            pos.get("strategy", strategy_name), pnl, regime,
            state=pre_state, action=sac_fraction, next_state=next_state,
            trade_value=cost)

        trade_rec.update({
            "status": "filled", "shares": round(shares, 8),
            "proceeds": round(proceeds, 2), "pnl": round(pnl, 2),
        })
        log.info("📉 SELL %s @ $%.4f  PnL=$%+.2f  [%s]",
                 symbol, exec_price, pnl, pos.get("strategy", "?"))

    log_trade(trade_rec)


# ─────────────────────────────────────────────────────────────────────────────
# EXIT MONITOR  (stop loss / take profit / max hold time)
# ─────────────────────────────────────────────────────────────────────────────

async def exit_monitor(loop: asyncio.AbstractEventLoop) -> None:
    """
    Independent coroutine: checks open positions every CHECK_EVERY_SECS.
    Fires ATR-based stop losses, take profits, and max hold-time exits.
    """
    while _bot_running:
        await asyncio.sleep(CHECK_EVERY_SECS)
        positions = get_all_positions()
        for pos in positions:
            sym   = pos["symbol"]
            candles = _candles_cache.get(sym)
            if not candles:
                continue

            current_price = candles[-1]["close"]
            count         = increment_candle_count(sym)
            stop          = pos.get("stop_price", 0)
            tp            = pos.get("tp_price", float("inf"))

            reason: Optional[str] = None
            if stop > 0 and current_price <= stop:
                reason = f"stop_loss (${current_price:,.4f} ≤ ${stop:,.4f})"
            elif tp < float("inf") and current_price >= tp:
                reason = f"take_profit (${current_price:,.4f} ≥ ${tp:,.4f})"
            elif count >= MAX_HOLD_CANDLES:
                reason = f"max_hold_time ({count} candles)"

            if reason:
                log.info("⏏️  EXIT %s — %s", sym, reason)
                fake = {
                    "signal": "sell", "symbol": sym, "price": current_price,
                    "regime": brain.current_regime, "on_fire": False,
                    "position_size_mult": 1.0,
                    "buy_weight": 0.0, "sell_weight": 0.0,
                    "time": _now(), "breakdown": [],
                }
                _execute_trade(fake, pos.get("strategy", "EXIT"),
                               candles, 0.5, None, loop)

        # Record equity snapshot
        cash       = get_cash()
        all_pos    = get_all_positions()
        eq         = cash + sum(
            _candles_cache.get(p["symbol"], [{}])[-1].get("close", p["avg_cost"]) * p["shares"]
            for p in all_pos
        )
        record_equity(round(eq, 2))
        brain.check_circuit_breaker(eq)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — history fetch + initial ML training
# ─────────────────────────────────────────────────────────────────────────────

async def startup() -> None:
    """Fetch candle history and run initial ML training before WebSocket starts."""
    log.info("🚀 Startup — fetching %d candles per symbol", CANDLE_LIMIT)

    async with aiohttp.ClientSession() as session:
        tasks  = [fetch_candle_history(session, sym) for sym in SYMBOLS]
        results = await asyncio.gather(*tasks)

    all_candles: dict[str, list[dict]] = {}
    for sym, candles in zip(SYMBOLS, results):
        if candles:
            upsert_candles_bulk(sym, candles)
            _candles_cache[sym] = candles
            all_candles[sym]    = candles

    log.info("Historical candles loaded for %d symbols", len(all_candles))

    # Initial XGBoost training across all history (worker process)
    loop = asyncio.get_event_loop()
    trained = await loop.run_in_executor(_executor, initial_train, all_candles)
    if trained:
        log.info("✅ Initial XGBoost model trained on startup history")

    # Run initial ML inference
    for sym, candles in all_candles.items():
        prob = await loop.run_in_executor(_executor, predict_signal, candles)
        brain.update_ml_prob(sym, prob)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _top_strategy(ensemble: dict, direction: str) -> str:
    bd  = ensemble.get("breakdown", [])
    top = max((b for b in bd if b["signal"] == direction),
              key=lambda x: x["alloc"], default=None)
    return top["strategy"] if top else "ENSEMBLE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    global _executor

    log.info("⚡ Quant Bot v3 — Phase 1-4 architecture booting")

    # Initialise SQLite schema and seed portfolio
    init_db()

    # Start ProcessPoolExecutor: 2 workers to avoid thermal throttling on Pi 5
    _executor = ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS)

    # Fetch history + initial ML training
    await startup()

    loop = asyncio.get_event_loop()

    # Launch all coroutines concurrently
    await asyncio.gather(
        websocket_listener(loop),
        exit_monitor(loop),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot shut down gracefully")
        _bot_running = False
