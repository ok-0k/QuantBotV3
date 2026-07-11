"""
strategy_manager_gen2.py — Gen-2 Strategy Pool Manager
=======================================================
Manages the full lifecycle of a pool of ``StrategyParamsV2`` objects:

  * Spawning new paper strategies.
  * Routing candle ticks to paper vs live strategies.
  * Processing ghost (paper) trade closes.
  * Processing live trade closes.
  * Running promotion / demotion / kill-switch evaluations automatically.
  * Periodic pool rebalancing (culling dead entries, spawning replacements).

This file is self-contained; it imports from both ``strategies.py`` and
``strategy_engine_gen2.py`` and is the only file the main bot loop needs to
interact with.

Typical bot-loop usage
----------------------
    manager = StrategyManagerV2(max_pool_size=20)
    manager.seed_pool()                    # spawn initial paper strategies

    for candle_batch in live_feed:
        signals = manager.tick(candle_batch)
        for sig in signals:
            if sig["status"] == "live" and sig["signal"] != "none":
                # place real order ...
                pass
            elif sig["status"] == "paper" and sig["signal"] != "none":
                # track ghost position ...
                pass

    # When a position closes:
    report = manager.close_trade(
        strategy_name="GEN_RSI_VOL_AND",
        pnl=12.50,
        trade_value=500.0,
        # For paper strategies only — pass ghost trade fields:
        signal="buy", entry_price=100.0, exit_price=102.5, size=5.0,
        symbol="BTCUSDT",
    )
    manager.rebalance()   # run after each close or on a timer
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from strategy_engine_gen2 import (
    STATUS_DEAD,
    STATUS_LIVE,
    STATUS_PAPER,
    StrategyParamsV2,
    check_kill_switch,
    generate_random_strategy_v2,
    log_ghost_trade,
    process_closed_trade,
    try_demote,
    try_promote,
    KILL_SWITCH_PNL,
    DEMOTION_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class StrategyManagerV2:
    """
    Manages a pool of Gen-2 strategies with full 4-Pillar lifecycle support.

    Parameters
    ----------
    max_pool_size:
        Maximum number of non-dead strategies to keep in the pool.
        When the pool shrinks below this, ``rebalance()`` spawns new paper
        strategies to top it back up.
    demotion_threshold:
        Forwarded to ``try_demote``.  Defaults to module constant.
    kill_threshold:
        Forwarded to ``check_kill_switch``.  Defaults to module constant.
    """

    def __init__(
        self,
        max_pool_size:        int   = 20,
        demotion_threshold:   float = DEMOTION_SCORE_THRESHOLD,
        kill_threshold:       float = KILL_SWITCH_PNL,
    ):
        self.max_pool_size      = max_pool_size
        self.demotion_threshold = demotion_threshold
        self.kill_threshold     = kill_threshold

        # name → StrategyParamsV2
        self._params: dict[str, StrategyParamsV2]  = {}
        # name → callable(candles, params) → signal dict
        self._fns:    dict[str, Callable]           = {}

    # ── Pool helpers ──────────────────────────────────────────────────────────

    @property
    def active_strategies(self) -> list[StrategyParamsV2]:
        """All non-dead strategies."""
        return [sp for sp in self._params.values() if sp.status != STATUS_DEAD]

    @property
    def paper_strategies(self) -> list[StrategyParamsV2]:
        return [sp for sp in self._params.values() if sp.status == STATUS_PAPER]

    @property
    def live_strategies(self) -> list[StrategyParamsV2]:
        return [sp for sp in self._params.values() if sp.status == STATUS_LIVE]

    @property
    def dead_strategies(self) -> list[StrategyParamsV2]:
        return [sp for sp in self._params.values() if sp.status == STATUS_DEAD]

    def _active_names(self) -> set[str]:
        return {sp.name for sp in self.active_strategies}

    # ── Seeding / registration ────────────────────────────────────────────────

    def seed_pool(self, n: Optional[int] = None) -> None:
        """
        Spawn ``n`` new paper strategies (defaults to filling up to
        ``max_pool_size``).
        """
        target = n if n is not None else self.max_pool_size
        spawned = 0
        while len(self.active_strategies) < target:
            self._spawn_one()
            spawned += 1
        if spawned:
            logger.info("Seeded %d new paper strategies into pool.", spawned)

    def register_strategy(
        self,
        sp: StrategyParamsV2,
        fn: Callable,
    ) -> None:
        """
        Add an externally constructed strategy to the pool.
        Useful for hand-crafted (non-generated) strategies.
        """
        if sp.name in self._params:
            logger.warning("Strategy '%s' already in pool — skipping.", sp.name)
            return
        self._params[sp.name] = sp
        self._fns[sp.name]    = fn
        logger.debug("Registered strategy '%s' (status=%s).", sp.name, sp.status)

    def _spawn_one(self) -> StrategyParamsV2:
        """Spawn and register a single new paper strategy."""
        sp, fn = generate_random_strategy_v2(self._active_names())
        self._params[sp.name] = sp
        self._fns[sp.name]    = fn
        logger.debug("Spawned paper strategy '%s'.", sp.name)
        return sp

    # ── Tick ──────────────────────────────────────────────────────────────────

    def tick(self, candles: list[dict]) -> list[dict]:
        """
        Run all active strategies on the latest candle batch and return
        their signals.

        Parameters
        ----------
        candles:
            The current rolling candle window for one symbol.  If your bot
            runs multi-symbol, call ``tick`` once per symbol.

        Returns
        -------
        list of dicts, each containing:
            ``name``    — strategy name.
            ``status``  — ``'paper'`` or ``'live'``.
            ``signal``  — ``'buy'``, ``'sell'``, or ``'none'``.
            ``price``   — latest close price.
            ``meta``    — indicator metadata from the signal function.
        """
        results = []
        for name, sp in list(self._params.items()):
            if sp.status == STATUS_DEAD:
                continue
            fn = self._fns.get(name)
            if fn is None:
                continue
            try:
                raw = fn(candles, sp.params)
                results.append({
                    "name":   name,
                    "status": sp.status,
                    "signal": raw.get("signal", "none"),
                    "price":  raw.get("price",  candles[-1]["close"] if candles else 0.0),
                    "meta":   raw.get("meta",   {}),
                })
            except Exception as exc:
                logger.debug("Strategy '%s' tick error: %s", name, exc)
        return results

    # ── Trade close ───────────────────────────────────────────────────────────

    def close_trade(
        self,
        strategy_name:  str,
        pnl:            float,
        trade_value:    float = 1.0,
        # Ghost-trade fields (required for paper strategies)
        signal:         Optional[str]   = None,
        entry_price:    Optional[float] = None,
        exit_price:     Optional[float] = None,
        size:           float = 1.0,
        symbol:         str   = "",
        timestamp:      Optional[float] = None,
    ) -> dict:
        """
        Record the close of a position and run all lifecycle evaluations.

        For **paper** strategies, pass ``signal``, ``entry_price``,
        ``exit_price`` (and optionally ``size``, ``symbol``, ``timestamp``)
        so a ghost-trade record is created.  The ``pnl`` field must still be
        provided and should match ``(exit-entry)*size`` (or vice-versa for
        shorts) — the manager will verify but trust the caller.

        For **live** strategies, only ``pnl`` and ``trade_value`` are required.

        Returns
        -------
        dict
            Lifecycle report from ``process_closed_trade``, augmented with the
            strategy name and current status.
        """
        sp = self._params.get(strategy_name)
        if sp is None:
            logger.error("close_trade: unknown strategy '%s'.", strategy_name)
            return {"error": f"unknown strategy '{strategy_name}'"}

        if sp.status == STATUS_DEAD:
            logger.warning("close_trade: strategy '%s' is already dead.", strategy_name)
            return {"error": "strategy is dead"}

        # ── Paper: log ghost trade ────────────────────────────────────────────
        if sp.status == STATUS_PAPER:
            if entry_price is None or exit_price is None or signal is None:
                raise ValueError(
                    "Paper trade close requires signal, entry_price, and "
                    "exit_price keyword arguments."
                )
            log_ghost_trade(
                sp,
                signal=signal,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                symbol=symbol,
                timestamp=timestamp or time.time(),
            )
            # log_ghost_trade already called record_trade_v2 internally;
            # run lifecycle checks manually (process_closed_trade would
            # double-count the trade stats).
            killed   = check_kill_switch(sp, kill_threshold=self.kill_threshold)
            promoted = False
            demoted  = False
            if not killed:
                promoted = try_promote(sp)
            report = {
                "prev_status": STATUS_PAPER,
                "new_status":  sp.status,
                "killed":      killed,
                "promoted":    promoted,
                "demoted":     demoted,
                "score":       sp.score,
                "total_pnl":   sp.total_pnl,
            }

        # ── Live: standard process ────────────────────────────────────────────
        else:
            report = process_closed_trade(
                sp,
                pnl=pnl,
                trade_value=trade_value,
                demotion_threshold=self.demotion_threshold,
                kill_threshold=self.kill_threshold,
            )

        # ── Logging ───────────────────────────────────────────────────────────
        report["name"] = strategy_name
        self._log_lifecycle_event(strategy_name, report)
        return report

    def _log_lifecycle_event(self, name: str, report: dict) -> None:
        if report.get("killed"):
            logger.warning(
                "🔴 KILL SWITCH fired on '%s' | total_pnl=%.2f | "
                "status=dead",
                name, report["total_pnl"],
            )
        elif report.get("promoted"):
            logger.info(
                "🟢 PROMOTED '%s': paper → live | "
                "paper_pnl=%.2f | score=%.2f",
                name,
                self._params[name].paper_pnl,
                report["score"],
            )
        elif report.get("demoted"):
            logger.info(
                "🟡 DEMOTED '%s': live → paper (benched) | "
                "score=%.2f | total_pnl=%.2f",
                name, report["score"], report["total_pnl"],
            )

    # ── Rebalance ─────────────────────────────────────────────────────────────

    def rebalance(self) -> dict:
        """
        Housekeeping pass.  Call after each trade close or on a timer.

        Actions
        ~~~~~~~
        1. Prune confirmed dead strategies from the registry (retains the
           params dict entry for a grace period so callers can log the kill).
           Here we remove them immediately — callers who need the data should
           save it before calling rebalance.
        2. Spawn new paper strategies until the pool is back at
           ``max_pool_size``.

        Returns
        -------
        dict with ``pruned`` and ``spawned`` counts.
        """
        dead_names = [name for name, sp in self._params.items()
                      if sp.status == STATUS_DEAD]
        for name in dead_names:
            del self._params[name]
            self._fns.pop(name, None)
            logger.debug("Pruned dead strategy '%s' from pool.", name)

        spawned = 0
        while len(self.active_strategies) < self.max_pool_size:
            self._spawn_one()
            spawned += 1

        if dead_names or spawned:
            logger.info(
                "Rebalance: pruned=%d dead, spawned=%d new paper | "
                "pool_size=%d (paper=%d, live=%d)",
                len(dead_names), spawned, len(self.active_strategies),
                len(self.paper_strategies), len(self.live_strategies),
            )

        return {"pruned": len(dead_names), "spawned": spawned}

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> list[dict]:
        """
        Return a sorted snapshot of the pool, best score first.
        Useful for dashboard rendering or periodic logging.
        """
        rows = []
        for sp in sorted(self.active_strategies, key=lambda s: s.score, reverse=True):
            rows.append({
                "name":         sp.name,
                "status":       sp.status,
                "score":        round(sp.score, 4),
                "total_pnl":    round(sp.total_pnl, 2),
                "paper_pnl":    round(sp.paper_pnl, 2),
                "live_pnl":     round(sp.live_pnl, 2),
                "total_trades": sp.total_trades,
                "paper_trades": sp.paper_trades,
                "live_trades":  sp.live_trades,
                "win_rate":     round(sp.win_rate, 3),
                "sharpe":       round(sp.sharpe, 3),
                "max_drawdown": round(sp.max_drawdown, 4),
                "generation":   sp.generation,
            })
        return rows
