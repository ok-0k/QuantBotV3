"""
strategy_engine_gen2.py — Gen-2 'Bulletproof Engine'
=====================================================
Drop-in replacement for the fitness, lifecycle, and data-model logic in
strategies.py.  Import this module alongside the existing strategies.py;
it does NOT touch indicators or signal functions — only scoring, promotion,
demotion, and kill-switch logic.

4-Pillar Architecture
---------------------
  Pillar 1 — 'Sniper' Fitness Function  : PnL-primary, expectancy-aware scoring.
  Pillar 2 — 'Incubator' Paper Tier      : All new GEN_ strategies start in paper mode.
  Pillar 3 — 'Bench'  (no deletes)       : Bad live strategies demote, not die.
  Pillar 4 — Hard Kill Switch            : Permanent delete at -$50 drawdown.

Usage
-----
    from strategies import (
        StrategyParams, ENTRY_FNS, FILTER_FNS, LOGIC_MODES,
        build_generated_strategy, _random_params,
    )
    from strategy_engine_gen2 import (
        StrategyParamsV2,
        calculate_sniper_score,
        record_trade_v2,
        try_promote,
        try_demote,
        check_kill_switch,
        generate_random_strategy_v2,
        log_ghost_trade,
    )
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Literal, Optional

# ── Re-export so callers only need to import from this module ──────────────────
# The indicator utilities and strategy functions stay in strategies.py.
# Only pull in what the engine references directly.
try:
    from strategies import (
        ENTRY_FNS,
        FILTER_FNS,
        LOGIC_MODES,
        _PARAM_POOLS,
        _random_params,
        build_generated_strategy,
    )
except ImportError:  # allow the file to be read standalone / in tests
    ENTRY_FNS = {}
    FILTER_FNS = {}
    LOGIC_MODES = ["AND", "GATE"]
    _PARAM_POOLS = {}

    def _random_params() -> dict:  # type: ignore[misc]
        return {}

    def build_generated_strategy(entry, filter_, logic, params, name):  # type: ignore[misc]
        return lambda c, p: {"signal": "none"}


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

#: A strategy in paper mode uses no real margin.
STATUS_PAPER: Literal["paper"] = "paper"
#: A strategy in live mode uses real margin.
STATUS_LIVE:  Literal["live"]  = "live"
#: Permanent death — kept briefly so the manager can log the event.
STATUS_DEAD:  Literal["dead"]  = "dead"

#: Minimum closed paper trades required before promotion is evaluated.
PROMOTION_MIN_TRADES: int   = 5
#: Paper net PnL must be strictly positive to earn promotion.
PROMOTION_MIN_PNL:    float = 0.0

#: Live score below this triggers a demotion back to paper.
DEMOTION_SCORE_THRESHOLD: float = -3.0

#: Hard kill — any strategy (paper or live) is permanently removed at this PnL.
KILL_SWITCH_PNL: float = -50.0

#: Minimum closed trades before the expectancy modifier is applied.
EXPECTANCY_MIN_TRADES: int = 3

#: Win-rate level that starts earning a positive expectancy bonus.
EXPECTANCY_WIN_RATE_THRESHOLD: float = 0.50

#: Maximum absolute value of the expectancy modifier (caps the bonus/penalty).
EXPECTANCY_MAX_MODIFIER: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 1 — SNIPER FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sniper_score(
    total_pnl: float,
    wins: int,
    total_trades: int,
) -> float:
    """
    Compute the Gen-2 'Sniper' fitness score.

    Design goals
    ~~~~~~~~~~~~
    * **Net PnL is king** — the base score is simply ``total_pnl``.
    * **Zero inactivity penalty** — time between trades is completely ignored.
      A strategy that fires once a week and prints money scores higher than a
      strategy that fires 200 times and loses money.
    * **Expectancy bonus** — applied only after ``EXPECTANCY_MIN_TRADES``
      closed trades so a single lucky win cannot inflate a new strategy's score.
      Modifier is proportional to how far win-rate deviates from the 50 % break-
      even line, capped at ``EXPECTANCY_MAX_MODIFIER``.

    Parameters
    ----------
    total_pnl:
        Sum of all realised PnL for this strategy (positive = profitable).
    wins:
        Number of winning closed trades.
    total_trades:
        Total number of closed trades (wins + losses).

    Returns
    -------
    float
        Composite fitness score.  Higher is better.  Negative means losing.
    """
    # ── Base: raw realised profit ────────────────────────────────────────────
    base = total_pnl

    # ── Expectancy modifier ──────────────────────────────────────────────────
    modifier = 0.0
    if total_trades >= EXPECTANCY_MIN_TRADES:
        win_rate = wins / total_trades
        # deviation from break-even (can be negative for bad win-rates)
        deviation = win_rate - EXPECTANCY_WIN_RATE_THRESHOLD
        # scale: ±0.50 deviation → ±EXPECTANCY_MAX_MODIFIER points
        raw_modifier = deviation * (EXPECTANCY_MAX_MODIFIER / 0.50)
        modifier = max(-EXPECTANCY_MAX_MODIFIER, min(EXPECTANCY_MAX_MODIFIER, raw_modifier))

    return base + modifier


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — STRATEGY PARAMS V2  (Incubator data model)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyParamsV2:
    """
    Gen-2 strategy parameter and performance container.

    Key additions over the v1 ``StrategyParams``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``status`` — ``'paper'`` | ``'live'`` | ``'dead'``
      All new strategies start in ``'paper'`` mode.
    * ``paper_trades`` / ``paper_pnl`` — tracked separately so the promotion
      gate uses *only* the paper incubation period, not lifetime PnL.
    * ``score`` is now computed by ``calculate_sniper_score`` — not an
      activity-decayed counter.
    * No time-decay anywhere in record_trade.

    Backwards-compatibility note
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The ``sharpe`` property and ``returns_buffer`` are retained from v1 so
    existing dashboard / logging code that reads those attributes does not
    break.  They are *not* used by the new scoring logic.
    """
    name:   str
    params: dict

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    status:     str = field(default=STATUS_PAPER)   # 'paper' | 'live' | 'dead'
    generation: int = field(default=0)

    # ── Lifetime performance (all trades, paper + live) ───────────────────────
    total_trades: int   = field(default=0)
    wins:         int   = field(default=0)
    losses:       int   = field(default=0)
    total_pnl:    float = field(default=0.0)

    # ── Paper-tier tracking (used only for promotion gate) ────────────────────
    paper_trades: int   = field(default=0)
    paper_pnl:    float = field(default=0.0)

    # ── Live-tier tracking (used for demotion gate) ───────────────────────────
    live_trades:  int   = field(default=0)
    live_pnl:     float = field(default=0.0)

    # ── Fitness (recomputed on every record_trade call) ───────────────────────
    score:        float = field(default=0.0)
    weight:       float = field(default=1.0)

    # ── Sharpe / returns buffer (retained for dashboard compat) ───────────────
    returns_buffer: list = field(default_factory=list)
    peak_score:     float = field(default=0.0)
    max_drawdown:   float = field(default=0.0)

    # ── DNA metadata ──────────────────────────────────────────────────────────
    is_generated: bool = field(default=False, init=False, repr=False)
    is_yolo:      bool = field(default=False, init=False, repr=False)
    blueprint:    dict = field(default_factory=dict, init=False, repr=False)

    # ── Ghost-trade log (paper mode only) ─────────────────────────────────────
    ghost_log:    list = field(default_factory=list, init=False, repr=False)

    # ─────────────────────────────────────────────────────────────────────────

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
        """Information-ratio Sharpe from recent trade returns (v1 compat)."""
        n = len(self.returns_buffer)
        if n < 5:
            return 0.0
        mean = sum(self.returns_buffer) / n
        var  = sum((r - mean) ** 2 for r in self.returns_buffer) / n
        if var < 1e-10:
            return 10.0 if mean > 0 else -10.0
        return mean / math.sqrt(var)

    @property
    def is_paper(self) -> bool:
        return self.status == STATUS_PAPER

    @property
    def is_live(self) -> bool:
        return self.status == STATUS_LIVE

    @property
    def is_dead(self) -> bool:
        return self.status == STATUS_DEAD

    def _refresh_score(self) -> None:
        """Recompute Sniper score from current lifetime stats."""
        self.score = calculate_sniper_score(
            total_pnl=self.total_pnl,
            wins=self.wins,
            total_trades=self.total_trades,
        )
        # Track peak / drawdown on the score itself (for dashboard gauges)
        if self.score > self.peak_score:
            self.peak_score = self.score
        drop = self.peak_score - self.score
        if drop > self.max_drawdown:
            self.max_drawdown = drop


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — record_trade_v2  (no activity decay)
# ─────────────────────────────────────────────────────────────────────────────

def record_trade_v2(
    sp: StrategyParamsV2,
    pnl: float,
    trade_value: float = 1.0,
) -> None:
    """
    Record a **closed** trade result on a StrategyParamsV2 object.

    Differences from v1 ``record_trade``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * No time-decay (``score *= 0.95`` is gone).
    * Score is recomputed from first principles via ``calculate_sniper_score``
      so it always reflects lifetime PnL + expectancy — never an arbitrary
      decayed counter.
    * Segregated paper / live counters are updated correctly by current status.

    Parameters
    ----------
    sp:
        The strategy whose stats are being updated.
    pnl:
        Realised profit/loss for this trade in dollars.
    trade_value:
        Notional value of the trade; used to normalise the returns buffer for
        Sharpe calculation.  Default 1.0 (dimensionless normalisation).
    """
    # ── Lifetime counters ─────────────────────────────────────────────────────
    sp.total_trades += 1
    sp.total_pnl    += pnl
    if pnl > 0:
        sp.wins   += 1
    else:
        sp.losses += 1

    # ── Segregated tier counters ──────────────────────────────────────────────
    if sp.status == STATUS_PAPER:
        sp.paper_trades += 1
        sp.paper_pnl    += pnl
    elif sp.status == STATUS_LIVE:
        sp.live_trades += 1
        sp.live_pnl    += pnl

    # ── Returns buffer (Sharpe compat — no round()) ───────────────────────────
    norm_r = pnl / max(abs(trade_value), 1.0)
    sp.returns_buffer.append(norm_r)
    if len(sp.returns_buffer) > 500:
        sp.returns_buffer = sp.returns_buffer[-400:]

    # ── Refresh composite score ───────────────────────────────────────────────
    sp._refresh_score()


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — Ghost trade logger  (paper mode simulation)
# ─────────────────────────────────────────────────────────────────────────────

def log_ghost_trade(
    sp: StrategyParamsV2,
    signal: str,
    entry_price: float,
    exit_price: float,
    size: float = 1.0,
    symbol: str = "",
    timestamp: Optional[float] = None,
) -> dict:
    """
    Simulate a trade without using real margin and record it in
    ``sp.ghost_log``.  Also calls ``record_trade_v2`` to keep all lifetime
    stats consistent (so promotion logic sees the real P&L numbers).

    This function is called by the strategy manager whenever a paper strategy
    emits a signal and the position is later closed — it should NOT be called
    for live strategies.

    Parameters
    ----------
    sp:
        Must have ``status == 'paper'``.
    signal:
        ``'buy'`` or ``'sell'`` — the direction of the ghost trade.
    entry_price:
        Price at which the ghost position was opened.
    exit_price:
        Price at which the ghost position was closed.
    size:
        Notional size (units / contracts).  Default 1.0.
    symbol:
        Ticker symbol, for logging.
    timestamp:
        Unix timestamp of close.  Defaults to ``None`` (manager can fill in).

    Returns
    -------
    dict
        The ghost trade record that was appended to ``sp.ghost_log``.

    Raises
    ------
    ValueError
        If called on a non-paper strategy (safety guard).
    """
    if sp.status != STATUS_PAPER:
        raise ValueError(
            f"log_ghost_trade called on strategy '{sp.name}' "
            f"which is in status='{sp.status}', not 'paper'."
        )

    if signal == "buy":
        pnl = (exit_price - entry_price) * size
    elif signal == "sell":
        pnl = (entry_price - exit_price) * size
    else:
        pnl = 0.0

    record = {
        "symbol":      symbol,
        "signal":      signal,
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "size":        size,
        "pnl":         round(pnl, 6),
        "timestamp":   timestamp,
    }
    sp.ghost_log.append(record)

    # Update stats (treats paper trade as a real closed trade for scoring purposes)
    record_trade_v2(sp, pnl=pnl, trade_value=entry_price * size)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — Promotion gate (paper → live)
# ─────────────────────────────────────────────────────────────────────────────

def try_promote(sp: StrategyParamsV2) -> bool:
    """
    Evaluate whether a paper strategy has earned promotion to live trading.

    Promotion criteria (both must be satisfied simultaneously):
    * At least ``PROMOTION_MIN_TRADES`` closed paper trades.
    * Net paper PnL is **strictly positive** (``paper_pnl > PROMOTION_MIN_PNL``).

    If promoted, ``sp.status`` is set to ``'live'`` and the paper-tier counters
    are reset to zero so the live-tier can be evaluated cleanly from day one.
    The ghost log is preserved for audit purposes.

    Parameters
    ----------
    sp:
        Strategy to evaluate.  No-op (returns ``False``) if already live or dead.

    Returns
    -------
    bool
        ``True`` if the strategy was promoted in this call, ``False`` otherwise.
    """
    if sp.status != STATUS_PAPER:
        return False

    meets_trade_gate = sp.paper_trades >= PROMOTION_MIN_TRADES
    meets_pnl_gate   = sp.paper_pnl   >  PROMOTION_MIN_PNL

    if meets_trade_gate and meets_pnl_gate:
        sp.status = STATUS_LIVE
        # Reset live counters so demotion threshold is measured from promotion date
        sp.live_trades = 0
        sp.live_pnl    = 0.0
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 3 — Demotion gate (live → paper  'The Bench')
# ─────────────────────────────────────────────────────────────────────────────

def try_demote(
    sp: StrategyParamsV2,
    score_threshold: float = DEMOTION_SCORE_THRESHOLD,
) -> bool:
    """
    Evaluate whether a live strategy should be demoted back to paper mode.

    **Why demotion, not deletion?**
    Market regimes rotate.  A mean-reversion strategy that bleeds in a strong
    trend will recover when the trend ends.  By moving it to the Bench (paper
    mode) instead of deleting it, the DNA is preserved and it can re-earn live
    status once market conditions suit it again.

    Demotion criteria:
    * Strategy must currently be ``'live'``.
    * Fitness score has dropped below ``score_threshold``
      (default ``DEMOTION_SCORE_THRESHOLD = -3.0``).

    On demotion:
    * ``sp.status`` → ``'paper'``
    * Live-tier counters are reset to zero so the strategy must prove itself
      afresh from paper before the next promotion attempt.
    * Paper-tier counters are also reset so it cannot coast on stale paper
      history from before its first promotion.

    Parameters
    ----------
    sp:
        Strategy to evaluate.
    score_threshold:
        Score below which the strategy is demoted.  Override to tighten
        or loosen the demotion gate without changing the module constant.

    Returns
    -------
    bool
        ``True`` if the strategy was demoted in this call, ``False`` otherwise.
    """
    if sp.status != STATUS_LIVE:
        return False

    if sp.score < score_threshold:
        sp.status = STATUS_PAPER
        # Reset both tier counters — must re-earn promotion from scratch
        sp.live_trades  = 0
        sp.live_pnl     = 0.0
        sp.paper_trades = 0
        sp.paper_pnl    = 0.0
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 4 — Hard Kill Switch
# ─────────────────────────────────────────────────────────────────────────────

def check_kill_switch(
    sp: StrategyParamsV2,
    kill_threshold: float = KILL_SWITCH_PNL,
) -> bool:
    """
    Permanently retire a strategy that has crossed the hard drawdown limit.

    This is the *only* path to permanent deletion.  It applies to both paper
    and live strategies so that a paper strategy cannot silently bleed
    indefinitely in the incubator.

    The kill-switch checks **lifetime** ``total_pnl`` (not per-tier PnL) so
    that a strategy cannot avoid the kill by bouncing between paper and live.

    Parameters
    ----------
    sp:
        Strategy to evaluate.
    kill_threshold:
        PnL level that triggers permanent death.  Must be negative.
        Defaults to ``KILL_SWITCH_PNL = -50.0``.

    Returns
    -------
    bool
        ``True`` if the strategy has been killed in this call (status set to
        ``'dead'``).  ``False`` if it is healthy.

    Notes
    -----
    The caller (strategy manager) is responsible for removing ``'dead'``
    strategies from the active pool after logging the kill event.
    """
    if sp.status == STATUS_DEAD:
        return False  # already dead — idempotent

    if sp.total_pnl <= kill_threshold:
        sp.status = STATUS_DEAD
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — Factory:  generate_random_strategy_v2
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_strategy_v2(existing_names: set) -> tuple:
    """
    Randomly assemble a new Gen-2 strategy from modular building blocks.

    Identical combinatorial logic to the original ``generate_random_strategy``
    except the returned ``StrategyParamsV2`` is **always** created with
    ``status='paper'``.  No newly generated strategy ever touches live margin.

    Parameters
    ----------
    existing_names:
        Set of strategy name strings already in the pool.  Used to avoid
        collisions by appending a numeric suffix.

    Returns
    -------
    (StrategyParamsV2, strategy_fn)
        The parameter container and the callable signal function.
    """
    if not ENTRY_FNS or not FILTER_FNS:
        raise RuntimeError(
            "ENTRY_FNS / FILTER_FNS are empty — ensure strategies.py is "
            "importable before calling generate_random_strategy_v2."
        )

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

    # ── Build the V2 params object ─────────────────────────────────────────
    sp = StrategyParamsV2(name=name, params=params)
    sp.is_generated = True
    sp.blueprint    = {"entry": entry, "filter": filter_, "logic": logic}
    # status defaults to STATUS_PAPER — never set to live here

    fn = build_generated_strategy(entry, filter_, logic, params, name)
    return sp, fn


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: full lifecycle tick
# ─────────────────────────────────────────────────────────────────────────────

def process_closed_trade(
    sp: StrategyParamsV2,
    pnl: float,
    trade_value: float = 1.0,
    *,
    demotion_threshold: float = DEMOTION_SCORE_THRESHOLD,
    kill_threshold: float = KILL_SWITCH_PNL,
) -> dict:
    """
    Single entry-point that records a closed trade **and** runs all four
    lifecycle checks in the correct order.  Returns a status-change report.

    Intended to be called by the strategy manager after every trade close.
    Paper ghost trades should use ``log_ghost_trade`` instead (which internally
    calls ``record_trade_v2``); live trades should call this function.

    Order of operations
    ~~~~~~~~~~~~~~~~~~~
    1. Record the trade result.
    2. Kill-switch check — if triggered, stop and return.
    3. If paper: evaluate promotion.
    4. If live:  evaluate demotion.

    Parameters
    ----------
    sp:
        Strategy to update.
    pnl:
        Realised profit/loss for this trade.
    trade_value:
        Notional size for Sharpe normalisation.
    demotion_threshold:
        Forwarded to ``try_demote``.
    kill_threshold:
        Forwarded to ``check_kill_switch``.

    Returns
    -------
    dict with keys:
        ``prev_status``  — status before this call.
        ``new_status``   — status after this call.
        ``killed``       — bool, True if kill switch fired.
        ``promoted``     — bool, True if promoted paper → live.
        ``demoted``      — bool, True if demoted live → paper.
        ``score``        — current score after update.
        ``total_pnl``    — cumulative PnL after this trade.
    """
    prev_status = sp.status

    # Step 1: record
    record_trade_v2(sp, pnl=pnl, trade_value=trade_value)

    killed   = False
    promoted = False
    demoted  = False

    # Step 2: kill switch (highest priority — overrides everything)
    if check_kill_switch(sp, kill_threshold=kill_threshold):
        killed = True
    elif sp.status == STATUS_PAPER:
        # Step 3: try promotion
        promoted = try_promote(sp)
    elif sp.status == STATUS_LIVE:
        # Step 4: try demotion
        demoted = try_demote(sp, score_threshold=demotion_threshold)

    return {
        "prev_status": prev_status,
        "new_status":  sp.status,
        "killed":      killed,
        "promoted":    promoted,
        "demoted":     demoted,
        "score":       sp.score,
        "total_pnl":   sp.total_pnl,
    }
