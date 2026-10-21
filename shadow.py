"""
Shadow learner (config.SHADOW_LEARNER, default on; env SHADOW_LEARNER=false
switches it off). Pre-registration: research/PREREG_SHADOW.md.

Every short entry the evidence gate refuses is also opened as a SHADOW trade:
no capital, no orders, no effect on any real decision. Shadow trades are
resolved on closed 1-minute candles with the bot's own exit rules (the pure
helpers bot.trail_stop_move, bot.breakeven_stop_move, bot.designed_decay_moves
and the take-profit fee gate bot._profit_clears_fees) and its own cost booking
(SLIPPAGE_PCT on every fill, accounting_v2.net_realized_pnl for fees), plus
0.03%/day borrow on shorts. Per exact strategy name, closed shadow trades
measure net R per trade (R = the initial stop distance).

Isolation: the entry path only appends a small record to an in-memory queue
(capture()); all database work happens in process(), which housekeeping runs
in an executor thread with its own connection and its own transactions, never
inside the booking transaction. Nothing here ever raises into the caller.

Evidence: running numbers are NOT evidence. A name can become a candidate only
at the two fixed evaluation dates T0 + 8 weeks and T0 + 16 weeks, under the
rule in CANDIDATE_RULE (t >= 3.6 = Bonferroni over ~150 names x 2 looks).
Promotion to VALIDATED_STRATEGIES is manual (independent code-vs-spec check,
PM review, owner edit). Demotion (runtime blocklist) is automatic.
"""

from __future__ import annotations

import json
import logging
import math
import statistics as st
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from accounting_v2 import net_realized_pnl
from config import (HARD_STOP_LOSS_USD, MAX_HOLD_OPEN_SECONDS, SHADOW_BORROW_DAILY,
                    SHADOW_LEARNER, SHADOW_MAX_HOLD_HOURS, SHADOW_NOTIONAL_USD,
                    SHADOW_QUEUE_MAX, SLIPPAGE_PCT)

log = logging.getLogger("shadow")

MINUTE_MS = 60_000
GAP_GRACE_MS = 5 * MINUTE_MS        # wait this long for a backfill before closing at the stop
NO_DATA_CLOSE_MS = 30 * MINUTE_MS   # no candles at all for this long: close at the stop
EVAL_OFFSETS_DAYS = (56, 112)       # T0 + 8 weeks, T0 + 16 weeks
NAME_FREEZE_LOOKBACK_DAYS = 14      # names eligible = seen in the 14 days before T0
CANDIDATE_RULE = {"min_trades": 300, "min_days": 20, "min_mean_r": 0.10,
                  "min_t": 3.6, "max_symbol_share": 0.25}
DEMOTION_RULE = {"window": 50, "max_mean_net_usd": 0.0}
T0_KEY = "shadow_t0"
STATS_KEY = "shadow_stats"
BLOCKLIST_KEY = "evidence_blocklist"

_queue: deque = deque(maxlen=SHADOW_QUEUE_MAX)
_lock = threading.Lock()            # process() is never run twice at once


@dataclass(frozen=True)
class Rules:
    """The bot's own pure exit helpers, passed in so this module never
    imports bot.py (and tests can pass the same functions)."""
    trail: Callable          # (is_short, tp, stop, high, low, atr) -> Optional[float]
    breakeven: Callable      # (is_short, avg, stop, high, low, close, atr) -> (Optional[float], bool)
    decay: Callable          # (entry, tp0, sl0, age_min, stop, tp, is_short) -> (tp|None, sl|None)
    clears_fees: Callable    # (avg_cost, shares, exit_price, is_short) -> bool
    atr: Callable            # (candles) -> float


# ── Entry path: in-memory only ───────────────────────────────────────────────

def capture(record: dict) -> None:
    """Queue one refused short signal. Never touches the database, never raises."""
    if not SHADOW_LEARNER:
        return
    try:
        _queue.append(dict(record))
    except Exception as exc:          # pragma: no cover - defensive
        log.debug("shadow capture failed: %s", exc)


def drain() -> list[dict]:
    out = []
    while True:
        try:
            out.append(_queue.popleft())
        except IndexError:
            return out


# ── Cost booking (the bot's own) ─────────────────────────────────────────────

def book_short(entry_exec: float, exit_price: float, hold_min: float, stop0: float) -> tuple[float, float]:
    """(net_usd, r_net) for a SHADOW_NOTIONAL_USD short: exit fill slipped by
    SLIPPAGE_PCT against us, fees via net_realized_pnl, 0.03%/day borrow."""
    shares = SHADOW_NOTIONAL_USD / entry_exec
    fill = exit_price * (1 + SLIPPAGE_PCT)
    entry_notional, exit_notional = shares * entry_exec, shares * fill
    _, net = net_realized_pnl(entry_notional - exit_notional, entry_notional, exit_notional)
    net -= entry_notional * SHADOW_BORROW_DAILY * max(hold_min, 0.0) / 1440.0
    risk = (stop0 - entry_exec) * shares
    return net, (net / risk if risk > 0 else float("nan"))


# ── Resolution on closed 1m candles ──────────────────────────────────────────

def step(trade: dict, candle: dict, history: list[dict], rules: Rules) -> Optional[tuple[str, float]]:
    """Apply one closed candle to an open short shadow trade: survival ->
    trail -> break-even -> decay -> stop in place (wick) -> target in place
    (wick, fee-gated) -> moved stop (close) -> moved target (close, fee-gated)
    -> 4h losing exit -> shadow cap. Mutates trade['stop'] / ['tp'].
    Returns (exit_reason, exit_price) or None. `history` = closed candles
    before this one (the ATR the tick path would see)."""
    entry, stop, tp = trade["entry"], trade["stop"], trade["tp"]
    o, h, l, c = (float(candle[k]) for k in ("open", "high", "low", "close"))
    shares = SHADOW_NOTIONAL_USD / entry
    age_min = (int(candle["time"]) + MINUTE_MS - trade["opened_ms"]) / MINUTE_MS
    stop_open, tp_open = stop, tp
    atr = rules.atr(history) if len(history) >= 2 else 0.0

    if (entry - c) * shares < HARD_STOP_LOSS_USD:
        return "hard_stop_survival", c
    new = rules.trail(True, tp, stop, h, l, atr)
    if new is not None:
        stop = new
    be, _ = rules.breakeven(True, entry, stop, h, l, c, atr)
    if be is not None:
        stop = be
    new_tp, new_sl = rules.decay(entry, trade["tp0"], trade["stop0"], age_min, stop, tp, True)
    if new_tp is not None:
        tp = new_tp
    if new_sl is not None:
        stop = new_sl
    trade["stop"], trade["tp"] = stop, tp

    # Levels in place since the minute began are tested on the wick, stop
    # first (ambiguity goes against the trade). A target touched this minute
    # exits before a stop the same move trailed/moved (the tick path exits at
    # the target on the tick that touches it). Levels moved this minute are
    # tested on the mark (the close) only.
    if stop_open > 0 and h >= stop_open:
        return "stop_loss_short", max(stop_open, o)
    if tp_open > 0 and l <= tp_open and rules.clears_fees(entry, shares, tp_open, True):
        return "take_profit_short", tp_open
    if stop > 0 and c >= stop:
        return "stop_loss_short", c
    if tp > 0 and c <= tp and rules.clears_fees(entry, shares, tp, True):
        return "take_profit_short", tp
    if age_min * 60 >= MAX_HOLD_OPEN_SECONDS and (entry - c) * shares < 0:
        return "time_hold_exit", c
    if age_min >= SHADOW_MAX_HOLD_HOURS * 60:
        return "shadow_max_hold", c
    return None


def resolve_trade(trade: dict, series: list[dict], rules: Rules, now_ms: int) -> Optional[tuple[str, float, int]]:
    """Walk the closed candles after trade['last_ms']. Never crosses a missing
    candle: inside GAP_GRACE_MS it waits for a backfill; after that it closes
    at the current stop. Returns (reason, price, candle_ms) or None; updates
    trade['last_ms'] / ['stop'] / ['tp'] as it goes."""
    idx = {int(k["time"]): i for i, k in enumerate(series)}
    while True:
        nxt = trade["last_ms"] + MINUTE_MS
        i = idx.get(nxt)
        if i is None:
            later = [t for t in idx if t > nxt]
            if not later:
                if now_ms - nxt >= NO_DATA_CLOSE_MS:
                    return "gap_stop", trade["stop"], nxt     # no candles at all for too long
                return None                              # candle not closed yet
            if now_ms - nxt < GAP_GRACE_MS:
                return None                              # give the backfill a chance
            return "gap_stop", trade["stop"], nxt
        out = step(trade, series[i], series[max(0, i - 299):i], rules)
        trade["last_ms"] = nxt
        if out is not None:
            return out[0], out[1], nxt


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def process(records: list[dict], candles: dict[str, list[dict]], rules: Rules,
            now_ms: int, db) -> dict:
    """Insert queued shadow entries (one open per (symbol, strategy)) and
    resolve every open shadow trade. Runs in an executor thread; each write is
    its own transaction on this thread's connection. Returns counters."""
    stats = {"queued": len(records), "opened": 0, "duplicates": 0, "closed": 0,
             "gap_stops": 0, "errors": 0}
    if not _lock.acquire(blocking=False):
        return {**stats, "skipped_busy": 1}
    try:
        with db.get_db() as conn:
            if conn.execute("SELECT value FROM portfolio WHERE key=?", (T0_KEY,)).fetchone() is None and records:
                conn.execute("INSERT OR IGNORE INTO portfolio VALUES (?, ?)", (T0_KEY, records[0]["opened_ts"]))
        for r in records:
            try:
                with db.get_db() as conn:
                    dup = conn.execute(
                        "SELECT 1 FROM shadow_trades WHERE status='open' AND symbol=? AND strategy=?",
                        (r["symbol"], r["strategy"])).fetchone()
                    if dup:
                        stats["duplicates"] += 1
                        continue
                    conn.execute(
                        "INSERT INTO shadow_trades (opened_ts, opened_ms, symbol, strategy, side, regime, "
                        "entry, stop0, tp0, stop, tp, last_ms, status, entry_features) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'open',?)",
                        (r["opened_ts"], r["opened_ms"], r["symbol"], r["strategy"], "short", r.get("regime"),
                         r["entry"], r["stop0"], r["tp0"], r["stop0"], r["tp0"], r["last_ms"],
                         json.dumps(r.get("features") or {})))
                    stats["opened"] += 1
            except Exception as exc:
                stats["errors"] += 1
                log.debug("shadow insert failed: %s", exc)
        with db.get_db() as conn:
            rows = [dict(x) for x in conn.execute("SELECT * FROM shadow_trades WHERE status='open'")]
        for t in rows:
            try:
                trade = {"entry": t["entry"], "stop": t["stop"], "tp": t["tp"], "stop0": t["stop0"],
                         "tp0": t["tp0"], "opened_ms": t["opened_ms"], "last_ms": t["last_ms"]}
                out = resolve_trade(trade, candles.get(t["symbol"], []), rules, now_ms)
                with db.get_db() as conn:
                    if out is None:
                        conn.execute("UPDATE shadow_trades SET stop=?, tp=?, last_ms=? WHERE id=?",
                                     (trade["stop"], trade["tp"], trade["last_ms"], t["id"]))
                        continue
                    reason, price, cms = out
                    hold = (cms + MINUTE_MS - t["opened_ms"]) / MINUTE_MS
                    net, r = book_short(t["entry"], price, hold, t["stop0"])
                    conn.execute(
                        "UPDATE shadow_trades SET stop=?, tp=?, last_ms=?, status='closed', closed_ts=?, "
                        "exit_price=?, exit_reason=?, net_usd=?, r_net=?, hold_min=? WHERE id=?",
                        (trade["stop"], trade["tp"], cms, _iso(cms + MINUTE_MS), price, reason,
                         net, r, hold, t["id"]))
                    stats["closed"] += 1
                    stats["gap_stops"] += reason == "gap_stop"
            except Exception as exc:
                stats["errors"] += 1
                log.debug("shadow resolve failed for %s: %s", t.get("symbol"), exc)
        return stats
    finally:
        _lock.release()


# ── Learning statistics ──────────────────────────────────────────────────────

def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def t0(conn) -> Optional[datetime]:
    row = conn.execute("SELECT value FROM portfolio WHERE key=?", (T0_KEY,)).fetchone()
    return _parse(row[0]) if row else None


def eval_dates(t0_dt: datetime) -> list[datetime]:
    return [t0_dt + timedelta(days=d) for d in EVAL_OFFSETS_DAYS]


def eligible_names(conn, t0_dt: datetime) -> set[str]:
    """Names frozen at T0: strategies of entry signals journaled in the
    NAME_FREEZE_LOOKBACK_DAYS before T0. Names first seen later are collected
    but can never become candidates."""
    lo = (t0_dt - timedelta(days=NAME_FREEZE_LOOKBACK_DAYS)).isoformat()
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT strategy FROM trades WHERE action='short' AND ts >= ? AND ts < ? "
        "AND strategy IS NOT NULL", (lo, t0_dt.isoformat()))}


def name_stats(rows: list[dict]) -> dict:
    """n, days, mean net R, day-clustered t, both-halves means, largest
    single-symbol share of net R."""
    rs = [x["r_net"] for x in rows if x["r_net"] is not None and math.isfinite(x["r_net"])]
    out = {"n": len(rs), "days": 0, "mean_r": None, "t": None, "half1": None, "half2": None,
           "max_symbol_share": None}
    if not rs:
        return out
    by_day = defaultdict(list)
    for x in rows:
        if x["r_net"] is not None and math.isfinite(x["r_net"]):
            by_day[x["opened_ts"][:10]].append(x["r_net"])
    days = sorted(by_day)
    means = [st.mean(by_day[d]) for d in days]
    out["days"], out["mean_r"] = len(days), st.mean(rs)
    if len(means) >= 2 and st.stdev(means) > 0:
        out["t"] = st.mean(means) / (st.stdev(means) / math.sqrt(len(means)))
    mid = days[len(days) // 2] if len(days) >= 2 else None
    if mid:
        h1 = [r for d in days if d < mid for r in by_day[d]]
        h2 = [r for d in days if d >= mid for r in by_day[d]]
        out["half1"] = st.mean(h1) if h1 else None
        out["half2"] = st.mean(h2) if h2 else None
    total = sum(rs)
    if total > 0:
        by_sym = defaultdict(float)
        for x in rows:
            if x["r_net"] is not None and math.isfinite(x["r_net"]):
                by_sym[x["symbol"]] += x["r_net"]
        out["max_symbol_share"] = max(by_sym.values()) / total
    return out


def passes(s: dict) -> bool:
    R = CANDIDATE_RULE
    return (s["n"] >= R["min_trades"] and s["days"] >= R["min_days"]
            and s["mean_r"] is not None and s["mean_r"] >= R["min_mean_r"]
            and s["t"] is not None and s["t"] >= R["min_t"]
            and (s["half1"] or 0) > 0 and (s["half2"] or 0) > 0
            and s["max_symbol_share"] is not None and s["max_symbol_share"] <= R["max_symbol_share"])


def _closed(conn, lo: datetime, hi: datetime) -> dict[str, list[dict]]:
    """Closed shadow trades opened in [lo, hi) and closed by hi, per name."""
    out = defaultdict(list)
    for r in conn.execute(
            "SELECT strategy, symbol, opened_ts, closed_ts, r_net FROM shadow_trades "
            "WHERE status='closed' AND opened_ts >= ? AND opened_ts < ? AND closed_ts <= ?",
            (lo.isoformat(), hi.isoformat(), hi.isoformat())):
        out[r[0]].append({"symbol": r[1], "opened_ts": r[2], "closed_ts": r[3], "r_net": r[4]})
    return out


def evaluate(conn, at: datetime) -> Optional[dict]:
    """The pre-registered evaluation at a fixed date (T0 + 8 or 16 weeks):
    candidates = eligible names that pass CANDIDATE_RULE on trades opened in
    [T0, at). Returns None if `at` is not an evaluation date or is in the future."""
    t0_dt = t0(conn)
    if t0_dt is None or at not in eval_dates(t0_dt) or at > datetime.now(timezone.utc):
        return None
    names = eligible_names(conn, t0_dt)
    per = {n: name_stats(rows) for n, rows in _closed(conn, t0_dt, at).items()}
    return {"at": at.isoformat(), "t0": t0_dt.isoformat(), "rule": CANDIDATE_RULE,
            "candidates": sorted(n for n, s in per.items() if n in names and passes(s)),
            "stats": per, "eligible": sorted(names)}


def summary(conn, now: Optional[datetime] = None, top: int = 10) -> dict:
    """Running numbers for the dashboard and the CLI. NOT evidence before the
    next evaluation date; completed evaluations are reported as they stand."""
    now = now or datetime.now(timezone.utc)
    t0_dt = t0(conn)
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM shadow_trades GROUP BY status").fetchall())
    out = {"enabled": SHADOW_LEARNER, "t0": t0_dt.isoformat() if t0_dt else None,
           "open": counts.get("open", 0), "closed": counts.get("closed", 0),
           "next_eval": None, "evaluations": [], "running": []}
    if t0_dt is None:
        return out
    dates = eval_dates(t0_dt)
    out["next_eval"] = next((d.isoformat() for d in dates if d > now), None)
    out["evaluations"] = [e for e in (evaluate(conn, d) for d in dates if d <= now) if e]
    per = {n: name_stats(rows) for n, rows in _closed(conn, t0_dt, now).items()}
    ranked = sorted(((n, s) for n, s in per.items() if s["n"] >= 30),
                    key=lambda kv: -(kv[1]["t"] if kv[1]["t"] is not None else -99))
    out["running"] = [{"name": n, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}}
                      for n, s in ranked[:top]]
    all_r = [r["r_net"] for rows in _closed(conn, t0_dt, now).values() for r in rows
             if r["r_net"] is not None and math.isfinite(r["r_net"])]
    out["all_mean_r"] = round(st.mean(all_r), 3) if all_r else None
    return out


# ── Demotion (automatic, only ever removes permission to trade) ──────────────

def demotion_check(conn, validated: frozenset) -> list[str]:
    """Names in VALIDATED_STRATEGIES whose last DEMOTION_RULE['window'] closed
    real trades have mean net P&L <= the floor. Added to the runtime
    blocklist; only the owner clears it."""
    newly = []
    if not validated:
        return newly
    block = set(blocklist(conn))
    for name in sorted(validated - block):
        rows = conn.execute(
            "SELECT net_pnl FROM trades WHERE action='cover' AND status='filled' AND net_pnl IS NOT NULL "
            "AND json_extract(entry_features, '$.strategy') = ? ORDER BY id DESC LIMIT ?",
            (name, DEMOTION_RULE["window"])).fetchall()
        if len(rows) >= DEMOTION_RULE["window"] and st.mean(r[0] for r in rows) <= DEMOTION_RULE["max_mean_net_usd"]:
            newly.append(name)
    if newly:
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?)",
                     (BLOCKLIST_KEY, json.dumps(sorted(block | set(newly)))))
    return newly


def blocklist(conn) -> list[str]:
    row = conn.execute("SELECT value FROM portfolio WHERE key=?", (BLOCKLIST_KEY,)).fetchone()
    try:
        return list(json.loads(row[0])) if row else []
    except (TypeError, ValueError):
        return []
