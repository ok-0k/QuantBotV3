"""
Read-only analytics over the trade journal, shared by the dashboard
(/api/insights) and scripts/experiment_report.py.

Opens its own read-only SQLite connection, so it can never write to or lock
the live database.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from config import DB_PATH, EXPERIMENT_NAME, FEE_GATE_ROUND_TRIP, SLIPPAGE_PCT, SYMBOLS

# Round-trip cost of a trade as a % of notional: fee + slippage on both fills.
COST_PCT = (FEE_GATE_ROUND_TRIP + 2 * SLIPPAGE_PCT) * 100

# ── Experiment: per-symbol-day metric ──────────────────────────────────────────
# The assignment unit is the symbol, so the decision metric is net P&L per
# symbol per calendar day (UTC), zero-trade symbols included. Delisted symbols
# never trade and are left out of both arms.
DELISTED_SYMBOLS = frozenset({"MATICUSDT", "MKRUSDT", "OCEANUSDT", "AGIXUSDT"})
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261006
DECISION_RULE = ("Adopt B (designed decay) unless the 95% CI shows B significantly worse "
                 "(upper bound < 0).")
EXPERIMENT_NOTES = {
    "decay_v2": "Arm B ran without persistent break-even/trailing stops "
                "(known bug, fixed behind V3_FIXES).",
}


def connect(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _utc_day(s: str) -> str:
    dt = _ts(s)
    return (dt if dt.tzinfo is None else dt.astimezone(timezone.utc)).date().isoformat()


def _pct(row) -> float | None:
    tv = row["trade_value"]
    return row["pnl"] / tv * 100 if tv else None


def journal_window(conn: sqlite3.Connection, days: float = 7.0) -> dict:
    """Closed-trade stats over the last `days` (journaled exits only)."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT pnl, trade_value, exit_reason, peak_profit_pct, hold_time_seconds FROM trades "
        "WHERE status='filled' AND action IN ('cover','sell') AND exit_reason IS NOT NULL "
        "AND pnl IS NOT NULL AND ts >= ?", (since,)).fetchall()
    out = {"days": days, "closed": len(rows), "cost_pct": round(COST_PCT, 3)}
    if not rows:
        return out | {"exit_mix": []}
    pnl = [r["pnl"] for r in rows]
    pct = [p for p in (_pct(r) for r in rows) if p is not None]
    holds = [r["hold_time_seconds"] / 60 for r in rows if r["hold_time_seconds"] is not None]
    peaks = [r["peak_profit_pct"] for r in rows if r["peak_profit_pct"] is not None]
    mix: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        mix[r["exit_reason"]].append(r["pnl"])
    out.update(
        net=round(sum(pnl), 2),
        win_rate=round(100 * sum(p > 0 for p in pnl) / len(pnl), 1),
        avg_pct=round(st.mean(pct), 3) if pct else None,
        median_hold_min=round(st.median(holds), 1) if holds else None,
        mean_hold_min=round(st.mean(holds), 1) if holds else None,
        cleared_cost_share=round(100 * sum(p >= COST_PCT for p in peaks) / len(peaks), 1) if peaks else None,
        exit_mix=sorted(
            ({"reason": k, "n": len(v), "share": round(100 * len(v) / len(rows), 1), "net": round(sum(v), 2)}
             for k, v in mix.items()),
            key=lambda x: -x["n"]),
    )
    return out


def lifetime(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT value FROM portfolio WHERE key='total_trades'").fetchone()
    return {
        "lifetime_trades": int(float(row[0])) if row else None,
        "rows_kept": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
        "first_row": conn.execute("SELECT MIN(ts) FROM trades").fetchone()[0],
    }


def arm_symbols() -> dict[str, list[str]]:
    """Symbols per arm by the bot's assignment rule (bot._experiment_arm:
    alternate by SYMBOLS index, A on even), delisted symbols excluded."""
    out: dict[str, list[str]] = {"A": [], "B": []}
    for i, sym in enumerate(SYMBOLS):
        if sym not in DELISTED_SYMBOLS:
            out["A" if i % 2 == 0 else "B"].append(sym)
    return out


def day_bootstrap_ci(values: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling whole days with replacement
    (absorbs cross-symbol correlation within a day). Fixed seed: same data,
    same interval."""
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def per_symbol_day(closed: list[tuple[str, str, str, float]], days: list[str]) -> dict | None:
    """Mean daily net P&L per symbol for each arm and the B - A difference with
    a day-bootstrap 95% CI. `closed` = (arm, symbol, utc_day, net_pnl)."""
    if not days:
        return None
    symbols = {arm: set(syms) for arm, syms in arm_symbols().items()}
    daily: dict[str, dict[str, float]] = {"A": defaultdict(float), "B": defaultdict(float)}
    for arm, sym, day, pnl in closed:
        if arm in daily and sym not in DELISTED_SYMBOLS:
            symbols[arm].add(sym)          # traded but no longer in SYMBOLS: still counts
            daily[arm][day] += pnl
    n = {arm: max(len(symbols[arm]), 1) for arm in daily}
    per_day = {arm: [daily[arm][d] / n[arm] for d in days] for arm in daily}
    diffs = [b - a for a, b in zip(per_day["A"], per_day["B"])]
    out = {
        "days": len(days), "first_day": days[0], "last_day": days[-1],
        "symbols": {arm: len(symbols[arm]) for arm in daily},
        "net_per_symbol": {arm: round(sum(daily[arm].values()) / n[arm], 3) for arm in daily},
        "mean": {arm: round(st.mean(per_day[arm]), 3) for arm in daily},
        "diff": round(st.mean(diffs), 3), "ci_lo": None, "ci_hi": None,
        "resamples": BOOTSTRAP_RESAMPLES, "rule": DECISION_RULE,
        "verdict": "not enough days yet (need 2)",
    }
    if len(days) >= 2:
        lo, hi = day_bootstrap_ci(diffs)
        out.update(ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                   verdict=("Keep A: B significantly worse" if hi < 0 else
                            "Adopt B: B significantly better" if lo > 0 else
                            "Adopt B: no significant difference"))
    return out


def experiment(conn: sqlite3.Connection, name: str | None = None) -> dict | None:
    """Per-arm stats for a symbol-split experiment: the headline net P&L per
    symbol-day (B - A, day-bootstrap 95% CI, pre-registered decision rule)
    plus the per-trade comparison (Welch 95% CI on mean net return)."""
    name = EXPERIMENT_NAME if name is None else name
    if not name:
        return None
    closed: dict[str, list] = defaultdict(list)
    entries: dict[str, list[str]] = defaultdict(list)
    skips: dict[str, Counter] = defaultdict(Counter)
    sym_days: list[tuple[str, str, str, float]] = []
    active_days: set[str] = set()
    for r in conn.execute(
        "SELECT ts, symbol, action, status, reason, pnl, trade_value, fee_total, exit_reason, "
        "hold_time_seconds, entry_features FROM trades WHERE entry_features LIKE ? ORDER BY id",
        (f'%"experiment": "{name}"%',),
    ):
        f = json.loads(r["entry_features"])
        arm = f.get("arm")
        if f.get("experiment") != name or arm is None:
            continue
        day = _utc_day(r["ts"])
        if r["action"] in ("short", "buy"):
            if r["status"] == "filled":
                entries[arm].append(r["ts"])
                active_days.add(day)
            else:
                skips[arm][r["reason"] or "?"] += 1
        elif r["status"] == "filled" and r["pnl"] is not None:
            closed[arm].append(r)
            sym_days.append((arm, r["symbol"], day, r["pnl"]))
            active_days.add(day)
    open_now: Counter = Counter()
    for (feats,) in conn.execute("SELECT entry_features FROM positions WHERE margin_reserved > 0 OR shares > 0"):
        try:
            f = json.loads(feats or "{}")
        except ValueError:
            continue
        if f.get("experiment") == name:
            open_now[f.get("arm")] += 1

    arms = {}
    rets = {}
    for arm in sorted(set(closed) | set(entries)):
        rows = closed.get(arm, [])
        ent = entries.get(arm, [])
        span_days = (_ts(ent[-1]) - _ts(ent[0])).total_seconds() / 86400 if len(ent) > 1 else 0.0
        a = {"entries": len(ent), "closed": len(rows), "open": open_now[arm],
             "entries_per_day": round(len(ent) / span_days, 1) if span_days > 0.04 else None,
             "skips": dict(skips[arm].most_common(5))}
        if rows:
            pnl = [x["pnl"] for x in rows]
            ret = [p for p in (_pct(x) for x in rows) if p is not None]
            rets[arm] = ret
            holds = [x["hold_time_seconds"] / 60 for x in rows if x["hold_time_seconds"] is not None]
            a.update(net=round(sum(pnl), 2), win_rate=round(100 * sum(p > 0 for p in pnl) / len(pnl), 1),
                     avg_trade=round(st.mean(pnl), 3), avg_pct=round(st.mean(ret), 3) if ret else None,
                     fees=round(sum(x["fee_total"] or 0 for x in rows), 2),
                     median_hold_min=round(st.median(holds), 1) if holds else None,
                     mean_hold_min=round(st.mean(holds), 1) if holds else None,
                     exits=dict(Counter(x["exit_reason"] or "?" for x in rows).most_common()))
        arms[arm] = a

    # Every calendar day from the first to the last experiment fill, so days
    # on which an arm made nothing count as zero.
    days: list[str] = []
    if active_days:
        d0 = datetime.fromisoformat(min(active_days)).date()
        d1 = datetime.fromisoformat(max(active_days)).date()
        days = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]

    out = {"name": name, "arms": arms, "diff": None, "ci_lo": None, "ci_hi": None,
           "verdict": "not enough closed trades yet",
           "per_symbol_day": per_symbol_day(sym_days, days) if sym_days else None,
           "note": EXPERIMENT_NOTES.get(name)}
    if len(rets) == 2 and all(len(v) >= 2 for v in rets.values()):
        a, b = (rets[k] for k in sorted(rets))
        diff = st.mean(b) - st.mean(a)
        se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        out.update(diff=round(diff, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                   verdict="not yet distinguishable" if lo <= 0 <= hi else ("B better" if lo > 0 else "A better"))
    return out


def snapshot(db_path=None) -> dict:
    """Everything the dashboard's Insights section shows, in one read."""
    conn = connect(db_path)
    try:
        return {
            "generated": datetime.now(timezone.utc).isoformat(),
            "week": journal_window(conn, 7.0),
            "day": journal_window(conn, 1.0),
            "lifetime": lifetime(conn),
            "experiment": experiment(conn),
        }
    finally:
        conn.close()
