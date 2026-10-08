"""
Read-only analytics over the trade journal, shared by the dashboard
(/api/insights) and scripts/experiment_report.py.

Opens its own read-only SQLite connection, so it can never write to or lock
the live database.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from config import DB_PATH, EXPERIMENT_NAME, FEE_GATE_ROUND_TRIP, SLIPPAGE_PCT

# Round-trip cost of a trade as a % of notional: fee + slippage on both fills.
COST_PCT = (FEE_GATE_ROUND_TRIP + 2 * SLIPPAGE_PCT) * 100


def connect(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


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


def experiment(conn: sqlite3.Connection, name: str | None = None) -> dict | None:
    """Per-arm stats for a symbol-split experiment, plus a 95% CI on the
    difference in mean net return per closed trade (B - A)."""
    name = EXPERIMENT_NAME if name is None else name
    if not name:
        return None
    closed: dict[str, list] = defaultdict(list)
    entries: dict[str, list[str]] = defaultdict(list)
    skips: dict[str, Counter] = defaultdict(Counter)
    for r in conn.execute(
        "SELECT ts, action, status, reason, pnl, trade_value, fee_total, exit_reason, "
        "hold_time_seconds, entry_features FROM trades WHERE entry_features LIKE ? ORDER BY id",
        (f'%"experiment": "{name}"%',),
    ):
        f = json.loads(r["entry_features"])
        arm = f.get("arm")
        if f.get("experiment") != name or arm is None:
            continue
        if r["action"] in ("short", "buy"):
            if r["status"] == "filled":
                entries[arm].append(r["ts"])
            else:
                skips[arm][r["reason"] or "?"] += 1
        elif r["status"] == "filled" and r["pnl"] is not None:
            closed[arm].append(r)
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

    out = {"name": name, "arms": arms, "diff": None, "ci_lo": None, "ci_hi": None,
           "verdict": "not enough closed trades yet"}
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
