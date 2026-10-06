"""
dashboard.py — FastAPI + SSE dashboard. Engine V1.5 data bridge + risk columns.
Revision: Execution-focused command centre.
  - Removed ML sections (Strategy Brain, Mutations, Gen Log, Strategy PnL, WinRate/Sharpe charts)
  - Timezone: all timestamps converted to America/Vancouver (Pacific, DST-aware)
  - Time-Based Performance section: Today + Week PnL, per-day breakdown
  - MFE / MAE columns in Trade History
  - Interactive Equity Curve with timeframe buttons + backend downsampling
  - Fee Drag KPI card
"""
from __future__ import annotations
import asyncio
import datetime
import json
import time
import zoneinfo
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import DASHBOARD_HOST, DASHBOARD_PORT, FEE_GATE_ROUND_TRIP, SSE_HEARTBEAT_SECS, STARTING_CASH, SYMBOLS
from db import (
    get_all_positions,
    get_cash,
    get_cash_curve_from_trades,
    get_equity_curve,
    get_equity_curve_since,
    get_equity_rebase,
    get_filled_trade_count,
    get_portfolio_stat,
    get_recent_trades,
    get_trades_last_7_days,
    init_db,
    load_brain_key,
)
from accounting_v2 import position_equity_components

# ── Vancouver timezone ─────────────────────────────────────────────────────────
VAN_TZ = zoneinfo.ZoneInfo("America/Vancouver")

# Fix W4: the dashboard is strictly read-only -- bot.py is the sole writer of
# persisted equity/PnL state. Peak equity used to be re-derived AND written
# back to portfolio.peak_equity_all_time on every telemetry build, racing
# bot.py's own writes to current_equity/return_pct. It now ratchets up in
# this process's memory only; a dashboard restart re-seeds it (harmlessly)
# from the read-only equity_curve/portfolio history the next call already
# fetches, so no information is actually lost by not persisting it.
_dashboard_peak_equity: float = 0.0


def _utc_to_van(ts_str: str | None) -> str:
    """Convert an ISO-8601 UTC timestamp string → Vancouver local time string."""
    if not ts_str:
        return ""
    # Ordered longest-to-shortest so the most specific format wins.
    # "%Y-%m-%d %H:%M" handles 16-char equity-curve timestamps (no seconds);
    # previously these fell through to the raw-UTC fallback, causing the equity
    # chart to display UTC times while all text metrics showed Pacific.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            dt = datetime.datetime.strptime(ts_str[:26], fmt).replace(tzinfo=datetime.timezone.utc)
            van_dt = dt.astimezone(VAN_TZ)
            return van_dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return ts_str[:19]


def _van_today_prefix() -> str:
    """Return today's date string in Vancouver time, e.g. '2024-07-15'."""
    return datetime.datetime.now(tz=VAN_TZ).strftime("%Y-%m-%d")


def _van_week_day_prefixes() -> list[tuple[str, str]]:
    """
    Return a true rolling-7-day window anchored to today in Vancouver time.
    Rows are ordered oldest → newest so Today appears at the bottom.
    Labels: 'Today', 'Yesterday', '2d ago', …, '6d ago'.
    """
    now = datetime.datetime.now(tz=VAN_TZ)
    result = []
    for i in range(6, -1, -1):   # 6d ago … today
        day = now - datetime.timedelta(days=i)
        if i == 0:
            label = "Today"
        elif i == 1:
            label = "Yesterday"
        else:
            label = f"{i}d ago"
        result.append((day.strftime("%Y-%m-%d"), label))
    return result


def _downsample_equity(curve: list[dict], max_points: int) -> list[dict]:
    """
    Keep at most max_points, evenly spaced in TIME (first point of each equal
    time bucket; the final bucket is represented by the latest point).

    Index-uniform thinning distorted mixed-density history — 30 s points for
    the last 2 days, 5-minute points before that (prune_db.sh) — by giving the
    dense recent stretch most of the chart width.
    """
    n = len(curve)
    if n <= max_points or max_points < 2:
        return curve
    times = [_parse_equity_ts(str(pt.get("time", ""))) for pt in curve]
    span = (times[-1] - times[0]).total_seconds()
    if span <= 0:
        return [curve[0], curve[-1]]
    width = span / (max_points - 1)
    out: list[dict] = []
    last_bucket = -1
    for pt, t in zip(curve, times):
        bucket = int((t - times[0]).total_seconds() // width)
        if bucket != last_bucket:
            out.append(pt)
            last_bucket = bucket
    out[-1] = curve[-1]
    return out


def _normalize_strategy(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw).strip()
    return s if s else None


def _position_risk_metrics(side: str, entry: float, stop_px: float, tp_px: float) -> dict[str, Any]:
    entry = max(entry, 1e-12)
    sp = float(stop_px or 0.0)
    tp = float(tp_px or 0.0)
    if side == "short":
        risk   = max(0.0, sp - entry)
        reward = max(0.0, entry - tp)
        sl_pct = ((sp - entry) / entry) * 100.0 if sp > 0 else None
        tp_pct = ((entry - tp) / entry) * 100.0 if tp > 0 else None
    else:
        risk   = max(0.0, entry - sp)
        reward = max(0.0, tp - entry)
        sl_pct = ((entry - sp) / entry) * 100.0 if sp > 0 else None
        tp_pct = ((tp - entry) / entry) * 100.0 if tp > 0 else None
    rr = (reward / risk) if risk > 1e-12 else None
    return {
        "risk_per_share":   round(risk, 8),
        "reward_per_share": round(reward, 8),
        "rr_ratio":         round(rr, 3) if rr is not None else None,
        "sl_pct":           round(sl_pct, 3) if sl_pct is not None else None,
        "tp_pct":           round(tp_pct, 3) if tp_pct is not None else None,
    }


def _format_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    tag = _normalize_strategy(out.get("strategy"))
    if tag is None:
        tag = _normalize_strategy(out.get("strat"))
    out["strategy"] = tag
    if out.get("net_pnl") is None and out.get("pnl") is not None:
        out["net_pnl"] = out.get("pnl")
    # MFE / MAE — map from new engine columns
    out["mfe"] = out.get("max_unrealized_pnl")   # peak profit while open
    out["mae"] = out.get("min_unrealized_pnl")   # max drawdown while open
    # Convert timestamps to Vancouver time
    for key in ("ts", "timestamp", "opened_ts", "opened_at"):
        if out.get(key):
            out[key] = _utc_to_van(out[key])
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Quant Bot — Execution Dashboard", lifespan=lifespan)


# ── Optional access token ─────────────────────────────────────────────────────
# DASHBOARD_AUTH_TOKEN unset/empty  -> open access (legacy behaviour).
# When set, every request must present the token via one of:
#   • Authorization: Bearer <token>
#   • ?token=<token>  (first browser visit; sets a session cookie)
#   • qb_auth cookie  (set automatically after a ?token= visit)
# Read-only surface, but account state should not be LAN-readable by default.
import hmac as _hmac
import os as _os

_AUTH_TOKEN = _os.getenv("DASHBOARD_AUTH_TOKEN", "").strip()
_AUTH_COOKIE = "qb_auth"


def _token_ok(candidate: str | None) -> bool:
    return bool(candidate) and _hmac.compare_digest(candidate, _AUTH_TOKEN)


@app.middleware("http")
async def _require_token(request: Request, call_next):
    if not _AUTH_TOKEN:
        return await call_next(request)

    supplied = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        supplied = auth_header[7:].strip()
    query_token = request.query_params.get("token")
    cookie_token = request.cookies.get(_AUTH_COOKIE)

    if _token_ok(supplied) or _token_ok(query_token) or _token_ok(cookie_token):
        response = await call_next(request)
        if query_token and _token_ok(query_token):
            response.set_cookie(
                _AUTH_COOKIE, _AUTH_TOKEN,
                httponly=True, samesite="strict", max_age=30 * 24 * 3600,
            )
        return response

    return JSONResponse({"detail": "unauthorized"}, status_code=401)


def _build_telemetry(equity_tf: str | None = None) -> dict:
    global _dashboard_peak_equity
    try:
        init_db()
    except Exception:
        pass

    cash         = get_cash()
    positions_db = get_all_positions()
    trades_raw   = get_recent_trades(100)
    equity_curve_raw = get_equity_curve(10_000)  # fetch large; we downsample per TF
    total_eq     = float(cash)
    pos_list: list[dict[str, Any]] = []

    for pos in positions_db:
        sym      = pos["symbol"]
        avg_cost = float(pos.get("avg_cost", 0.0) or 0.0)
        shares   = float(pos.get("shares", 0.0) or 0.0)
        side     = (pos.get("side") or "long").lower()
        raw_lp   = float(get_portfolio_stat(f"last_price_{sym}", "0") or 0)
        lp       = raw_lp if raw_lp > 0 else avg_cost
        stop_price = float(pos.get("stop_price", 0.0) or 0.0)
        tp_price   = float(pos.get("tp_price", 0.0) or 0.0)
        strat_tag  = _normalize_strategy(pos.get("strategy"))

        pos_value, unr, contrib = position_equity_components(
            side=side, avg_cost=avg_cost, quantity=shares, mark_price=lp,
            margin_reserved=float(pos.get("margin_reserved", 0.0) or 0.0),
        )
        total_eq += contrib

        if side == "short":
            total_invested = round(pos_value, 2)
            potential_loss = (
                round((stop_price - avg_cost) * shares, 2) if stop_price > avg_cost else None
            )
            potential_gain = (
                round((avg_cost - tp_price) * shares, 2) if 0 < tp_price < avg_cost else None
            )
            stop_loss_total = round(stop_price * shares, 2) if stop_price else None
            tp_total        = round(tp_price * shares, 2)   if tp_price  else None
        else:
            total_invested = round(pos_value, 2)
            potential_loss = (
                round((avg_cost - stop_price) * shares, 2) if stop_price and stop_price < avg_cost else None
            )
            potential_gain = (
                round((tp_price - avg_cost) * shares, 2) if tp_price and tp_price > avg_cost else None
            )
            stop_loss_total = round(stop_price * shares, 2) if stop_price else None
            tp_total        = round(tp_price * shares, 2)   if tp_price  else None

        rm = _position_risk_metrics(side, avg_cost, stop_price, tp_price)

        # Convert timestamps to Vancouver
        opened_raw = pos.get("opened_ts") or pos.get("opened_at") or ""
        pos_list.append({
            **pos,
            "strategy":          strat_tag,
            "last_price":        round(lp, 6),
            "unrealised_pnl":    round(unr, 2),
            "total_invested":    total_invested,
            "deployed_equity_pct": 0.0,
            "stop_loss_total":   stop_loss_total,
            "tp_total":          tp_total,
            "potential_loss":    potential_loss,
            "potential_gain":    potential_gain,
            "sl_price":          stop_price,
            "tp_price":          tp_price,
            "sl_pct":            rm["sl_pct"],
            "tp_pct":            rm["tp_pct"],
            "rr_ratio":          rm["rr_ratio"],
            "risk_per_share":    rm["risk_per_share"],
            "reward_per_share":  rm["reward_per_share"],
            "opened_ts":         _utc_to_van(opened_raw),
        })

    realised     = float(get_portfolio_stat("realised_pnl",     "0.0"))
    realised_net = float(get_portfolio_stat("realised_pnl_net", "0.0"))
    try:
        margin_health = json.loads(get_portfolio_stat("margin_health_json", "") or "{}")
    except (json.JSONDecodeError, TypeError):
        margin_health = {}

    unrealised_pnl_total = round(sum(float(r.get("unrealised_pnl") or 0.0) for r in pos_list), 2)

    for row in pos_list:
        inv = float(row.get("total_invested") or 0.0)
        row["deployed_equity_pct"] = round(100.0 * inv / max(total_eq, 1e-9), 3)

    # equity_rebase_baseline/ts (set by rebase_equity_baseline.py) take over
    # as the reference point for return_pct / peak-equity when present,
    # instead of always using STARTING_CASH. STARTING_CASH keeps its original
    # meaning everywhere else (fresh-install seed amount, bot.py's
    # cash-invariant check) -- this only changes what the dashboard displays.
    rebase_baseline, rebase_ts = get_equity_rebase()
    ref_equity = rebase_baseline if rebase_baseline and rebase_baseline > 0 else STARTING_CASH

    total_net_pnl = round(total_eq - ref_equity, 2)
    return_pct    = round(total_net_pnl / ref_equity * 100, 3)

    # ── Peak equity ────────────────────────────────────────────────────────────
    # When a rebase is active, only equity observed AFTER the rebase counts
    # toward the peak -- otherwise pre-rebase curve history (inflated by the
    # since-fixed C3 capital leak) would immediately drag "peak since rebase"
    # straight back up.
    rebase_dt = _parse_equity_ts(rebase_ts) if rebase_ts else None
    if rebase_dt is not None:
        curve_equity_vals = [
            float(pt.get("equity", 0.0) or 0.0) for pt in equity_curve_raw
            if _parse_equity_ts(str(pt.get("time", ""))) >= rebase_dt
        ]
    else:
        curve_equity_vals = [float(pt.get("equity", 0.0) or 0.0) for pt in equity_curve_raw]
    # Fix W4: read the last-known persisted peak (harmless -- a read, not a
    # write) as a floor, but hold the ratcheted result in process memory only.
    # bot.py never writes this key, and neither do we anymore -- the previous
    # set_portfolio_stat call here was racing bot.py's own current_equity /
    # return_pct writes every ~2s.
    stored_peak         = float(get_portfolio_stat("peak_equity_all_time", str(ref_equity)) or ref_equity)
    _dashboard_peak_equity = max(_dashboard_peak_equity, stored_peak, total_eq, ref_equity, *curve_equity_vals)
    peak_equity          = _dashboard_peak_equity
    drawdown_pct = round((total_eq - peak_equity) / peak_equity * 100, 3) if peak_equity > 0 else 0.0

    # ── Prices ─────────────────────────────────────────────────────────────────
    prices = []
    for sym in SYMBOLS:
        price = float(get_portfolio_stat(f"last_price_{sym}", "0") or "0")
        prev  = float(get_portfolio_stat(f"prev_price_{sym}", "0") or "0")
        chg   = round((price - prev) / prev * 100, 2) if prev > 0 else 0.0
        prices.append({"symbol": sym.replace("USDT", ""), "price": price, "change": chg})

    # Fix 2: time-based window (7 days) instead of an arbitrary row count cap.
    all_trades = get_trades_last_7_days()

    def _trade_net_amt(row: dict[str, Any]) -> float | None:
        if row.get("net_pnl") is not None:
            return float(row["net_pnl"])
        if row.get("pnl") is not None:
            return float(row["pnl"])
        return None

    closed         = [t for t in all_trades if _trade_net_amt(t) is not None]

    # ── Lifetime trade count — brain-snapshot + current DB count ───────────────
    # The trades table may be pruned (prune_db.sh), so SELECT COUNT(*) alone
    # understates the true total.  brain_state.total_trades holds the count at
    # the time of the last manual snapshot (default: 6629, known pre-prune
    # baseline).  db_count captures trades since that snapshot.
    brain_snapshot   = int(load_brain_key("total_trades", 6629) or 0)
    db_count         = get_filled_trade_count()
    lifetime_trades  = brain_snapshot + db_count

    # ── Session win rate from available trade window ──────────────────────────
    closed_wins      = sum(1 for t in closed if (_trade_net_amt(t) or 0) > 0)
    closed_total     = len(closed)
    win_rate_overall = (
        round(closed_wins / closed_total * 100, 1) if closed_total > 0 else 0.0
    )

    # Long / Short win-rate breakdown (recent sample)
    long_closed   = [t for t in closed if (t.get("side") or "long").lower() == "long"]
    short_closed  = [t for t in closed if (t.get("side") or "long").lower() == "short"]
    long_wins     = sum(1 for t in long_closed if (_trade_net_amt(t) or 0) > 0)
    short_wins    = sum(1 for t in short_closed if (_trade_net_amt(t) or 0) > 0)
    win_rate_long = round(long_wins / len(long_closed) * 100, 1) if long_closed else None
    win_rate_short = round(short_wins / len(short_closed) * 100, 1) if short_closed else None

    # ── Daily PnL — two metrics ────────────────────────────────────────────────
    # (A) Realized-only: closed trades today in Pacific time → weekly table
    # (B) Equity-based: current_equity − equity_at_midnight_pacific → KPI cards
    # Using (B) for the headline ensures floating PnL is included, matching what
    # the equity chart shows.
    today_prefix = _van_today_prefix()

    def _ts_van_date(t: dict) -> str:
        raw = t.get("ts") or t.get("timestamp") or ""
        van = _utc_to_van(raw)
        return van[:10] if van else ""

    daily_closed = [t for t in all_trades if _ts_van_date(t) == today_prefix and _trade_net_amt(t) is not None]
    daily_pnl    = round(sum(_trade_net_amt(t) for t in daily_closed), 2)
    daily_trades = len(daily_closed)
    daily_wins   = sum(1 for t in daily_closed if (_trade_net_amt(t) or 0) > 0)

    # Equity-based daily PnL: find the equity snapshot closest to/before Pacific midnight
    van_midnight = datetime.datetime.now(tz=VAN_TZ).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    midnight_utc_str = van_midnight.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    equity_at_midnight: float = float(STARTING_CASH)
    for _pt in equity_curve_raw:
        _pt_time = _pt.get("time", "")   # stored as "YYYY-MM-DD HH:MM" UTC
        if _pt_time <= midnight_utc_str:
            equity_at_midnight = float(_pt.get("equity", STARTING_CASH) or STARTING_CASH)
        else:
            break  # past midnight — stop scanning forward
    daily_equity_pnl = round(total_eq - equity_at_midnight, 2)

    # ── Weekly breakdown ───────────────────────────────────────────────────────
    week_days = _van_week_day_prefixes()
    weekly_total_pnl = 0.0
    weekly_breakdown = []
    for date_prefix, day_name in week_days:
        day_trades = [t for t in all_trades if _ts_van_date(t) == date_prefix and _trade_net_amt(t) is not None]
        day_pnl    = round(sum(_trade_net_amt(t) for t in day_trades), 2) if day_trades else 0.0
        day_wins   = sum(1 for t in day_trades if (_trade_net_amt(t) or 0) > 0)
        day_wr     = round(day_wins / len(day_trades) * 100, 1) if day_trades else None
        weekly_total_pnl += day_pnl
        weekly_breakdown.append({
            "date":        date_prefix,
            "day":         day_name,
            "pnl":         day_pnl,
            "trades":      len(day_trades),
            "wins":        day_wins,
            "win_rate":    day_wr,
            "is_today":    date_prefix == today_prefix,
        })
    weekly_total_pnl = round(weekly_total_pnl, 2)

    # ── Fee drag ───────────────────────────────────────────────────────────────
    # For trades closed before V2.3 (fee_total was stored as 0), derive the fee
    # from trade_value × FEE_GATE_ROUND_TRIP (0.12%) as a conservative estimate.
    def _effective_fee(t: dict[str, Any]) -> float:
        stored = float(t.get("fee_total") or 0.0)
        if stored > 0:
            return stored
        # V2.2 era: fee_total==0 → estimate from notional
        tv = float(t.get("trade_value") or 0.0)
        return tv * FEE_GATE_ROUND_TRIP if tv > 0 else 0.0

    total_fees_paid = round(sum(_effective_fee(t) for t in all_trades), 4)
    gross_pnl       = round(sum(float(t.get("gross_pnl") or t.get("pnl") or 0.0) for t in closed), 2)
    net_pnl_total   = round(sum(_trade_net_amt(t) for t in closed), 2)
    fee_efficiency  = round(net_pnl_total / gross_pnl * 100, 2) if gross_pnl != 0 else None

    # Profit factor = gross wins / |gross losses|  (using net_pnl per trade)
    _wins_sum  = sum((_trade_net_amt(t) or 0) for t in closed if (_trade_net_amt(t) or 0) > 0)
    _loss_sum  = abs(sum((_trade_net_amt(t) or 0) for t in closed if (_trade_net_amt(t) or 0) < 0))
    profit_factor = round(_wins_sum / _loss_sum, 3) if _loss_sum > 0 else None

    # ── Equity curve — apply timeframe downsampling ────────────────────────────
    TF_MINUTES = {"1H": 60, "2H": 120, "5H": 300, "12H": 720, "24H": 1440, "1W": 10080, "1M": 43200}
    MAX_POINTS = {"1H": 60, "2H": 120, "5H": 150, "12H": 200, "24H": 300, "1W": 350, "1M": 400}

    tf = (equity_tf or "24H").upper()
    tf_minutes = TF_MINUTES.get(tf, 1440)
    max_pts    = MAX_POINTS.get(tf, 300)

    # Filter curve to requested window (curve points have a 'time' field)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    cutoff_utc = now_utc - datetime.timedelta(minutes=tf_minutes)
    if tf in ("1W", "1M"):
        # Long windows read their exact time range (history beyond 2 days is
        # retained at 5-minute resolution) rather than the row-capped raw curve.
        equity_window = get_equity_curve_since(cutoff_utc.isoformat()) or equity_curve_raw[-max_pts:]
    else:
        equity_window = [
            pt for pt in equity_curve_raw
            if _parse_equity_ts(pt.get("time", "")) >= cutoff_utc
        ] or equity_curve_raw[-max_pts:]

    equity_curve_out = _downsample_equity(equity_window, max_pts)

    # Convert equity curve timestamps to Vancouver
    for pt in equity_curve_out:
        if pt.get("time"):
            pt["time"] = _utc_to_van(pt["time"])

    # Fix 1 — equity curve fallback: if the table was wiped or has only one row,
    # Chart.js refuses to draw a line with < 2 distinct points.  Synthesise a
    # flat baseline from the current equity so the chart renders immediately.
    if len(equity_curve_out) == 0:
        _now_van = datetime.datetime.now(tz=VAN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        equity_curve_out = [
            {"time": _now_van, "equity": round(total_eq, 2)},
            {"time": _now_van, "equity": round(total_eq, 2)},
        ]
    elif len(equity_curve_out) == 1:
        equity_curve_out = equity_curve_out + [equity_curve_out[0]]

    trades_out = [_format_trade_row(dict(t)) for t in trades_raw[:100]]

    # Current Vancouver time for header
    van_now = datetime.datetime.now(tz=VAN_TZ)
    ts_display = van_now.strftime("%H:%M:%S") + " PT"

    # ── Brain / regime state ───────────────────────────────────────────────────
    # bot.py writes current_regime and circuit_open to portfolio every 60 s via
    # housekeeping_loop so the dashboard never needs a direct import of brain.py.
    current_regime = get_portfolio_stat("current_regime", "ranging") or "ranging"
    circuit_open   = (get_portfolio_stat("circuit_open", "0") or "0") == "1"

    # Fix 3 — HUD AI penalty: read size_mult for current side:regime from the
    # persisted edge profiles so the HUD can display the live penalty level.
    _ep_raw = load_brain_key("adaptive_edge_profiles_v1", {}) or {}
    if isinstance(_ep_raw, dict):
        _ep_key = f"short:{current_regime}"
        _ep_data = _ep_raw.get(_ep_key) or {}
        _current_size_mult = float(_ep_data.get("size_mult", 1.0))
    else:
        _current_size_mult = 1.0

    # Regime distribution from closed trade history (proxy for recent regime mix)
    regime_history = [
        {"regime": t.get("regime") or "ranging"}
        for t in all_trades
        if t.get("regime")
    ]

    return {
        "portfolio": {
            "cash":                  round(cash, 2),
            # total_equity = cash + mark-to-market value of all open positions
            # (position_equity_components accumulates long: mark×qty, short: margin+uPnL)
            "total_equity":          round(total_eq, 2),
            "total_net_pnl":         total_net_pnl,
            "return_pct":            return_pct,
            "realised_pnl":          round(realised, 2),
            # realised_pnl_net deducts 0.12% round-trip fees (V2.3+).
            "realised_pnl_net":      round(realised_net, 2),
            "unrealised_pnl_total":  unrealised_pnl_total,
            # lifetime_trades = brain_state.total_trades (pre-prune snapshot, default
            # 6629) + get_filled_trade_count() (rows added since snapshot).
            "lifetime_trades":       lifetime_trades,
            "total_trades":          lifetime_trades,
            "drawdown_pct":          drawdown_pct,
            "peak_equity":           round(peak_equity, 2),
            "win_rate":              win_rate_overall,
            "win_rate_long":         win_rate_long,
            "win_rate_short":        win_rate_short,
            # daily_pnl: realized closed-trade PnL today (consistent with weekly table)
            "daily_pnl":             daily_pnl,
            # daily_equity_pnl: equity-based total daily change (includes floating PnL).
            # This matches what the equity curve chart shows and eliminates the
            # 1H/24H discrepancy from open losing positions.
            "daily_equity_pnl":      daily_equity_pnl,
            "daily_trades":          daily_trades,
            "daily_wins":            daily_wins,
            "weekly_pnl":            weekly_total_pnl,
            "weekly_breakdown":      weekly_breakdown,
            "total_fees_paid":       total_fees_paid,
            "fee_efficiency":        fee_efficiency,
            "gross_pnl":             gross_pnl,
            "profit_factor":         profit_factor,
            # ref_equity (not the raw STARTING_CASH constant): the frontend's
            # balance self-check (starting_cash + total_net_pnl == total_equity)
            # and the "vs $X" / today's-%% widgets all key off this field, so it
            # must match whatever return_pct/total_net_pnl were computed against
            # above, or a rebase would manufacture a brand new false "drift".
            "starting_cash":         round(ref_equity, 2),
        },
        "margin_health": margin_health,
        "positions":    pos_list,
        "trades":       trades_out,
        "equity_curve": equity_curve_out,
        "equity_tf":    tf,
        "cash_curve":   [
            {**pt, "time": _utc_to_van(pt["time"])}
            for pt in get_cash_curve_from_trades(200)
        ],
        "prices":       prices,
        "ts":           ts_display,
        "engine_version": "2.1",
        # Brain state — populated from portfolio stats written by bot.py housekeeping.
        "brain": {
            "current_regime":    current_regime,
            "circuit_open":      circuit_open,
            "regime_history":    regime_history,
            "current_size_mult": round(_current_size_mult, 4),
        },
    }


def _parse_equity_ts(ts_str: str) -> datetime.datetime:
    """Parse equity curve time string → UTC datetime for windowing."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(ts_str[:19], fmt)
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)


async def _sse_generator(request: Request) -> AsyncGenerator:
    while True:
        if await request.is_disconnected():
            break
        try:
            data = _build_telemetry()
            # Strip heavy curve arrays from SSE — served by dedicated endpoints
            # so SSE payloads stay small and don't clobber the user's TF selection.
            data["equity_curve"] = []
            data["equity_tf"]    = None
            data["cash_curve"]   = []  # served by /api/cash
            yield {"event": "update", "data": json.dumps(data)}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        await asyncio.sleep(SSE_HEARTBEAT_SECS)


@app.get("/stream")
async def stream(request: Request):
    return EventSourceResponse(_sse_generator(request))


@app.get("/api/data")
async def api_data():
    return _build_telemetry()


@app.get("/api/equity")
async def api_equity(tf: str = "24H"):
    """Lightweight endpoint for equity curve timeframe switching."""
    data = _build_telemetry(equity_tf=tf.upper())
    return JSONResponse({"equity_curve": data["equity_curve"], "equity_tf": data["equity_tf"]})


@app.get("/api/cash")
async def api_cash(limit: int = 2000):
    """Cash-flow curve endpoint.  limit caps at 10 000 to prevent abuse."""
    limit = min(max(limit, 1), 10_000)
    pts = get_cash_curve_from_trades(limit)
    return JSONResponse({
        "cash_curve": [
            {**pt, "time": _utc_to_van(pt["time"])}
            for pt in pts
        ]
    })


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0a0a0b">
<title>QuantBot — Portfolio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
/* ══════════════════════════════ DESIGN TOKENS ══════════════════════════════ */
:root{
  --bg:        #0a0a0b;
  --card:      rgba(255,255,255,0.035);
  --card-hover:rgba(255,255,255,0.05);
  --border:    rgba(255,255,255,0.07);
  --border-soft: rgba(255,255,255,0.045);

  --t1: #f7f7f8;
  --t2: #98989f;
  --t3: #626269;

  --pos: #22c55e;
  --pos-dim: rgba(34,197,94,0.14);
  --neg: #fb4d5c;
  --neg-dim: rgba(251,77,92,0.14);
  --warn: #f5a623;
  --warn-dim: rgba(245,166,35,0.14);

  --radius-lg: 22px;
  --radius-md: 14px;
  --radius-sm: 9px;
  --shadow-card: 0 1px 1px rgba(0,0,0,.35), 0 10px 28px -10px rgba(0,0,0,.6);

  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, system-ui, sans-serif;
}

*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
html{height:100%;-webkit-text-size-adjust:100%;}
body{
  background:var(--bg);
  color:var(--t2);
  font-family:var(--font);
  font-size:14px;line-height:1.45;
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
}
::selection{background:rgba(255,255,255,.16);}
::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.10);border-radius:8px;}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.18);}

.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1;}
.pos{color:var(--pos)!important;}
.neg{color:var(--neg)!important;}
.warn{color:var(--warn)!important;}
.dim{color:var(--t3)!important;}

@keyframes fadeUp{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}
@keyframes scrollTicker{from{transform:translateX(0);}to{transform:translateX(-50%);}}

.page{max-width:920px;margin:0 auto;padding:0 20px 64px;}

/* ══════════════════════════════ TOPBAR ══════════════════════════════ */
.topbar{
  max-width:920px;margin:0 auto;
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 20px 10px;
}
.brand{display:flex;align-items:center;gap:9px;}
.brand-mark{width:8px;height:8px;border-radius:50%;background:var(--pos);flex-shrink:0;animation:pulse 2.4s ease infinite;}
.brand-mark.err{background:var(--neg);animation:pulse .9s ease infinite;}
.brand-mark.cb{background:var(--warn);}
.brand-name{font-weight:700;font-size:14px;color:var(--t1);letter-spacing:-.01em;}
.topbar-right{display:flex;align-items:center;gap:10px;}
.regime-pill{
  font-size:11px;font-weight:600;padding:4px 10px;border-radius:999px;
  background:var(--card);border:1px solid var(--border-soft);color:var(--t2);
  letter-spacing:.01em;white-space:nowrap;
}
.clock{font-size:11px;color:var(--t3);white-space:nowrap;}
.clock .num{font-size:11px;}

/* ══════════════════════════════ ALERT BANNER ══════════════════════════════ */
.alert-banner{
  max-width:920px;margin:0 auto 14px;padding:0 20px;
}
.alert-banner-inner{
  display:flex;align-items:center;gap:10px;
  background:var(--neg-dim);border:1px solid rgba(251,77,92,.28);
  color:#ffd7da;font-size:12.5px;font-weight:500;
  padding:11px 16px;border-radius:var(--radius-md);
  animation:fadeUp .3s ease;
}
.alert-dot{width:6px;height:6px;border-radius:50%;background:var(--neg);flex-shrink:0;animation:pulse 1s infinite;}

/* ══════════════════════════════ TICKER ══════════════════════════════ */
.ticker{
  height:30px;overflow:hidden;
  border-top:1px solid var(--border-soft);border-bottom:1px solid var(--border-soft);
  display:flex;align-items:center;margin-bottom:28px;
}
.ticker-track{display:flex;white-space:nowrap;animation:scrollTicker 90s linear infinite;will-change:transform;}
.ticker:hover .ticker-track{animation-play-state:paused;}
.t-item{display:inline-flex;align-items:center;gap:6px;padding:0 16px;font-size:11.5px;color:var(--t3);border-right:1px solid var(--border-soft);}
.t-item .t-sym{color:var(--t2);font-weight:600;}
.t-item .t-chg.pos{color:var(--pos);}
.t-item .t-chg.neg{color:var(--neg);}

/* ══════════════════════════════ HERO ══════════════════════════════ */
.hero{padding:6px 0 8px;animation:fadeUp .4s ease;}
.hero-label{font-size:13px;color:var(--t3);font-weight:500;margin-bottom:10px;}
.hero-balance{
  font-size:clamp(40px,8vw,64px);font-weight:800;color:var(--t1);
  letter-spacing:-.03em;line-height:1;
}
.hero-delta{
  display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;
  margin-top:14px;font-size:16px;font-weight:600;
}
.hero-delta-sub{font-size:13px;font-weight:500;color:var(--t3);}
.hero-peak{font-size:12.5px;color:var(--t3);margin-top:6px;}

/* Chart */
.chart-toolbar{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;
  margin-top:30px;margin-bottom:14px;
}
.period-badge{font-size:12.5px;font-weight:600;color:var(--t3);}
.tf-pills{display:flex;gap:2px;background:var(--card);border:1px solid var(--border-soft);border-radius:999px;padding:3px;}
.tf-pill{
  font-family:var(--font);font-size:12px;font-weight:600;color:var(--t3);
  padding:6px 14px;border-radius:999px;border:none;background:transparent;cursor:pointer;
  transition:background .15s,color .15s;
}
.tf-pill:hover{color:var(--t1);}
.tf-pill.active{background:var(--t1);color:#0a0a0b;}

.chart-outer{position:relative;}
.chart-wrap{height:280px;width:100%;}
.chart-tooltip{
  position:absolute;top:6px;left:0;pointer-events:none;opacity:0;
  background:rgba(20,20,23,.92);border:1px solid var(--border);
  border-radius:10px;padding:7px 11px;box-shadow:var(--shadow-card);
  transition:opacity .1s;white-space:nowrap;z-index:5;
}
.chart-tooltip .tt-val{font-size:13.5px;font-weight:700;color:var(--t1);}
.chart-tooltip .tt-time{font-size:10.5px;color:var(--t3);margin-top:1px;}

/* ══════════════════════════════ CARDS / SECTIONS ══════════════════════════════ */
section.block{margin-top:36px;}
.block-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}
.block-title{font-size:15px;font-weight:700;color:var(--t1);letter-spacing:-.01em;}
.count-badge{font-size:11px;font-weight:600;color:var(--t3);background:var(--card);border:1px solid var(--border-soft);padding:2px 9px;border-radius:999px;}

/* Stats grid */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px;}
.stat-card{
  background:var(--card);border:1px solid var(--border-soft);border-radius:var(--radius-md);
  padding:15px 16px;transition:background .15s,border-color .15s;
}
.stat-card:hover{background:var(--card-hover);}
.stat-label{font-size:11.5px;color:var(--t3);font-weight:500;margin-bottom:7px;}
.stat-value{font-size:19px;font-weight:700;color:var(--t1);letter-spacing:-.01em;display:block;}
.stat-sub{font-size:11px;color:var(--t3);margin-top:4px;display:block;}
.stat-bar-track{height:3px;border-radius:3px;background:rgba(255,255,255,.08);margin-top:9px;overflow:hidden;}
.stat-bar-fill{height:100%;border-radius:3px;transition:width .4s ease;}

/* Week card */
.week-card{
  background:var(--card);border:1px solid var(--border-soft);border-radius:var(--radius-lg);
  padding:18px 20px 16px;
}
.week-card-hdr{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:16px;}
.week-card-title{font-size:12.5px;color:var(--t3);font-weight:500;}
.week-card-total{font-size:15px;font-weight:700;}
.week-bars{display:flex;align-items:flex-end;gap:8px;height:88px;}
.week-bar-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;gap:7px;}
.week-bar-track{flex:1;width:100%;max-width:26px;display:flex;align-items:flex-end;}
.week-bar-fill{width:100%;border-radius:4px 4px 2px 2px;min-height:3px;transition:height .4s ease;}
.week-bar-lbl{font-size:10px;color:var(--t3);font-weight:500;}
.week-bar-lbl.today{color:var(--t1);font-weight:700;}

/* Generic list card */
.list-card{background:var(--card);border:1px solid var(--border-soft);border-radius:var(--radius-lg);overflow:hidden;}
.empty-state{padding:36px 20px;text-align:center;color:var(--t3);font-size:13px;}

/* Position row */
.pos-row{padding:15px 18px;border-bottom:1px solid var(--border-soft);transition:background .12s;}
.pos-row:last-child{border-bottom:none;}
.pos-row:hover{background:rgba(255,255,255,.02);}
.pos-top{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;}
.pos-id{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.pos-symbol{font-size:14.5px;font-weight:700;color:var(--t1);letter-spacing:-.01em;}
.side-pill{font-size:10px;font-weight:700;letter-spacing:.03em;padding:2px 7px;border-radius:6px;}
.side-pill.long{color:var(--pos);background:var(--pos-dim);}
.side-pill.short{color:var(--neg);background:var(--neg-dim);}
.pos-qty{font-size:12px;color:var(--t3);margin-top:4px;}
.pos-pnl-wrap{text-align:right;flex-shrink:0;}
.pos-pnl-val{font-size:15px;font-weight:700;}
.pos-pnl-pct{font-size:11.5px;margin-top:2px;}
.pos-meta{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:11px;}
.pos-meta span{font-size:11px;color:var(--t3);}
.pos-meta .strat{color:var(--t2);}
.pos-meta .rr{color:var(--warn);font-weight:600;}

/* Activity row */
.act-row{padding:13px 18px;border-bottom:1px solid var(--border-soft);transition:background .12s;}
.act-row:last-child{border-bottom:none;}
.act-row:hover{background:rgba(255,255,255,.02);}
.act-top{display:flex;align-items:center;justify-content:space-between;gap:12px;}
.act-id{display:flex;align-items:center;gap:9px;min-width:0;}
.act-chip{font-size:9.5px;font-weight:700;letter-spacing:.04em;padding:3px 7px;border-radius:6px;flex-shrink:0;}
.act-chip.buy{color:var(--pos);background:var(--pos-dim);}
.act-chip.sell,.act-chip.short{color:var(--neg);background:var(--neg-dim);}
.act-chip.cover{color:var(--warn);background:var(--warn-dim);}
.act-symbol{font-size:13px;font-weight:600;color:var(--t1);}
.act-time{font-size:11px;color:var(--t3);flex-shrink:0;}
.act-pnl{font-size:14px;font-weight:700;flex-shrink:0;}
.act-meta{font-size:10.5px;color:var(--t3);margin-top:6px;padding-left:0;}
.act-meta span+span::before{content:'·';margin:0 6px;color:var(--border);}

/* System grid */
.system-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:0;}
.system-tile{padding:16px 18px;border-bottom:1px solid var(--border-soft);border-right:1px solid var(--border-soft);}
.system-tile:nth-last-child(-n+2){border-bottom:none;}
.system-label{font-size:11px;color:var(--t3);font-weight:500;margin-bottom:6px;}
.system-value{font-size:14.5px;font-weight:700;color:var(--t1);}
.ledger-ok{color:var(--pos);}
.ledger-fail{color:var(--warn);}

/* ══════════════════════════════ RESPONSIVE ══════════════════════════════ */
@media(max-width:600px){
  .topbar{padding:14px 16px 8px;}
  .page{padding:0 16px 48px;}
  .clock{display:none;}
  .hero-balance{font-size:clamp(34px,12vw,48px);}
  .chart-wrap{height:220px;}
  .system-tile:nth-child(2n){border-right:none;}
  .system-tile:nth-last-child(-n+2){border-bottom:1px solid var(--border-soft);}
  .system-tile:nth-last-child(-n+1),.system-tile:nth-last-child(-n+2):last-child{border-bottom:none;}
  .pos-top{flex-direction:column;}
  .pos-pnl-wrap{text-align:left;}
}
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <span class="brand-mark" id="status-dot"></span>
    <span class="brand-name">QuantBot</span>
  </div>
  <div class="topbar-right">
    <span class="regime-pill" id="regime-pill">—</span>
    <span class="clock num" id="clock">—</span>
  </div>
</header>

<div class="alert-banner" id="alert-banner" hidden>
  <div class="alert-banner-inner">
    <span class="alert-dot"></span>
    <span id="alert-text"></span>
  </div>
</div>

<div class="ticker">
  <div class="ticker-track" id="ticker-track">
    <div class="t-item"><span class="t-sym">Connecting…</span></div>
  </div>
</div>

<div class="page">

  <!-- ══════════════════════════ HERO ══════════════════════════ -->
  <section class="hero">
    <div class="hero-label">Portfolio Value</div>
    <h1 class="hero-balance num" id="hero-balance">$0.00</h1>
    <div class="hero-delta">
      <span id="hero-delta-pct" class="num">—</span>
      <span id="hero-delta-amt" class="num dim">—</span>
      <span class="hero-delta-sub">all-time</span>
    </div>
    <div class="hero-peak num" id="hero-peak">—</div>

    <div class="chart-toolbar">
      <span class="period-badge num" id="period-badge">—</span>
      <div class="tf-pills" id="tf-pills">
        <button class="tf-pill" data-tf="1H">1H</button>
        <button class="tf-pill active" data-tf="24H">24H</button>
        <button class="tf-pill" data-tf="1W">1W</button>
        <button class="tf-pill" data-tf="1M">1M</button>
      </div>
    </div>

    <div class="chart-outer">
      <div class="chart-wrap" id="chart-wrap"></div>
      <div class="chart-tooltip" id="chart-tooltip"><div class="tt-val"></div><div class="tt-time"></div></div>
    </div>
  </section>

  <!-- ══════════════════════════ STATS ══════════════════════════ -->
  <section class="block">
    <div class="block-hdr"><span class="block-title">Overview</span></div>
    <div class="stats-grid" id="stats-grid"></div>
  </section>

  <!-- ══════════════════════════ THIS WEEK ══════════════════════════ -->
  <section class="block">
    <div class="week-card">
      <div class="week-card-hdr">
        <span class="week-card-title">This Week</span>
        <span class="week-card-total num" id="week-total">—</span>
      </div>
      <div class="week-bars" id="week-bars"></div>
    </div>
  </section>

  <!-- ══════════════════════════ POSITIONS ══════════════════════════ -->
  <section class="block">
    <div class="block-hdr">
      <span class="block-title">Open Positions</span>
      <span class="count-badge" id="pos-count">0</span>
    </div>
    <div class="list-card" id="positions-list">
      <div class="empty-state">No open positions</div>
    </div>
  </section>

  <!-- ══════════════════════════ ACTIVITY ══════════════════════════ -->
  <section class="block">
    <div class="block-hdr">
      <span class="block-title">Recent Activity</span>
      <span class="count-badge" id="trades-count">0</span>
    </div>
    <div class="list-card" id="activity-list">
      <div class="empty-state">No trades yet</div>
    </div>
  </section>

  <!-- ══════════════════════════ SYSTEM ══════════════════════════ -->
  <section class="block">
    <div class="block-hdr"><span class="block-title">System</span></div>
    <div class="list-card">
      <div class="system-grid" id="system-grid"></div>
    </div>
  </section>

</div>

<script>
'use strict';

/* ══════════════════════════════ HELPERS ══════════════════════════════ */
const g = id => document.getElementById(id);

function fmt$(v, d=2){
  const n = Number(v) || 0;
  const sign = n < 0 ? '-' : '';
  return sign + '$' + Math.abs(n).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
}
function fmtSigned$(v, d=2){
  const n = Number(v) || 0;
  return (n >= 0 ? '+' : '') + fmt$(n, d);
}
function fmtPct(v, d=2){
  const n = Number(v) || 0;
  return (n >= 0 ? '+' : '') + n.toFixed(d) + '%';
}
function fmtPrice(v){
  const n = Number(v) || 0;
  if(!n) return '—';
  if(n >= 10000) return '$' + n.toLocaleString('en-US', {maximumFractionDigits:0});
  if(n >= 100) return '$' + n.toFixed(2);
  if(n >= 1) return '$' + n.toFixed(4);
  return '$' + n.toFixed(6);
}
function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function pnlCls(v){ return v > 0 ? 'pos' : v < 0 ? 'neg' : 'dim'; }
/* Return-on-investment metrics read as strictly up/down (never neutral) —
   unlike per-trade PnL, an exact-0% return is not meaningfully "flat". */
function retCls(v){ return v >= 0 ? 'pos' : 'neg'; }

function parseLocal(ts){
  return new Date(String(ts).replace(' ', 'T'));
}
function relTime(ts){
  if(!ts) return '—';
  const ms = Date.now() - parseLocal(ts).getTime();
  if(isNaN(ms)) return '—';
  if(ms < 0) return 'just now';
  const m = Math.floor(ms/60000);
  if(m < 1) return 'just now';
  if(m < 60) return m + 'm ago';
  const h = Math.floor(m/60);
  if(h < 24) return h + 'h ago';
  const d = Math.floor(h/24);
  if(d < 7) return d + 'd ago';
  return String(ts).slice(5,10);
}
function durFmt(ts){
  if(!ts) return '—';
  const ms = Date.now() - parseLocal(ts).getTime();
  if(isNaN(ms) || ms < 0) return '—';
  const m = Math.floor(ms/60000), h = Math.floor(m/60), d = Math.floor(h/24);
  if(d > 0) return d + 'd ' + (h%24) + 'h';
  if(h > 0) return h + 'h ' + (m%60) + 'm';
  return m + 'm';
}
function regimeLabel(r){
  const map = {trend_up:'Trending Up', trending_up:'Trending Up', trend_down:'Trending Down',
    trending_down:'Trending Down', ranging:'Ranging', volatile:'Volatile', neutral:'Neutral'};
  return map[r] || (r ? r.replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase()) : 'Unknown');
}

function vanClock(){
  return new Intl.DateTimeFormat('en-US', {
    timeZone:'America/Vancouver', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false
  }).format(new Date()) + ' PT';
}
setInterval(()=>{ const c = g('clock'); if(c) c.textContent = vanClock(); }, 1000);

/* ══════════════════════════════ CHART ══════════════════════════════ */
const TF_LIST = ['1H','24H','1W','1M'];
const TF_LABEL = {'1H':'past hour','24H':'past 24 hours','1W':'past week','1M':'past month'};
let activeTf = '24H';
let chart = null, series = null, lastCurve = [];
let lastEqFetch = 0;

function toChartPoints(curve){
  const seen = new Set();
  const out = [];
  for(const pt of (curve||[])){
    if(!pt || !pt.time) continue;
    const t = Math.floor(parseLocal(pt.time).getTime()/1000);
    if(!t || isNaN(t) || seen.has(t)) continue;
    seen.add(t);
    out.push({time:t, value:Number(pt.equity)||0});
  }
  out.sort((a,b)=>a.time-b.time);
  return out;
}

function ensureChart(){
  if(chart || typeof LightweightCharts === 'undefined') return;
  const wrap = g('chart-wrap');
  chart = LightweightCharts.createChart(wrap, {
    autoSize:true,
    layout:{background:{color:'transparent'}, textColor:'#626269', fontFamily:'Inter, system-ui, sans-serif', fontSize:11},
    grid:{vertLines:{visible:false}, horzLines:{visible:false}},
    rightPriceScale:{borderVisible:false, textColor:'#626269'},
    timeScale:{borderVisible:false, timeVisible:true, secondsVisible:false, fixLeftEdge:true, fixRightEdge:true},
    crosshair:{
      mode:LightweightCharts.CrosshairMode.Magnet,
      vertLine:{color:'rgba(255,255,255,.18)', width:1, style:LightweightCharts.LineStyle.Solid, labelVisible:false},
      horzLine:{visible:false, labelVisible:false}
    },
    handleScroll:false,
    handleScale:false,
  });
  series = chart.addAreaSeries({
    lineColor:'#22c55e', topColor:'rgba(34,197,94,.25)', bottomColor:'rgba(34,197,94,0)',
    lineWidth:2, priceLineVisible:false, lastValueVisible:false,
    crosshairMarkerRadius:4, crosshairMarkerBorderWidth:2,
    crosshairMarkerBorderColor:'#0a0a0b', crosshairMarkerBackgroundColor:'#22c55e',
  });

  const tooltip = g('chart-tooltip');
  chart.subscribeCrosshairMove(param=>{
    if(!series || !param || !param.time || !param.point || param.point.x < 0){
      tooltip.style.opacity = 0;
      return;
    }
    const d = param.seriesData.get(series);
    if(!d || d.value == null){ tooltip.style.opacity = 0; return; }
    const dt = new Date(param.time * 1000);
    const intraday = activeTf === '1H' || activeTf === '24H';
    const timeStr = new Intl.DateTimeFormat('en-US', intraday
      ? {hour:'numeric', minute:'2-digit', hour12:true}
      : {month:'short', day:'numeric', hour:'numeric', minute:'2-digit', hour12:true}
    ).format(dt);
    tooltip.querySelector('.tt-val').textContent = fmt$(d.value);
    tooltip.querySelector('.tt-time').textContent = timeStr;
    tooltip.style.opacity = 1;
    const wrapWidth = g('chart-wrap').clientWidth;
    const ttWidth = tooltip.offsetWidth || 90;
    let x = param.point.x - ttWidth/2;
    x = Math.max(4, Math.min(x, wrapWidth - ttWidth - 4));
    tooltip.style.left = x + 'px';
  });
}

function renderChart(curve){
  const points = toChartPoints(curve);
  if(points.length < 2) return;
  lastCurve = points;
  ensureChart();
  if(!series) return;

  const first = points[0].value, last = points[points.length-1].value;
  const up = last >= first;
  const color = up ? '#22c55e' : '#fb4d5c';
  const topColor = up ? 'rgba(34,197,94,.25)' : 'rgba(251,77,92,.22)';
  series.applyOptions({
    lineColor:color, topColor:topColor, bottomColor:'rgba(0,0,0,0)',
    crosshairMarkerBackgroundColor:color,
  });
  series.setData(points);
  chart.timeScale().fitContent();

  const delta = last - first;
  const pctDelta = first !== 0 ? (delta/first*100) : 0;
  const badge = g('period-badge');
  if(badge){
    badge.textContent = fmtSigned$(delta) + ' (' + fmtPct(pctDelta) + ') ' + (TF_LABEL[activeTf] || '');
    badge.className = 'period-badge num ' + (delta >= 0 ? 'pos' : 'neg');
  }
}

document.getElementById('tf-pills').addEventListener('click', e=>{
  const btn = e.target.closest('.tf-pill');
  if(!btn) return;
  const tf = btn.dataset.tf;
  if(tf === activeTf) return;
  activeTf = tf;
  document.querySelectorAll('.tf-pill').forEach(b=>b.classList.toggle('active', b.dataset.tf === tf));
  fetch('/api/equity?tf=' + tf)
    .then(r=>r.json())
    .then(d=>{ if(d.equity_curve && d.equity_curve.length) renderChart(d.equity_curve); })
    .catch(console.error);
});

/* ══════════════════════════════ STATS GRID ══════════════════════════════ */
function renderStats(p){
  const eq = p.total_equity || 0;
  const deployed = Math.max(0, eq - (p.cash||0));
  const deployedPct = eq > 0 ? (deployed/eq*100) : 0;
  const cashPct = eq > 0 ? ((p.cash||0)/eq*100) : 0;

  const cards = [];
  cards.push({label:'Free Cash', value:fmt$(p.cash||0), sub:cashPct.toFixed(1)+'% uninvested'});
  cards.push({label:'Total Return', value:fmtPct(p.return_pct||0), cls:retCls(p.return_pct||0), sub:'vs '+fmt$(p.starting_cash||0)});
  cards.push({label:'Net PnL (all-time)', value:fmtSigned$(p.total_net_pnl||0), cls:pnlCls(p.total_net_pnl||0),
    sub:'R '+fmtSigned$(p.realised_pnl_net||0)+' · U '+fmtSigned$(p.unrealised_pnl_total||0)});
  const dd = p.drawdown_pct || 0;
  cards.push({label:'Drawdown', value:fmtPct(dd), cls: dd < -5 ? 'neg' : dd < -2 ? 'warn' : 'pos',
    sub:'from peak '+fmt$(p.peak_equity||0)});
  const lifetimeTrades = p.lifetime_trades || p.total_trades || 0;
  cards.push({label:'Win Rate', value:(p.win_rate||0).toFixed(1)+'%',
    sub:lifetimeTrades.toLocaleString()+' trades · L '+(p.win_rate_long!=null?p.win_rate_long.toFixed(0)+'%':'—')
      +' / S '+(p.win_rate_short!=null?p.win_rate_short.toFixed(0)+'%':'—'),
    bar:Math.max(0,Math.min(100,p.win_rate||0)), barColor:'var(--pos)'});
  cards.push({label:'Tickets Traded', value:lifetimeTrades.toLocaleString(), sub:'lifetime executions (all symbols)'});
  const pf = p.profit_factor;
  cards.push({label:'Profit Factor', value: pf==null?'—':pf.toFixed(2)+'×',
    cls: pf==null?'':(pf>=1.5?'pos':pf>=1?'warn':'neg'), sub:'gross '+fmtSigned$(p.gross_pnl||0)});
  cards.push({label:'Fees Paid', value:fmt$(-(p.total_fees_paid||0)), cls:(p.total_fees_paid||0)>0?'neg':'dim',
    sub: p.fee_efficiency!=null ? p.fee_efficiency.toFixed(1)+'% net/gross' : 'no closed trades'});
  cards.push({label:'Today', value:fmtSigned$(p.daily_equity_pnl!=null?p.daily_equity_pnl:p.daily_pnl||0),
    cls:pnlCls(p.daily_equity_pnl!=null?p.daily_equity_pnl:p.daily_pnl||0),
    sub:(p.daily_trades||0)+' trades'});
  cards.push({label:'Deployed', value:fmt$(deployed), sub:deployedPct.toFixed(1)+'% of equity',
    bar:deployedPct, barColor: deployedPct>95?'var(--neg)':deployedPct>80?'var(--warn)':'var(--t2)'});

  g('stats-grid').innerHTML = cards.map(c=>`
    <div class="stat-card">
      <div class="stat-label">${c.label}</div>
      <span class="stat-value num ${c.cls||''}">${c.value}</span>
      <span class="stat-sub">${c.sub||''}</span>
      ${c.bar!=null ? `<div class="stat-bar-track"><div class="stat-bar-fill" style="width:${c.bar.toFixed(1)}%;background:${c.barColor}"></div></div>` : ''}
    </div>
  `).join('');
}

/* ══════════════════════════════ WEEK BARS ══════════════════════════════ */
function renderWeek(breakdown, weekTotal){
  const totalEl = g('week-total');
  if(totalEl){
    totalEl.textContent = fmtSigned$(weekTotal||0);
    totalEl.className = 'week-card-total num ' + pnlCls(weekTotal||0);
  }
  const wrap = g('week-bars');
  if(!breakdown || !breakdown.length){ wrap.innerHTML = ''; return; }
  const maxAbs = Math.max(...breakdown.map(d=>Math.abs(d.pnl)), 0.01);
  wrap.innerHTML = breakdown.map(d=>{
    const pct = Math.max(4, Math.min(100, Math.abs(d.pnl)/maxAbs*100));
    const color = d.pnl >= 0 ? 'var(--pos)' : 'var(--neg)';
    const title = d.day + ' ' + d.date + ': ' + fmtSigned$(d.pnl) + ' (' + d.trades + ' trades)';
    return `<div class="week-bar-col" title="${esc(title)}">
      <div class="week-bar-track"><div class="week-bar-fill" style="height:${pct}%;background:${color}"></div></div>
      <span class="week-bar-lbl${d.is_today?' today':''}">${d.day==='Today'?'Today':d.day.slice(0,3)}</span>
    </div>`;
  }).join('');
}

/* ══════════════════════════════ POSITIONS ══════════════════════════════ */
function renderPositions(positions){
  const cnt = g('pos-count');
  if(cnt) cnt.textContent = String((positions||[]).length);
  const el = g('positions-list');
  if(!positions || !positions.length){
    el.innerHTML = '<div class="empty-state">No open positions</div>';
    return;
  }
  el.innerHTML = positions.map(p=>{
    const side = (p.side||'long').toLowerCase();
    const avgCost = p.avg_cost || 0;
    const lp = p.last_price || avgCost;
    const upnl = p.unrealised_pnl || 0;
    const upnlPct = avgCost > 0 ? ((lp-avgCost)/avgCost*(side==='short'?-1:1)*100) : 0;
    const rr = p.rr_ratio;
    const slPx = p.sl_price||0, tpPx = p.tp_price||0;
    const meta = [];
    if(slPx>0) meta.push('Stop '+fmtPrice(slPx)+(p.sl_pct!=null?' ('+fmtPct(-Math.abs(p.sl_pct))+')':''));
    if(tpPx>0) meta.push('Take '+fmtPrice(tpPx)+(p.tp_pct!=null?' (+'+Math.abs(p.tp_pct).toFixed(2)+'%)':''));
    if(rr!=null && rr>0) meta.push('<span class="rr">R∶R 1∶'+rr.toFixed(2)+'</span>');
    if(p.strategy) meta.push('<span class="strat">'+esc(p.strategy)+'</span>');
    const openedTs = p.opened_ts || p.opened_at || '';
    if(openedTs) meta.push(durFmt(openedTs)+' open');
    return `<div class="pos-row">
      <div class="pos-top">
        <div>
          <div class="pos-id">
            <span class="pos-symbol">${p.symbol}</span>
            <span class="side-pill ${side}">${side==='short'?'SHORT':'LONG'}</span>
          </div>
          <div class="pos-qty num">${Number(p.shares||0).toFixed(5)} @ ${fmtPrice(avgCost)} → ${fmtPrice(lp)}</div>
        </div>
        <div class="pos-pnl-wrap">
          <div class="pos-pnl-val num ${pnlCls(upnl)}">${fmtSigned$(upnl)}</div>
          <div class="pos-pnl-pct num ${pnlCls(upnlPct)}">${fmtPct(upnlPct)}</div>
        </div>
      </div>
      <div class="pos-meta">${meta.map(m=>'<span>'+m+'</span>').join('')}</div>
    </div>`;
  }).join('');
}

/* ══════════════════════════════ ACTIVITY ══════════════════════════════ */
function renderActivity(trades){
  const cnt = g('trades-count');
  if(cnt) cnt.textContent = String((trades||[]).length);
  const el = g('activity-list');
  if(!trades || !trades.length){
    el.innerHTML = '<div class="empty-state">No trades yet</div>';
    return;
  }
  el.innerHTML = trades.slice(0,50).map(t=>{
    const action = (t.action||'').toLowerCase();
    const net = t.net_pnl != null ? t.net_pnl : t.pnl;
    const ts = t.ts || t.timestamp || '';
    const meta = [];
    if(t.strategy) meta.push(esc(t.strategy));
    if(t.regime) meta.push(t.regime);
    if(t.fee_total != null) meta.push('fee ' + fmt$(t.fee_total, 4));
    if(t.mfe != null) meta.push('MFE ' + fmtSigned$(t.mfe));
    if(t.mae != null) meta.push('MAE ' + fmtSigned$(t.mae));
    return `<div class="act-row">
      <div class="act-top">
        <div class="act-id">
          <span class="act-chip ${action}">${action.toUpperCase()}</span>
          <span class="act-symbol">${t.symbol}</span>
          <span class="act-time">${relTime(ts)}</span>
        </div>
        <div class="act-pnl num ${net!=null?pnlCls(net):'dim'}">${net!=null?fmtSigned$(net):'—'}</div>
      </div>
      ${meta.length ? `<div class="act-meta">${meta.map(m=>'<span>'+m+'</span>').join('')}</div>` : ''}
    </div>`;
  }).join('');
}

/* ══════════════════════════════ TICKER ══════════════════════════════ */
function renderTicker(prices){
  if(!prices || !prices.length) return;
  const all = [...prices, ...prices];
  g('ticker-track').innerHTML = all.map(p=>{
    const cls = p.change > 0 ? 'pos' : p.change < 0 ? 'neg' : '';
    const arr = p.change > 0 ? '▲' : p.change < 0 ? '▼' : '—';
    return `<div class="t-item"><span class="t-sym">${p.symbol}</span><span class="num">${fmtPrice(p.price)}</span><span class="t-chg ${cls}">${arr} ${Math.abs(p.change).toFixed(2)}%</span></div>`;
  }).join('');
}

/* ══════════════════════════════ SYSTEM ══════════════════════════════ */
function renderSystem(p, brain, marginHealth){
  const tiles = [];
  tiles.push({label:'Circuit Breaker', value: brain.circuit_open ? 'OPEN' : 'Clear',
    cls: brain.circuit_open ? 'neg' : 'pos'});
  const sm = brain.current_size_mult != null ? brain.current_size_mult : 1.0;
  const smPct = Math.round(sm*100);
  tiles.push({label:'AI Sizing', value: smPct>=100 ? '100% nominal' : ('Penalized to '+smPct+'%'),
    cls: smPct>=100?'pos':smPct<75?'neg':'warn'});
  const mh = marginHealth || {};
  const marginParts = [];
  if(mh.utilization_pct != null) marginParts.push(mh.utilization_pct.toFixed(1)+'% util');
  if(mh.margin_level != null) marginParts.push('ML '+mh.margin_level.toFixed(3));
  tiles.push({label:'Margin ('+(mh.source||'—')+')', value: marginParts.length?marginParts.join(' · '):'—',
    cls: mh.halt_new_entries ? 'neg' : ''});
  let ledgerVal = '—', ledgerCls = '';
  if(p.starting_cash != null && p.total_net_pnl != null && p.total_equity != null){
    const computed = Number((p.starting_cash + p.total_net_pnl).toFixed(2));
    const diff = Math.abs(computed - p.total_equity);
    const ok = diff < 0.02;
    ledgerVal = ok ? 'Balanced' : ('Drift ' + fmt$(diff));
    ledgerCls = ok ? 'ledger-ok' : 'ledger-fail';
  }
  tiles.push({label:'Ledger Check', value:ledgerVal, cls:ledgerCls});

  g('system-grid').innerHTML = tiles.map(t=>`
    <div class="system-tile">
      <div class="system-label">${t.label}</div>
      <div class="system-value num ${t.cls||''}">${t.value}</div>
    </div>
  `).join('');
}

/* ══════════════════════════════ APPLY UPDATE ══════════════════════════════ */
function applyUpdate(d){
  const p = d.portfolio || {};
  const brain = d.brain || {};
  const marginHealth = d.margin_health || {};

  /* Status dot + regime pill + alert banner */
  const dot = g('status-dot');
  if(dot) dot.className = 'brand-mark' + (brain.circuit_open ? ' cb' : '');
  const regimeEl = g('regime-pill');
  if(regimeEl) regimeEl.textContent = regimeLabel(brain.current_regime);

  const bannerParts = [];
  if(brain.circuit_open) bannerParts.push('Circuit breaker engaged — new entries paused after drawdown limit');
  if(marginHealth.halt_new_entries) bannerParts.push('New entries halted' + (marginHealth.halt_reason ? ' — ' + marginHealth.halt_reason : ''));
  const banner = g('alert-banner');
  if(banner){
    if(bannerParts.length){
      g('alert-text').textContent = bannerParts.join(' · ');
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
  }

  /* Hero */
  const eq = p.total_equity || 0;
  const heroBal = g('hero-balance');
  if(heroBal) heroBal.textContent = fmt$(eq);
  const retPct = p.return_pct || 0;
  const deltaPctEl = g('hero-delta-pct');
  if(deltaPctEl){
    deltaPctEl.textContent = fmtPct(retPct);
    deltaPctEl.className = 'num ' + retCls(retPct);
  }
  const deltaAmtEl = g('hero-delta-amt');
  if(deltaAmtEl) deltaAmtEl.textContent = '(' + fmtSigned$(p.total_net_pnl||0) + ')';
  const peakEl = g('hero-peak');
  if(peakEl) peakEl.textContent = 'Peak ' + fmt$(p.peak_equity||0);

  /* Stats, week, positions, activity, ticker, system */
  renderStats(p);
  if(p.weekly_breakdown && p.weekly_breakdown.length) renderWeek(p.weekly_breakdown, p.weekly_pnl||0);
  renderPositions(d.positions || []);
  renderActivity(d.trades || []);
  if(d.prices && d.prices.length) renderTicker(d.prices);
  renderSystem(p, brain, marginHealth);

  /* Chart: SSE strips equity_curve (equity_tf=null); re-fetch active TF, throttled */
  if(d.equity_curve && d.equity_curve.length && (!d.equity_tf || d.equity_tf === activeTf)){
    renderChart(d.equity_curve);
  } else {
    const now = Date.now();
    if(now - lastEqFetch > 30000){
      lastEqFetch = now;
      fetch('/api/equity?tf=' + activeTf)
        .then(r=>r.json())
        .then(eqd=>{ if(eqd.equity_curve && eqd.equity_curve.length) renderChart(eqd.equity_curve); })
        .catch(console.error);
    }
  }
}

/* ══════════════════════════════ LIVE WIRE-UP ══════════════════════════════ */
const es = new EventSource('/stream');
es.addEventListener('update', e=>{
  try{
    applyUpdate(JSON.parse(e.data));
    const dot = g('status-dot');
    if(dot && !dot.classList.contains('cb')) dot.classList.remove('err');
  }catch(err){ console.error('SSE parse error', err); }
});
es.onerror = ()=>{ const dot = g('status-dot'); if(dot) dot.className = 'brand-mark err'; };

fetch('/api/data').then(r=>r.json()).then(applyUpdate).catch(console.error);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    r = HTMLResponse(content=DASHBOARD_HTML)
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


if __name__ == "__main__":
    # Env overrides let a sandbox instance run beside production and allow
    # locking the bind address down (e.g. DASHBOARD_BIND=127.0.0.1) without
    # code changes. Defaults preserve current behaviour.
    _bind = _os.getenv("DASHBOARD_BIND", DASHBOARD_HOST).strip() or DASHBOARD_HOST
    _port = int(_os.getenv("DASHBOARD_PORT", str(DASHBOARD_PORT)))
    print(f"\n⚡ Dashboard  →  http://{_bind}:{_port}"
          f"  (auth: {'token required' if _AUTH_TOKEN else 'OPEN — set DASHBOARD_AUTH_TOKEN'})\n")
    uvicorn.run("dashboard:app", host=_bind, port=_port,
                log_level="warning", reload=False)
