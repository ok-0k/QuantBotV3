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
    get_filled_trade_count,
    get_portfolio_stat,
    get_recent_trades,
    get_trades_last_7_days,
    init_db,
    load_brain_key,
    set_portfolio_stat,
)
from accounting_v2 import position_equity_components

# ── Vancouver timezone ─────────────────────────────────────────────────────────
VAN_TZ = zoneinfo.ZoneInfo("America/Vancouver")


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
    """LTTB-inspired uniform downsampler: keep at most max_points from the curve."""
    n = len(curve)
    if n <= max_points:
        return curve
    step = n / max_points
    out = [curve[0]]
    for i in range(1, max_points - 1):
        idx = int(i * step)
        out.append(curve[idx])
    out.append(curve[-1])
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

    total_net_pnl = round(total_eq - STARTING_CASH, 2)
    return_pct    = round(total_net_pnl / STARTING_CASH * 100, 3)

    # ── Peak equity ────────────────────────────────────────────────────────────
    stored_peak      = float(get_portfolio_stat("peak_equity_all_time", str(STARTING_CASH)) or STARTING_CASH)
    curve_equity_vals = [float(pt.get("equity", 0.0) or 0.0) for pt in equity_curve_raw]
    peak_equity      = max(stored_peak, total_eq, STARTING_CASH, *curve_equity_vals)
    if peak_equity > stored_peak:
        set_portfolio_stat("peak_equity_all_time", round(peak_equity, 6))
    drawdown_pct = round((total_eq - peak_equity) / peak_equity * 100, 3) if peak_equity > 0 else 0.0

    set_portfolio_stat("current_equity", round(total_eq, 6))
    set_portfolio_stat("return_pct",     round(return_pct, 6))

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
        # Use all available data but downsample heavily
        equity_window = equity_curve_raw
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
            "starting_cash":         round(STARTING_CASH, 2),
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
<title>QuantBot — Command Centre</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ── DESIGN TOKENS ──────────────────────────────────────────────────────────── */
:root{
  --font-ui:      'IBM Plex Mono', monospace;
  --font-mono:    'IBM Plex Mono', monospace;
  --font-display: 'Syne', sans-serif;

  --bg-base:     #080c18;
  --bg-surface:  #0d1123dd;   /* semi-transparent for glass effect */
  --bg-elevated: #141828ee;
  --bg-border:   #2a3050;
  --bg-hover:    #151c30cc;
  --accent:      #00d4ff;
  --accent-dim:  rgba(0,212,255,.12);

  --pos:  #00e676;
  --neg:  #ff3d57;
  --warn: #ffc107;

  /* Fix 2: high-contrast typography — previous values (#4e4e60, #25252f)
     were nearly invisible on the dark background. */
  --t1:  #ffffff;    /* primary: crisp white */
  --t2:  #a0aec0;    /* secondary: readable silver */
  --t3:  #4a5568;    /* dim: visible but de-emphasised */
  --t4:  #2d3748;    /* borders / very dim */

  /* Glassmorphism */
  --glass-bg:     rgba(13,17,35,0.75);
  --glass-border: rgba(255,255,255,0.07);
  --glass-blur:   blur(10px);
}

*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
html{height:100%;overflow-x:hidden;}
body{
  background:var(--bg-base);
  color:var(--t2);
  font-family:var(--font-ui);
  font-size:12px;line-height:1.5;
  min-height:100vh;overflow-x:hidden;
}
::-webkit-scrollbar{width:2px;height:2px;}
::-webkit-scrollbar-track{background:var(--bg-base);}
::-webkit-scrollbar-thumb{background:var(--bg-border);}

@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.2}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.08}}
@keyframes scroll-ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}

/* ── HEADER ─────────────────────────────────────────────────────────────────── */
.dashboard-header{
  position:sticky;top:0;z-index:100;
  height:34px;
  background:var(--bg-base);
  border-bottom:1px solid var(--bg-border);
  display:flex;align-items:stretch;
}
.hdr-brand{
  display:flex;align-items:center;gap:8px;
  padding:0 14px;
  border-right:1px solid var(--bg-border);
  flex-shrink:0;
}
.brand-dot{
  width:5px;height:5px;border-radius:50%;
  background:var(--pos);
  animation:pulse-dot 2.5s ease infinite;
  flex-shrink:0;
}
.brand-dot.err{background:var(--neg);animation:blink 0.8s infinite;}
.brand-name{
  font-family:var(--font-display);font-size:12px;font-weight:800;
  letter-spacing:.12em;text-transform:uppercase;color:var(--t1);
  white-space:nowrap;
}
.brand-ver{
  font-family:var(--font-mono);font-size:7px;font-weight:600;
  letter-spacing:.1em;padding:1px 5px;
  border:1px solid var(--accent);color:var(--accent);
  text-transform:uppercase;white-space:nowrap;
  opacity:.7;
}
.hdr-status{
  display:flex;align-items:stretch;
  flex:1;min-width:0;overflow:hidden;
}
.hdr-pill{
  font-family:var(--font-mono);font-size:9px;font-weight:500;
  letter-spacing:.08em;padding:0 12px;
  text-transform:uppercase;
  border-right:1px solid var(--bg-border);
  white-space:nowrap;display:flex;align-items:center;flex-shrink:0;
}
.pill-regime{color:var(--t2);}
.pill-cb-ok{color:var(--t3);}
.pill-cb-open{color:var(--neg);background:rgba(255,61,87,.05);animation:blink 1.2s infinite;}
.hdr-ts{
  font-family:var(--font-mono);font-size:9px;color:var(--t3);
  font-variant-numeric:tabular-nums;padding:0 12px;
  display:flex;align-items:center;
}
.hdr-meta{
  display:flex;align-items:center;
  border-left:1px solid var(--bg-border);padding:0 12px;flex-shrink:0;
}
.hdr-tick{font-family:var(--font-mono);font-size:9px;color:var(--t3);white-space:nowrap;}

/* ── TICKER ──────────────────────────────────────────────────────────────────── */
.ticker-wrap{
  height:22px;background:var(--bg-surface);
  border-bottom:1px solid var(--bg-border);
  overflow:hidden;display:flex;align-items:center;
}
.ticker-track{
  display:flex;white-space:nowrap;
  animation:scroll-ticker 120s linear infinite;will-change:transform;
}
.ticker-wrap:hover .ticker-track{animation-play-state:paused;}
.t-item{display:inline-flex;align-items:center;gap:5px;padding:0 12px;border-right:1px solid var(--bg-border);font-size:9px;}
.t-sym{font-family:var(--font-mono);color:var(--t1);font-weight:600;letter-spacing:.4px;}
.t-price{font-family:var(--font-mono);color:var(--t2);font-variant-numeric:tabular-nums;}
.t-up{color:var(--pos);}.t-dn{color:var(--neg);}.t-flat{color:var(--t3);}

/* ── MAIN ─────────────────────────────────────────────────────────────────────── */
.main{padding:8px 12px 48px;max-width:1920px;margin:0 auto;}

/* ── KPI GRID ─────────────────────────────────────────────────────────────────── */
.kpi-grid{
  display:grid;
  grid-template-columns:1.7fr repeat(4,1fr);
  gap:1px;background:var(--bg-border);
  border:1px solid var(--bg-border);margin-bottom:7px;
}
.kpi-winrate{grid-column:2/4;}
.kpi-exposure{grid-column:1/-1;}

.kpi-card{
  background:var(--bg-surface);padding:9px 11px;
  position:relative;min-height:68px;
  transition:background .15s;
}
.kpi-card:hover{background:var(--bg-hover);}
.kpi-equity{background:var(--bg-elevated);}
.kpi-equity:hover{background:var(--bg-elevated);}
.kpi-drawdown{border-top:2px solid var(--warn);}

.kpi-label{
  font-family:var(--font-mono);font-size:8px;font-weight:500;
  letter-spacing:.12em;text-transform:uppercase;
  color:var(--t3);margin-bottom:4px;
  display:flex;align-items:center;justify-content:space-between;
}
.kpi-badge{
  font-family:var(--font-mono);font-size:7px;font-weight:600;
  padding:1px 4px;border:1px solid var(--pos);color:var(--pos);
  letter-spacing:.1em;text-transform:uppercase;animation:pulse-dot 3s infinite;
}
.kpi-value{
  font-family:var(--font-mono);
  font-size:clamp(16px,2vw,22px);font-weight:600;
  line-height:1.1;color:var(--t1);
  font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1;display:block;
}
.kpi-sub{
  font-family:var(--font-mono);font-size:10px;color:var(--t3);
  font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1;
  margin-top:3px;display:block;
}
.c-pos{color:var(--pos)!important;}.c-neg{color:var(--neg)!important;}
.c-warn{color:var(--warn)!important;}.c-accent{color:var(--accent)!important;}
.c-dim{color:var(--t3)!important;}

.wr-track{width:100%;height:1px;background:var(--bg-border);overflow:hidden;margin:4px 0 2px;}
.wr-fill{height:100%;background:var(--pos);width:0%;transition:width .3s;}

.pf-footer{display:flex;align-items:center;gap:6px;margin-top:4px;}
.pf-item{font-family:var(--font-mono);font-size:10px;font-variant-numeric:tabular-nums;}
.pf-item em{font-style:normal;font-size:8px;color:var(--t3);margin-right:3px;text-transform:uppercase;letter-spacing:.08em;}
.pf-divider{color:var(--bg-border);font-size:12px;}

/* Exposure bar */
.kpi-exposure{background:var(--bg-surface);padding:7px 11px;}
.kpi-exposure:hover{background:var(--bg-hover);}
.exp-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;margin-bottom:6px;}
.exp-stats{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.exp-stat{font-family:var(--font-mono);font-size:10px;font-variant-numeric:tabular-nums;}
.exp-stat em{font-style:normal;font-size:8px;color:var(--t3);margin-right:3px;text-transform:uppercase;letter-spacing:.08em;}
.exp-stat strong{font-weight:500;color:var(--t1);}
.exp-sep{color:var(--bg-border);font-size:11px;}
.exp-pct{font-family:var(--font-mono);font-size:10px;font-weight:600;color:var(--t2);font-variant-numeric:tabular-nums;}
.exp-track{height:2px;background:var(--bg-border);overflow:hidden;}
.exp-fill{height:100%;background:var(--accent);width:0%;transition:width .3s;}

/* ── ACCOUNT STRIP ────────────────────────────────────────────────────────────── */
.acct-strip{
  display:flex;align-items:stretch;
  background:var(--bg-surface);border:1px solid var(--bg-border);
  margin-bottom:7px;font-variant-numeric:tabular-nums;overflow:hidden;flex-wrap:wrap;
}
.acct-strip>*{padding:5px 10px;border-right:1px solid var(--bg-border);}
.acct-strip>*:last-child{border-right:none;}
.acct-lbl{font-family:var(--font-mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;white-space:nowrap;flex-shrink:0;display:flex;align-items:center;}
.acct-ok{font-family:var(--font-mono);color:var(--pos);font-size:10px;display:flex;align-items:center;}
.acct-fail{font-family:var(--font-mono);color:var(--neg);font-size:10px;display:flex;align-items:center;}
.acct-val{font-family:var(--font-mono);font-size:10px;display:flex;align-items:center;}

/* ── SESSION ROW ──────────────────────────────────────────────────────────────── */
.sess-row{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--bg-border);border:1px solid var(--bg-border);margin-bottom:7px;
}
.sess-cell{background:var(--bg-surface);padding:7px 11px;transition:background .15s;}
.sess-cell:hover{background:var(--bg-hover);}
.sess-lbl{font-family:var(--font-mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;}
.sess-val{font-family:var(--font-mono);font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1;margin-bottom:2px;color:var(--t1);}
.sess-sub{font-family:var(--font-mono);font-size:9px;color:var(--t3);}

/* ── RISK / MARGIN ROW ────────────────────────────────────────────────────────── */
.risk-row{
  background:var(--bg-surface);border:1px solid var(--bg-border);
  padding:5px 11px;display:flex;align-items:center;gap:10px;margin-bottom:7px;
}
.risk-lbl{font-family:var(--font-mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;white-space:nowrap;flex-shrink:0;}
.margin-info{font-family:var(--font-mono);font-size:10px;color:var(--t2);flex:1;font-variant-numeric:tabular-nums;}

/* ── FEE DRAG CARD ────────────────────────────────────────────────────────────── */
.fee-bar{
  display:grid;grid-template-columns:1fr 1fr 1fr;
  gap:1px;background:var(--bg-border);border:1px solid var(--bg-border);margin-bottom:7px;
}
.fee-cell{background:var(--bg-surface);padding:7px 11px;transition:background .15s;}
.fee-cell:hover{background:var(--bg-hover);}
.fee-lbl{font-family:var(--font-mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;}
.fee-val{font-family:var(--font-mono);font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--t1);}
.fee-sub{font-family:var(--font-mono);font-size:9px;color:var(--t3);margin-top:2px;}

/* ── CHARTS ─────────────────────────────────────────────────────────────────────── */
/* hero chart — full-width band directly below the KPI grid */
.hero-chart-card{
  background:var(--glass-bg);
  border:1px solid var(--glass-border);
  border-top:2px solid rgba(0,229,255,0.18);
  padding:10px 13px;
  margin-bottom:7px;
  backdrop-filter:var(--glass-blur);
  -webkit-backdrop-filter:var(--glass-blur);
}
.hero-chart-inner{height:180px;position:relative;}
.hero-chart-inner canvas{max-height:180px;}
/* legacy .chart-card kept for any surviving references */
.chart-card{background:var(--bg-surface);padding:9px 11px;}
.chart-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;flex-wrap:wrap;gap:6px;}
.chart-title{font-family:var(--font-mono);font-size:8px;font-weight:500;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;}
.chart-badge{font-family:var(--font-mono);font-size:9px;font-weight:600;padding:1px 6px;border:1px solid var(--bg-border);font-variant-numeric:tabular-nums;}
.chart-inner{height:155px;position:relative;}
.chart-inner canvas{max-height:155px;}

/* Timeframe buttons */
.tf-btns{display:flex;gap:2px;flex-wrap:wrap;}
.tf-btn{
  font-family:var(--font-mono);font-size:8px;font-weight:500;
  letter-spacing:.06em;padding:2px 6px;
  border:1px solid var(--bg-border);background:transparent;
  color:var(--t3);cursor:pointer;transition:all .15s;
}
.tf-btn:hover{border-color:var(--accent);color:var(--accent);}
.tf-btn.active{border-color:var(--accent);color:var(--accent);background:var(--accent-dim);}

/* ── TIME-BASED PERFORMANCE ───────────────────────────────────────────────────── */
.tbp-section{margin-bottom:7px;}
.tbp-hdr{padding:4px 0;margin-bottom:4px;}
.tbp-title{font-family:var(--font-mono);font-size:8px;font-weight:500;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;}
.tbp-grid{
  display:grid;grid-template-columns:1fr 1fr;
  gap:1px;background:var(--bg-border);border:1px solid var(--bg-border);margin-bottom:4px;
}
.tbp-kpi{background:var(--bg-surface);padding:8px 11px;transition:background .15s;}
.tbp-kpi:hover{background:var(--bg-hover);}
.tbp-kpi-lbl{font-family:var(--font-mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;}
.tbp-kpi-val{font-family:var(--font-mono);font-size:16px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--t1);}
.tbp-kpi-sub{font-family:var(--font-mono);font-size:9px;color:var(--t3);margin-top:2px;}

/* Weekly breakdown table */
.week-wrap{background:var(--bg-surface);border:1px solid var(--bg-border);overflow:hidden;}
.week-table{width:100%;border-collapse:collapse;}
.week-table th{
  text-align:right;font-family:var(--font-mono);font-size:7px;font-weight:500;
  color:var(--t3);text-transform:uppercase;letter-spacing:.12em;
  padding:6px 10px;border-bottom:1px solid var(--bg-border);background:var(--bg-elevated);
}
.week-table th.left{text-align:left;}
.week-table td{
  padding:6px 10px;border-bottom:1px solid var(--bg-border);
  font-family:var(--font-mono);font-size:10px;vertical-align:middle;text-align:right;
  font-variant-numeric:tabular-nums;color:var(--t2);
}
.week-table td.left{text-align:left;}
.week-table tbody tr:last-child td{border-bottom:none;}
.week-table tbody tr:hover td{background:var(--bg-elevated);}
.week-table .today-row td{background:var(--bg-elevated);}
.week-table .today-row td.day-cell{color:var(--accent);}
.day-bar-wrap{display:flex;align-items:center;gap:6px;}
.day-bar-track{flex:1;height:2px;background:var(--bg-border);overflow:hidden;}
.day-bar-fill{height:100%;}

/* ── DATA TABLES ─────────────────────────────────────────────────────────────── */
/* Fix 4: high-density, minimalist table aesthetic for desktop.
   Thin dividers, compact padding, clean hover glow — no heavy borders. */
.tbl-section{margin-bottom:7px;}
.tbl-hdr{display:flex;align-items:center;gap:8px;padding:4px 0;}
.tbl-title{font-family:var(--font-mono);font-size:8px;font-weight:500;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;}
.tbl-count{font-family:var(--font-mono);font-size:8px;padding:1px 6px;border:1px solid var(--bg-border);background:var(--bg-surface);color:var(--t3);}
.tbl-wrap{background:var(--bg-surface);border:1px solid var(--bg-border);overflow:hidden;overflow-x:auto;-webkit-overflow-scrolling:touch;}
table{width:100%;border-collapse:collapse;min-width:700px;}
th{
  text-align:right;font-family:var(--font-mono);font-size:7px;font-weight:600;
  color:var(--t3);text-transform:uppercase;letter-spacing:.14em;
  padding:5px 8px;border-bottom:1px solid var(--bg-border);
  white-space:nowrap;background:rgba(20,24,40,0.8);
}
th.left{text-align:left;}
td{
  padding:4px 8px;
  border-bottom:1px solid rgba(42,48,80,0.5);  /* hairline divider */
  font-family:var(--font-mono);font-size:9.5px;vertical-align:middle;
  white-space:nowrap;text-align:right;
  font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1;
  color:var(--t2);transition:background .08s;
}
td.left{text-align:left;}
tbody tr:last-child td{border-bottom:none;}
tbody tr:hover td{
  background:rgba(0,212,255,0.04);
  border-bottom-color:rgba(0,212,255,0.08);
}
tbody tr.row-long  td:first-child{border-left:2px solid rgba(0,230,118,.25);padding-left:7px;}
tbody tr.row-short td:first-child{border-left:2px solid rgba(255,61,87,.25);padding-left:7px;}
.empty-row td{text-align:center;color:var(--t3);padding:18px;font-size:10px;}
.sym-cell{font-family:var(--font-mono);font-weight:600;color:var(--t1);letter-spacing:.5px;}
.side-badge{display:inline-flex;align-items:center;gap:2px;padding:1px 5px;font-family:var(--font-mono);font-size:7px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;border:1px solid;}
.side-long{color:var(--pos);border-color:rgba(0,230,118,.2);}
.side-short{color:var(--neg);border-color:rgba(255,61,87,.2);}
.chip{display:inline-block;padding:1px 5px;font-family:var(--font-mono);font-size:7px;font-weight:600;letter-spacing:.06em;border:1px solid;}
.chip-buy{color:var(--pos);border-color:rgba(0,230,118,.2);}
.chip-sell{color:var(--neg);border-color:rgba(255,61,87,.2);}
.chip-short{color:var(--neg);border-color:rgba(255,61,87,.25);}
.chip-cover{color:var(--warn);border-color:rgba(255,193,7,.25);}
.strat-tag{font-family:var(--font-mono);font-size:9px;color:var(--t2);font-weight:400;max-width:120px;display:inline-block;overflow:hidden;text-overflow:ellipsis;}
.pnl-pos{color:var(--pos);font-weight:600;}.pnl-neg{color:var(--neg);font-weight:600;}.pnl-zero{color:var(--t3);}
.rr-val{color:var(--warn);font-weight:600;}
.hold-warn{color:var(--warn);}.hold-danger{color:var(--neg);}
.time-cell{line-height:1.5;}
.time-dur{font-family:var(--font-mono);font-weight:600;color:var(--warn);font-size:9px;}
.time-ts{font-family:var(--font-mono);font-size:8px;color:var(--t3);display:block;}

/* MFE/MAE cells */
.mfe-val{color:var(--pos);font-size:9px;}
.mae-val{color:var(--neg);font-size:9px;}
.mfe-dash,.mae-dash{color:var(--t3);}

/* ── COLLAPSIBLE ─────────────────────────────────────────────────────────────── */
details.sect{margin-bottom:7px;}
details.sect>summary{list-style:none;display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;padding:4px 0;}
details.sect>summary::-webkit-details-marker{display:none;}
details.sect>summary .sect-title{font-family:var(--font-mono);font-size:8px;font-weight:500;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;}
details.sect>summary:hover .sect-title{color:var(--t2);}
details.sect>summary .sect-arrow{font-size:8px;color:var(--t3);margin-left:auto;}
details.sect>.sect-body{padding-top:4px;}

/* ── PRICES GRID ─────────────────────────────────────────────────────────────── */
.prices-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:1px;background:var(--bg-border);}
.price-card{background:var(--bg-surface);padding:6px 9px;transition:background .15s;}
.price-card:hover{background:var(--bg-hover);}
.price-card.active{border-top:1px solid rgba(0,212,255,.35);}
.pc-sym{font-family:var(--font-mono);font-weight:600;font-size:9px;color:var(--t1);margin-bottom:2px;letter-spacing:.3px;}
.pc-price{font-family:var(--font-mono);font-size:10px;color:var(--t2);font-variant-numeric:tabular-nums;}
.pc-chg{font-family:var(--font-mono);font-size:9px;font-variant-numeric:tabular-nums;}
.pc-up{color:var(--pos);}.pc-dn{color:var(--neg);}.pc-flat{color:var(--t3);}

/* ── GLASSMORPHISM PANELS ────────────────────────────────────────────────── */
.kpi-card,.chart-card,.hero-chart-card,.sess-cell,.fee-cell,.tbp-kpi,.week-wrap,.tbl-wrap,
.acct-strip,.risk-row{
  backdrop-filter:var(--glass-blur);
  -webkit-backdrop-filter:var(--glass-blur);
}
.kpi-card{background:var(--glass-bg);border:1px solid var(--glass-border);}
/* ═══════════════════════════════════════════════════════════════════════════
   EQUITY + CASH-FLOW CHART CARD  (4-fix overhaul)
   ═══════════════════════════════════════════════════════════════════════════ */

/* Fix 4: Card depth — gradient top border + layered shadow for elevation */
.kpi-equity{
  grid-column:1/-1!important;
  display:flex!important;
  padding:0!important;
  min-height:230px;

  /* background */
  background:rgba(6,10,22,0.82)!important;

  /* Fix 4: crisp border + neon fade box-shadow */
  border:1px solid rgba(0,229,255,0.20)!important;
  border-top:3px solid transparent!important;
  border-image:linear-gradient(90deg,#00E5FF 0%,#7000FF 100%) 1!important;
  border-image-slice:1!important;
  box-shadow:
    0 6px 32px rgba(0,0,0,0.55),
    0 1px 0   rgba(0,229,255,0.12),
    inset 0 0 60px rgba(0,229,255,0.02)!important;
  overflow:hidden;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
}

/* Fix 2: Left column — generous padding, clear right-side breathing room */
.kpi-eq-stats{
  display:flex;flex-direction:column;justify-content:center;
  padding:20px 28px 20px 20px;   /* 28px right creates clear separation */
  min-width:175px;max-width:210px;flex-shrink:0;
  border-right:1px solid rgba(0,229,255,0.09);
  gap:2px;
}

/* Fix 3: Badge repositioned as a clean "Free Cash" data row inside left column */
.kpi-eq-divider{
  height:1px;background:rgba(255,255,255,0.05);
  margin:8px 0 6px;
}
.kpi-eq-cash-row{
  display:flex;flex-direction:column;gap:1px;
}
.kpi-eq-cash-lbl{
  font-family:var(--font-mono);font-size:7px;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;color:var(--t3);
}
.kpi-eq-cash-val{
  font-family:var(--font-mono);font-size:13px;font-weight:600;
  color:var(--t2);font-variant-numeric:tabular-nums;
}

/* Fix 2: Right column — comfortable internal padding, flex column */
.kpi-eq-chart{
  flex:1;min-width:0;
  display:flex;flex-direction:column;
  padding:14px 16px 12px 14px;
  gap:10px;
}

/* Fix 1: TF toolbar — its own isolated row, never touching the canvas */
.cash-tf-toolbar{
  display:flex;align-items:center;flex-wrap:wrap;
  gap:4px;
  flex-shrink:0;
  padding-bottom:2px;
  border-bottom:1px solid rgba(255,255,255,0.04);
}

/* Fix 1: Larger, more legible TF buttons with comfortable touch targets */
.cash-tf-btn{
  font-family:var(--font-mono);
  font-size:11px;
  font-weight:500;
  letter-spacing:.04em;
  padding:5px 11px;
  border-radius:4px;
  border:1px solid rgba(255,255,255,0.10);
  background:rgba(255,255,255,0.04);
  color:rgba(160,174,192,0.8);   /* silver, not invisible grey */
  cursor:pointer;
  white-space:nowrap;
  transition:color .12s,border-color .12s,background .12s,box-shadow .12s;
  line-height:1.2;
}
.cash-tf-btn:hover{
  color:var(--t1);
  border-color:rgba(255,255,255,0.28);
  background:rgba(255,255,255,0.08);
}
/* Fix 1: Active state — cyan glow, clearly distinguishable */
.cash-tf-btn.active{
  color:#00E5FF!important;
  border-color:#00E5FF!important;
  background:rgba(0,229,255,0.10)!important;
  box-shadow:0 0 10px rgba(0,229,255,0.28),inset 0 0 8px rgba(0,229,255,0.06)!important;
}

/* Canvas wrapper — fills all remaining height after toolbar */
.kpi-eq-canvas-wrap{
  flex:1;position:relative;min-height:120px;
}
.kpi-eq-canvas-wrap canvas{
  position:absolute;inset:0;width:100%!important;height:100%!important;
}

/* hero-chart-card: removed from DOM, CSS kept safe */
.hero-chart-card,.hero-chart-inner{display:none;}
.chart-card{background:var(--glass-bg);border:1px solid var(--glass-border);}
.sess-cell{background:var(--glass-bg);}
.fee-cell{background:var(--glass-bg);}
.tbp-kpi{background:var(--glass-bg);}
.week-wrap{background:var(--glass-bg);}
.tbl-wrap{background:rgba(8,12,24,0.6);border-color:var(--glass-border);}

/* Fix 2: trade row PnL glow */
tbody tr.trade-win  td{background:rgba(0,230,118,0.04);}
tbody tr.trade-win:hover td{background:rgba(0,230,118,0.08);}
tbody tr.trade-loss td{background:rgba(255,61,87,0.04);}
tbody tr.trade-loss:hover td{background:rgba(255,61,87,0.08);}

/* ── FIX 5: HEARTBEAT ───────────────────────────────────────────────────── */
.brand-dot{
  width:7px;height:7px;border-radius:50%;
  background:var(--pos);
  animation:pulse-dot 2.5s ease infinite;
  box-shadow:0 0 5px 1px rgba(0,230,118,.5);
  flex-shrink:0;
}
.brand-dot.err{
  background:var(--neg);animation:none;
  box-shadow:0 0 7px 2px rgba(255,61,87,.6);
}
.brand-dot.cb-open{
  background:var(--warn);animation:blink 1.0s infinite;
  box-shadow:0 0 7px 2px rgba(255,193,7,.5);
}

/* ── FIX 3: HUD ─────────────────────────────────────────────────────────── */
.hud{
  position:sticky;top:34px;z-index:99;
  display:flex;align-items:stretch;
  background:rgba(8,12,24,0.94);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border-bottom:1px solid var(--glass-border);
  border-top:1px solid rgba(0,212,255,0.10);
  height:54px;overflow:hidden;
}
.hud-item{
  display:flex;flex-direction:column;justify-content:center;
  padding:0 18px;border-right:1px solid var(--glass-border);flex-shrink:0;
}
.hud-item:last-child{border-right:none;flex:1;}
.hud-lbl{
  font-family:var(--font-mono);font-size:7px;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;color:var(--t3);
  margin-bottom:2px;
}
.hud-val{
  font-family:var(--font-display);font-size:clamp(15px,2vw,22px);
  font-weight:700;line-height:1;font-variant-numeric:tabular-nums;
}
.hud-sub{
  font-family:var(--font-mono);font-size:9px;color:var(--t3);
  font-variant-numeric:tabular-nums;margin-top:1px;
}
.hud-ai-bar{
  height:3px;background:var(--bg-border);border-radius:2px;
  margin-top:4px;overflow:hidden;width:min(130px,100%);
}
.hud-ai-fill{height:100%;border-radius:2px;transition:width .5s,background .5s;}

/* ── RESPONSIVE ─────────────────────────────────────────────────────────────── */
@media(min-width:640px) and (max-width:1023px){
  .kpi-grid{grid-template-columns:repeat(3,1fr);}
  .kpi-winrate{grid-column:1/3;}
  .kpi-exposure{grid-column:1/-1;}
  .sess-row{grid-template-columns:repeat(2,1fr);}
  .tbp-grid{grid-template-columns:1fr 1fr;}
  .fee-bar{grid-template-columns:1fr 1fr;}
}
@media(max-width:639px){
  html,body{overflow-x:hidden;}
  .kpi-grid{grid-template-columns:repeat(2,1fr);}
  .kpi-winrate{grid-column:1/-1;}
  .kpi-exposure{grid-column:1/-1;}
  .kpi-card{padding:8px 9px;min-height:58px;}
  .sess-row{grid-template-columns:repeat(2,1fr);}
  .tbp-grid{grid-template-columns:1fr;}
  .fee-bar{grid-template-columns:1fr;}
  .main{padding:5px 7px 24px;}
  .dashboard-header{height:38px;}
  .hud{top:38px;height:auto;flex-wrap:wrap;}
  .hud-item{padding:8px 14px;border-bottom:1px solid var(--glass-border);border-right:none;flex:unset;}
  .hdr-pill.pill-regime,.hdr-pill.pill-cb-ok,.hdr-ts{display:none;}
  .hdr-pill.pill-cb-open{display:flex!important;}
  /* Stack equity card vertically on mobile */
  .kpi-equity{flex-direction:column;min-height:unset;}
  .kpi-eq-stats{
    max-width:unset;padding:16px 18px 12px;
    border-right:none;border-bottom:1px solid rgba(0,229,255,0.08);
  }
  .kpi-eq-chart{padding:10px 14px 12px;}
  .cash-tf-btn{font-size:10px;padding:5px 9px;}
  .kpi-eq-canvas-wrap{min-height:140px;}
}

/* ── FIX 4: MOBILE TRADE CARD STACK (@media ≤768px) ─────────────────────── */
@media(max-width:768px){
  /* Trade history: collapse rows into vertical cards */
  #trades-body tr{
    display:block;
    background:var(--glass-bg);
    border:1px solid var(--glass-border);
    border-radius:6px;margin-bottom:6px;padding:6px 10px;
    backdrop-filter:var(--glass-blur);
  }
  #trades-body tr.trade-win{border-color:rgba(0,230,118,.22);}
  #trades-body tr.trade-loss{border-color:rgba(255,61,87,.22);}
  /* Hide all tds; show only those with data-label */
  #trades-body td{display:none;}
  #trades-body td[data-label]{
    display:flex;align-items:center;justify-content:space-between;
    padding:3px 0;border:none;text-align:right;
    font-size:10px;font-variant-numeric:tabular-nums;
  }
  #trades-body td[data-label]::before{
    content:attr(data-label);
    font-size:7.5px;color:var(--t3);text-transform:uppercase;
    letter-spacing:.10em;margin-right:10px;flex-shrink:0;
  }
  /* NET PNL cell: full width, huge, centered */
  #trades-body td[data-label="NET PNL"]{
    justify-content:center;flex-direction:column;align-items:center;
    padding:10px 0 6px;
  }
  #trades-body td[data-label="NET PNL"]::before{display:none;}
  #trades-body td[data-label="NET PNL"] div:first-child{
    font-size:clamp(22px,6vw,30px)!important;font-weight:700;line-height:1.1;
  }
  /* Hide the wide-table header on mobile */
  .tbl-section .tbl-wrap table thead{display:none;}
  .tbl-section .tbl-wrap table{min-width:unset;}
  /* Keep positions table scrollable */
  #pos-body tr{display:table-row;}
  #pos-body td{display:table-cell;}
  .tbl-section:has(#pos-body) .tbl-wrap{overflow-x:auto;}
  .tbl-section:has(#pos-body) .tbl-wrap table{min-width:700px;}
  .tbl-section:has(#pos-body) .tbl-wrap table thead{display:table-header-group;}
}
</style>
</head>
<body>

<!-- ── HEADER ─────────────────────────────────────────────────────────────── -->
<header class="dashboard-header">
  <div class="hdr-brand">
    <span class="brand-dot" id="h-dot"></span>
    <span class="brand-name">QuantBot</span>
    <span class="brand-ver">EXEC</span>
  </div>
  <div class="hdr-status">
    <span class="hdr-pill pill-regime" id="h-regime">RANGING</span>
    <span class="hdr-pill pill-cb-ok"  id="h-cb">CB CLEAR</span>
    <span class="hdr-ts" id="h-ts">—</span>
  </div>
  <div class="hdr-meta">
    <span class="hdr-tick" id="h-tick">TICK —</span>
  </div>
</header>

<!-- ── FIX 3: STICKY HUD ───────────────────────────────────────────────────── -->
<div class="hud" id="hud">
  <div class="hud-item">
    <span class="hud-lbl">Today&#39;s Net P&amp;L</span>
    <span class="hud-val" id="hud-pnl">—</span>
    <span class="hud-sub" id="hud-pnl-pct">—</span>
  </div>
  <div class="hud-item">
    <span class="hud-lbl">Max Drawdown</span>
    <span class="hud-val" id="hud-dd">—</span>
  </div>
  <div class="hud-item">
    <span class="hud-lbl">Profit Factor</span>
    <span class="hud-val" id="hud-pf">—</span>
  </div>
  <div class="hud-item">
    <span class="hud-lbl">Total Fees Paid</span>
    <span class="hud-val" id="hud-fees">—</span>
  </div>
  <div class="hud-item hud-ai">
    <span class="hud-lbl">AI Sizing Penalty</span>
    <span class="hud-val" id="hud-sizing">—</span>
    <div class="hud-ai-bar"><div class="hud-ai-fill" id="hud-ai-fill" style="width:100%;background:var(--pos)"></div></div>
  </div>
</div>

<!-- ── TICKER TAPE ─────────────────────────────────────────────────────────── -->
<div class="ticker-wrap">
  <div class="ticker-track" id="ticker-track">
    <div class="t-item"><span class="t-sym">CONNECTING</span><span class="t-flat">—</span></div>
  </div>
</div>

<div class="main">

  <!-- ── KPI GRID ──────────────────────────────────────────────────────────── -->
  <div class="kpi-grid">
    <div class="kpi-card kpi-equity">

      <!-- ── Left column: equity metrics ───────────────────────────────────── -->
      <div class="kpi-eq-stats">
        <div class="kpi-label">Portfolio Equity<span class="kpi-badge">LIVE</span></div>
        <span class="kpi-value" id="kpi-eq" style="color:#00E5FF">$—</span>
        <span class="kpi-sub" id="kpi-peak-sub">Peak  —</span>
        <div class="kpi-eq-divider"></div>
        <div class="kpi-eq-cash-row">
          <span class="kpi-eq-cash-lbl">Free Cash</span>
          <span class="kpi-eq-cash-val" id="cash-badge">—</span>
        </div>
      </div>

      <!-- ── Right column: TF toolbar + chart ──────────────────────────────── -->
      <div class="kpi-eq-chart">

        <!-- TF toolbar — its own isolated row, never mixed with canvas -->
        <div class="cash-tf-toolbar" id="cash-tf-btns">
          <button class="cash-tf-btn" data-tf="10m">10m</button>
          <button class="cash-tf-btn" data-tf="15m">15m</button>
          <button class="cash-tf-btn" data-tf="30m">30m</button>
          <button class="cash-tf-btn" data-tf="1H">1H</button>
          <button class="cash-tf-btn" data-tf="2H">2H</button>
          <button class="cash-tf-btn" data-tf="3H">3H</button>
          <button class="cash-tf-btn" data-tf="5H">5H</button>
          <button class="cash-tf-btn" data-tf="12H">12H</button>
          <button class="cash-tf-btn active" data-tf="24H">24H</button>
          <button class="cash-tf-btn" data-tf="2D">2D</button>
          <button class="cash-tf-btn" data-tf="7D">7D</button>
          <button class="cash-tf-btn" data-tf="14D">14D</button>
          <button class="cash-tf-btn" data-tf="1M">1M</button>
          <button class="cash-tf-btn" data-tf="2M">2M</button>
          <button class="cash-tf-btn" data-tf="3M">3M</button>
        </div>

        <!-- Canvas — isolated from toolbar, fills remaining height -->
        <div class="kpi-eq-canvas-wrap"><canvas id="cashChart"></canvas></div>

      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Free Cash</div>
      <span class="kpi-value" id="kpi-cash">$—</span>
      <span class="kpi-sub" id="kpi-cash-pct">—% uninvested</span>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Return</div>
      <span class="kpi-value c-neg" id="kpi-ret">—%</span>
      <span class="kpi-sub" id="kpi-ret-start">—</span>
    </div>
    <div class="kpi-card kpi-drawdown">
      <div class="kpi-label">Drawdown</div>
      <span class="kpi-value c-warn" id="kpi-dd">—%</span>
      <span class="kpi-sub" id="kpi-dd-sub">from peak</span>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Net PnL <span style="font-size:7px;text-transform:none;letter-spacing:.01em;font-weight:400">(total)</span></div>
      <span class="kpi-value" id="kpi-netpnl">$—</span>
      <span class="kpi-sub" id="kpi-netpnl-sub">realized + open</span>
    </div>

    <div class="kpi-card kpi-winrate">
      <div class="kpi-label">Win Rate</div>
      <span class="kpi-value" id="kpi-wr">—%</span>
      <div class="wr-track"><div class="wr-fill" id="wr-bar-fill"></div></div>
      <span class="kpi-sub" id="kpi-wr-sub">— trades (lifetime)</span>
      <div class="pf-footer" id="kpi-wr-side-breakdown" style="margin-top:4px">
        <span class="pf-item"><em>L</em><span id="kpi-long-wr" style="color:var(--pos)">—</span></span>
        <span class="pf-divider">|</span>
        <span class="pf-item"><em>S</em><span id="kpi-short-wr" style="color:var(--neg)">—</span></span>
      </div>
      <span id="kpi-trades" style="display:none">—</span>
      <span id="kpi-pos-sub" style="display:none">—</span>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Profit Factor</div>
      <span class="kpi-value c-neg" id="kpi-pf">—</span>
      <div class="pf-footer" id="kpi-pf-sub">
        <span class="pf-item"><em>GP</em><span class="c-pos">—</span></span>
        <span class="pf-divider">|</span>
        <span class="pf-item"><em>GL</em><span class="c-neg">—</span></span>
      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Expectancy</div>
      <span class="kpi-value" id="kpi-exp" style="font-size:clamp(12px,1.5vw,18px)">—</span>
      <span class="kpi-sub">avg W / avg L</span>
    </div>

    <div class="kpi-card kpi-exposure">
      <div class="exp-header">
        <div class="kpi-label" style="margin-bottom:0">Capital Exposure</div>
        <div class="exp-stats">
          <span class="exp-stat"><em>Deployed</em><strong id="exp-deployed">$0.00</strong></span>
          <span class="exp-sep">·</span>
          <span class="exp-stat"><em>Equity</em><strong id="exp-equity-val">—</strong></span>
          <span class="exp-sep">·</span>
          <span class="exp-pct" id="exp-pct">0.0%</span>
        </div>
      </div>
      <div class="exp-track"><div class="exp-fill" id="exp-fill"></div></div>
      <span id="exp-detail" style="display:none">$0 / $0</span>
    </div>
  </div><!-- /.kpi-grid -->

  <!-- ── ACCOUNT STRIP ──────────────────────────────────────────────────────── -->
  <div class="acct-strip">
    <span class="acct-lbl">Equity Proof</span>
    <span id="acct-id" class="acct-ok">—</span>
    <span class="acct-lbl" style="margin-left:auto">Realized</span>
    <span id="acct-realized" class="acct-val">—</span>
    <span class="acct-lbl">Open</span>
    <span id="acct-open" class="acct-val">—</span>
  </div>

  <!-- ── FEE DRAG ────────────────────────────────────────────────────────────── -->
  <div class="fee-bar" style="grid-template-columns:1fr 1fr 1fr 1fr">
    <div class="fee-cell">
      <div class="fee-lbl">Total Fees Paid</div>
      <div class="fee-val c-neg" id="fee-total">—</div>
      <div class="fee-sub">0.12% round-trip × notional</div>
    </div>
    <div class="fee-cell">
      <div class="fee-lbl">Net vs Gross Efficiency</div>
      <div class="fee-val" id="fee-eff">—</div>
      <div class="fee-sub" id="fee-eff-sub">net / gross PnL</div>
    </div>
    <div class="fee-cell">
      <div class="fee-lbl">Gross PnL (before fees)</div>
      <div class="fee-val" id="fee-gross">—</div>
      <div class="fee-sub" id="fee-gross-sub">fees drag: —</div>
    </div>
    <div class="fee-cell" style="border-left:2px solid var(--accent-dim)">
      <div class="fee-lbl">Lifetime Trades <span style="font-size:6px;opacity:.5">DB COUNT(*)</span></div>
      <div class="fee-val c-accent" id="lifetime-trades">—</div>
      <div class="fee-sub" id="lifetime-trades-sub">all filled executions</div>
    </div>
  </div>

  <!-- ── SESSION ROW ────────────────────────────────────────────────────────── -->
  <div class="sess-row">
    <div class="sess-cell">
      <div class="sess-lbl">Today&#39;s P&amp;L <span style="font-size:7px;opacity:.5">(PT)</span></div>
      <div class="sess-val" id="sess-daily-pnl">$—</div>
      <div class="sess-sub" id="sess-daily-trades">0 closed trades</div>
    </div>
    <div class="sess-cell">
      <div class="sess-lbl">Today&#39;s Win Rate</div>
      <div class="sess-val" id="sess-daily-wr">—%</div>
      <div class="sess-sub" id="sess-daily-wr-sub">— W / — T</div>
    </div>
    <div class="sess-cell">
      <div class="sess-lbl">Open Unrealized</div>
      <div class="sess-val" id="sess-unr">$—</div>
      <div class="sess-sub" id="sess-unr-sub">across — positions</div>
    </div>
    <div class="sess-cell">
      <div class="sess-lbl">Realized Net (all time)</div>
      <div class="sess-val" id="sess-realized">$—</div>
      <div class="sess-sub">closed trades only</div>
    </div>
  </div>

  <!-- ── MARGIN HEALTH ──────────────────────────────────────────────────────── -->
  <div class="risk-row">
    <span class="risk-lbl">Margin Health</span>
    <span class="margin-info" id="margin-info">—</span>
    <span class="hdr-pill" id="margin-halt"
      style="display:none;color:var(--neg);border-color:rgba(255,61,87,.35);background:rgba(255,61,87,.06)">
      ⚠ HALT
    </span>
  </div>

  <!-- ── TIME-BASED PERFORMANCE ─────────────────────────────────────────────── -->
  <div class="tbp-section">
    <div class="tbp-hdr"><span class="tbp-title">Rolling 7-Day Performance</span></div>
    <div class="tbp-grid">
      <div class="tbp-kpi">
        <div class="tbp-kpi-lbl">Today&#39;s PnL <span style="font-size:7px;opacity:.5">(Vancouver PT)</span></div>
        <div class="tbp-kpi-val" id="tbp-today-pnl">—</div>
        <div class="tbp-kpi-sub" id="tbp-today-sub">— trades · — W/R</div>
      </div>
      <div class="tbp-kpi">
        <div class="tbp-kpi-lbl">This Week&#39;s Total PnL</div>
        <div class="tbp-kpi-val" id="tbp-week-pnl">—</div>
        <div class="tbp-kpi-sub" id="tbp-week-sub">Mon → Sun</div>
      </div>
    </div>
    <div class="week-wrap">
      <table class="week-table">
        <thead><tr>
          <th class="left">Day</th>
          <th class="left">Date</th>
          <th>PnL</th>
          <th>Trades</th>
          <th>Win Rate</th>
          <th style="min-width:140px">Performance</th>
        </tr></thead>
        <tbody id="week-body">
          <tr><td colspan="6" style="text-align:center;color:var(--t3);padding:14px">Waiting for data…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── OPEN POSITIONS TABLE ───────────────────────────────────────────────── -->
  <div class="tbl-section">
    <div class="tbl-hdr">
      <span class="tbl-title">Open Positions</span>
      <span class="tbl-count" id="pos-count">0</span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th class="left">Symbol</th><th class="left">Side</th>
          <th>Qty</th><th>Entry</th><th>Mark</th>
          <th>Invested</th><th>%Eq</th>
          <th>uPnL</th><th>uPnL%</th>
          <th class="left">Strategy</th>
          <th>Stop</th><th>Take</th><th>R&#8758;R</th>
          <th>Candles</th><th>Opened (PT)</th>
        </tr></thead>
        <tbody id="pos-body">
          <tr class="empty-row"><td colspan="15">No open positions</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── TRADE HISTORY TABLE ────────────────────────────────────────────────── -->
  <div class="tbl-section">
    <div class="tbl-hdr">
      <span class="tbl-title">Trade History</span>
      <span class="tbl-count" id="trades-count">0</span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th class="left">Time (PT)</th><th class="left">Symbol</th>
          <th class="left">Side</th><th class="left">Action</th>
          <th class="left">Strategy</th><th>Regime</th>
          <th>Price</th><th>Exec</th>
          <th>Net PnL</th><th>Peak Profit (MFE)</th><th>Max DD (MAE)</th>
          <th>Fees</th><th>%Alloc</th>
        </tr></thead>
        <tbody id="trades-body">
          <tr class="empty-row"><td colspan="13">No trades yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── LIVE PRICES ───────────────────────────────────────────────────────── -->
  <details class="sect">
    <summary>
      <span class="sect-title">Live Market Prices</span>
      <span style="font-size:8px;color:var(--t3);margin-left:8px" id="prices-summary">—</span>
      <span class="sect-arrow">▼</span>
    </summary>
    <div class="sect-body">
      <div class="prices-grid" id="prices-grid"></div>
    </div>
  </details>

</div><!-- /.main -->

<script>
/* ── PALETTE ──────────────────────────────────────────────────────────────── */
const C={
  g:'#00e676',r:'#ff3d57',y:'#ffc107',b:'#00d4ff',
  dim:'#2d3748',dim2:'#4a5568',text:'#a0aec0'
};
const RC={trend_up:C.g,trending_up:C.g,trend_down:C.r,trending_down:C.r,ranging:C.b,volatile:C.y,neutral:C.dim2};

/* Fix 2: chart axis text was '#25252f' — invisible on dark canvas */
Chart.defaults.color='#a0aec0';
Chart.defaults.borderColor='#2a3050';
Chart.defaults.font={family:"'IBM Plex Mono',monospace",size:9};

let charts={},_openSyms=new Set(),_pricesData=[],_lastExpUpdate=0,_activeTf='24H',_lastEqFetch=0;

/* ── CASH CHART STATE ─────────────────────────────────────────────────────── */
let _cashData=[],_activeCashTf='24H',_lastCashFetch=0,_cashFetching=false;
const CASH_TF_MINS={
  '10m':10,'15m':15,'30m':30,
  '1H':60,'2H':120,'3H':180,'5H':300,'12H':720,'24H':1440,
  '2D':2880,'7D':10080,'14D':20160,'1M':43200,'2M':86400,'3M':129600
};
/* Minimum trade-count needed to usefully populate each TF window */
const CASH_TF_LIMIT={'10m':100,'15m':150,'30m':200,'1H':300,'2H':400,'3H':500,
  '5H':600,'12H':800,'24H':1000,'2D':1500,'7D':2500,'14D':4000,'1M':5000,'2M':7000,'3M':10000};

function _parseCashTs(ts){
  /* "YYYY-MM-DD HH:MM:SS" already in Vancouver — parse as local */
  return new Date(String(ts).replace(' ','T'));
}
function filterCashByTf(data,tf){
  const mins=CASH_TF_MINS[tf]||1440;
  const cutoff=new Date(Date.now()-mins*60*1000);
  const filtered=data.filter(d=>_parseCashTs(d.time)>=cutoff);
  return filtered.length>=2?filtered:data.slice(-2);/* always render something */
}
function setCashTf(tf){
  if(tf===_activeCashTf&&_cashData.length)return;
  _activeCashTf=tf;
  document.querySelectorAll('.cash-tf-btn').forEach(b=>{
    b.classList.toggle('active',b.dataset.tf===tf);
  });
  const needLimit=CASH_TF_LIMIT[tf]||1000;
  /* If we already have enough history, just re-filter in memory */
  if(_cashData.length>=needLimit){
    const filtered=filterCashByTf(_cashData,tf);
    renderCash(filtered);
    return;
  }
  /* Otherwise fetch more from the server */
  if(_cashFetching)return;
  _cashFetching=true;
  fetch('/api/cash?limit='+needLimit)
    .then(r=>r.json())
    .then(d=>{
      if(d.cash_curve&&d.cash_curve.length){
        _cashData=d.cash_curve;
        renderCash(filterCashByTf(_cashData,_activeCashTf));
      }
    })
    .catch(console.error)
    .finally(()=>{_cashFetching=false;});
}

const g=s=>document.getElementById(s);
const fmt$=(v,d=2)=>'$'+Number(v).toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d});
const fmtP=v=>(v>=0?'+':'')+Number(v).toFixed(3)+'%';
const fmtPnl=(v,d=2)=>v==null?'—':(v>=0?'+':'')+fmt$(v,d);
const fmtPrice=v=>{if(!v||v===0)return'—';if(v>=10000)return'$'+Number(v).toLocaleString('en',{minimumFractionDigits:0,maximumFractionDigits:0});if(v>=100)return'$'+Number(v).toFixed(2);if(v>=1)return'$'+Number(v).toFixed(4);return'$'+Number(v).toFixed(6);};
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const pnlCls=v=>v>0?'pnl-pos':v<0?'pnl-neg':'pnl-zero';
const pnlColor=v=>v>0?C.g:v<0?C.r:C.dim2;
const sideBadge=side=>(side||'long').toLowerCase()==='short'?'<span class="side-badge side-short">▼ SHORT</span>':'<span class="side-badge side-long">▲ LONG</span>';
const stratTag=name=>{if(!name)return'<span style="color:var(--t3)">—</span>';return`<span class="strat-tag" title="${esc(name)}">${esc(String(name))}</span>`;};

/* Vancouver-aware timestamp display — backend already converted; just display */
function fmtTs(ts){if(!ts)return'—';return String(ts).slice(0,16).replace('T',' ');}
function fmtDur(ts){
  if(!ts)return'—';
  /* ts is already Vancouver local — convert to approximate UTC offset for diff */
  const ms=Date.now()-new Date(ts).getTime();
  if(ms<0)return'—';
  const m=Math.floor(ms/60000),h=Math.floor(m/60),d=Math.floor(h/24);
  if(d>0)return d+'d '+(h%24)+'h';if(h>0)return h+'h '+(m%60)+'m';return m+'m';
}

/* Vancouver current time via Intl (DST-aware) */
function vanNow(){
  return new Intl.DateTimeFormat('en-CA',{
    timeZone:'America/Vancouver',
    hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false
  }).format(new Date());
}

setInterval(()=>{
  document.querySelectorAll('.pos-dur-cell').forEach(td=>{
    const el=td.querySelector('.time-dur');
    if(el&&td.dataset.ts)el.textContent=fmtDur(td.dataset.ts);
  });
},30000);

/* ── EQUITY TIMEFRAME BUTTONS (legacy — eqChart may be removed) ────────────── */
document.querySelectorAll('.tf-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const tf=btn.dataset.tf;
    if(tf===_activeTf)return;
    _activeTf=tf;
    document.querySelectorAll('.tf-btn').forEach(b=>b.classList.toggle('active',b.dataset.tf===tf));
    fetch('/api/equity?tf='+tf)
      .then(r=>r.json())
      .then(d=>{if(d.equity_curve&&d.equity_curve.length)renderEquity(d.equity_curve);})
      .catch(console.error);
  });
});

/* ── CASH TIMEFRAME BUTTONS ──────────────────────────────────────────────── */
document.getElementById('cash-tf-btns')&&
document.getElementById('cash-tf-btns').addEventListener('click',e=>{
  const btn=e.target.closest('.cash-tf-btn');
  if(!btn)return;
  setCashTf(btn.dataset.tf);
});

/* Initial cash chart load */
(function loadInitialCash(){
  fetch('/api/cash?limit=1000')
    .then(r=>r.json())
    .then(d=>{
      if(d.cash_curve&&d.cash_curve.length>=2){
        _cashData=d.cash_curve;
        renderCash(filterCashByTf(_cashData,_activeCashTf));
      }
    })
    .catch(console.error);
})();

/* ── EQUITY CHART ─────────────────────────────────────────────────────────── */
function renderEquity(data){
  if(!data||!data.length)return;
  const _ec=g('eqChart');if(!_ec)return;  /* canvas removed — no-op */
  if(data.length<2)data=[data[0],data[0]];
  const ctx=_ec.getContext('2d');
  const vals=data.map(d=>d.equity||0);
  const last=vals[vals.length-1],first=vals[0];
  const up=last>=first;
  const color=up?C.g:C.r;
  const badge=g('eq-badge');
  if(badge){
    const delta=last-first;
    badge.textContent=(delta>=0?'+':'')+fmt$(delta);
    badge.style.color=color;
    badge.style.borderColor=up?'rgba(0,230,118,.2)':'rgba(255,61,87,.2)';
    badge.style.background=up?'rgba(0,230,118,.04)':'rgba(255,61,87,.04)';
  }
  /* Fix 1b: Y-axis auto-scales tightly around the data range (±0.5% padding)
     so a $10,000 baseline doesn't compress a $50 move into a flat line. */
  const minV=Math.min(...vals),maxV=Math.max(...vals);
  const pad=Math.max((maxV-minV)*0.15,maxV*0.003,5);
  const yMin=Math.floor((minV-pad)/5)*5;
  const yMax=Math.ceil((maxV+pad)/5)*5;
  const cfg={
    data:{
      labels:data.map(d=>(d.time||'').slice(11,16)||d.time||''),
      datasets:[{
        data:vals,borderColor:color,borderWidth:1.4,
        fill:true,backgroundColor:color+'14',
        pointRadius:0,tension:0.3
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:{duration:200},
      plugins:{legend:{display:false}},
      scales:{
        x:{
          ticks:{maxTicksLimit:6,color:C.text,maxRotation:0},
          grid:{color:'#2a305055'}
        },
        y:{
          min:yMin,max:yMax,
          ticks:{callback:v=>fmt$(v,0),color:C.text},
          grid:{color:'#2a305055'}
        }
      }
    }
  };
  if(charts.eq){
    charts.eq.data=cfg.data;
    charts.eq.options.scales.y.min=yMin;
    charts.eq.options.scales.y.max=yMax;
    charts.eq.update('none');
  }else{charts.eq=new Chart(ctx,{type:'line',...cfg});}
}

/* ── CASH FLOW CHART ──────────────────────────────────────────────────────── */
function renderCash(data){
  if(!data||data.length<2)return;
  const ctx=g('cashChart');if(!ctx)return;
  const vals=data.map(d=>d.cash||0);
  const last=vals[vals.length-1],first=vals[0];
  const up=last>=first;
  const color=up?C.g:C.r;
  /* cash-badge is now the .kpi-eq-cash-val span inside the left stats column */
  const badge=g('cash-badge');
  if(badge){
    badge.textContent=fmt$(last);
    badge.style.color=up?'var(--pos)':'var(--neg)';
  }
  const minV=Math.min(...vals),maxV=Math.max(...vals);
  const pad=Math.max((maxV-minV)*0.15,maxV*0.003,5);
  const tfMins=CASH_TF_MINS[_activeCashTf]||1440;
  const cfg={
    data:{
      /* Short TF: HH:MM  |  Multi-day TF: MM-DD HH:MM */
      labels:data.map(d=>{const t=d.time||'';return tfMins<=1440?t.slice(11,16):t.slice(5,16).replace('T',' ');}),
      datasets:[{
        data:vals,borderColor:color,borderWidth:1.2,
        fill:true,backgroundColor:color+'0d',
        pointRadius:1.5,pointBackgroundColor:color,tension:0.15
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:{duration:200},
      plugins:{legend:{display:false},tooltip:{callbacks:{
        label:ctx=>'Cash: '+fmt$(ctx.parsed.y)
      }}},
      scales:{
        x:{ticks:{maxTicksLimit:6,color:C.text,maxRotation:0},grid:{color:'#2a305055'}},
        y:{
          min:Math.floor((minV-pad)/10)*10,
          max:Math.ceil((maxV+pad)/10)*10,
          ticks:{callback:v=>fmt$(v,0),color:C.text},
          grid:{color:'#2a305055'}
        }
      }
    }
  };
  if(charts.cash){
    charts.cash.data=cfg.data;
    charts.cash.options.scales.y.min=cfg.options.scales.y.min;
    charts.cash.options.scales.y.max=cfg.options.scales.y.max;
    charts.cash.update('none');
  }else{charts.cash=new Chart(ctx.getContext('2d'),{type:'line',...cfg});}
}

/* ── REGIME CHART ─────────────────────────────────────────────────────────── */
function renderRegime(history){
  const _rc=g('regimeChart');if(!_rc)return;  /* canvas removed — no-op */
  const ctx=_rc.getContext('2d');
  const counts={trend_up:0,trend_down:0,ranging:0,volatile:0};
  (history||[]).forEach(h=>{
    const k=h.regime;
    if(k==='trending_up'||k==='trend_up')counts.trend_up++;
    else if(k==='trending_down'||k==='trend_down')counts.trend_down++;
    else if(k==='ranging')counts.ranging++;
    else if(k==='volatile')counts.volatile++;
  });
  const labels=['▲ Trend','▼ Trend','Ranging','Volatile'];
  const data=Object.values(counts);
  const bg=[C.g+'bb',C.r+'bb',C.b+'bb',C.y+'bb'];
  if(charts.regime){charts.regime.data.datasets[0].data=data;charts.regime.update('none');return;}
  charts.regime=new Chart(ctx,{
    type:'doughnut',
    data:{labels,datasets:[{data,backgroundColor:bg,borderWidth:1,borderColor:'#07080900'}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'65%',
      plugins:{
        legend:{position:'right',labels:{boxWidth:7,padding:8,color:C.text,font:{size:8}}},
        tooltip:{callbacks:{label:ctx=>ctx.label+': '+ctx.parsed+' Trades'}}
      }
    }
  });
}

/* ── WEEKLY BREAKDOWN ─────────────────────────────────────────────────────── */
function renderWeeklyBreakdown(breakdown,todayPnl,weekPnl,todayTrades,todayWins){
  /* KPI cards */
  const todayEl=g('tbp-today-pnl');
  if(todayEl){
    todayEl.textContent=fmtPnl(todayPnl);
    todayEl.className='tbp-kpi-val '+(todayPnl>=0?'c-pos':'c-neg');
  }
  const todaySubEl=g('tbp-today-sub');
  if(todaySubEl){
    const wr=todayTrades>0?Math.round(todayWins/todayTrades*100):null;
    todaySubEl.textContent=todayTrades+' trades · '+(wr!=null?wr+'% W/R':'—');
  }
  const weekEl=g('tbp-week-pnl');
  if(weekEl){
    weekEl.textContent=fmtPnl(weekPnl);
    weekEl.className='tbp-kpi-val '+(weekPnl>=0?'c-pos':'c-neg');
  }
  const weekSubEl=g('tbp-week-sub');
  if(weekSubEl){
    const wt=breakdown.reduce((s,d)=>s+d.trades,0);
    weekSubEl.textContent=wt+' trades this week';
  }

  /* Bar chart scale */
  const tbody=g('week-body');
  if(!tbody||!breakdown||!breakdown.length)return;
  const maxAbs=Math.max(...breakdown.map(d=>Math.abs(d.pnl)),0.01);
  tbody.innerHTML=breakdown.map(d=>{
    const pct=Math.min(Math.abs(d.pnl)/maxAbs*100,100);
    const barColor=d.pnl>=0?C.g:C.r;
    const wrStr=d.win_rate!=null?d.win_rate.toFixed(1)+'%':'—';
    const rowCls=d.is_today?'today-row':'';
    const dayLabel=d.is_today?`<strong style="color:var(--accent)">${d.day}</strong>`:d.day;
    return`<tr class="${rowCls}">
      <td class="left day-cell">${dayLabel}</td>
      <td class="left" style="color:var(--t3);font-size:9px">${d.date}</td>
      <td class="${pnlCls(d.pnl)}">${d.trades>0||d.pnl!==0?fmtPnl(d.pnl):'—'}</td>
      <td>${d.trades>0?d.trades:'—'}</td>
      <td style="color:${d.win_rate!=null?(d.win_rate>=50?C.g:C.r):'var(--t3)'}">${wrStr}</td>
      <td>
        <div class="day-bar-wrap">
          <div class="day-bar-track">
            <div class="day-bar-fill" style="width:${pct.toFixed(1)}%;background:${barColor}"></div>
          </div>
          <span style="font-size:9px;color:var(--t3);min-width:28px;text-align:right">${d.trades>0?d.wins+'W':''}</span>
        </div>
      </td>
    </tr>`;
  }).join('');
}

/* ── FEE DRAG ─────────────────────────────────────────────────────────────── */
function renderFeeDrag(p){
  const fees=p.total_fees_paid||0;
  const eff=p.fee_efficiency;
  const gross=p.gross_pnl||0;
  const fEl=g('fee-total');if(fEl){fEl.textContent=fmtPnl(-fees);fEl.className='fee-val '+(fees>0?'c-neg':'c-dim');}
  const eEl=g('fee-eff');
  if(eEl){
    if(eff!=null){
      eEl.textContent=eff.toFixed(2)+'%';
      eEl.className='fee-val '+(eff>=95?'c-pos':eff>=80?'c-warn':'c-neg');
    }else{eEl.textContent='—';eEl.className='fee-val';}
  }
  const esEl=g('fee-eff-sub');if(esEl)esEl.textContent=eff!=null?'net / gross ratio':'no closed trades';
  const gEl=g('fee-gross');if(gEl){gEl.textContent=fmtPnl(gross);gEl.className='fee-val '+(gross>=0?'c-pos':'c-neg');}
  const gsEl=g('fee-gross-sub');if(gsEl)gsEl.textContent='fees drag: '+fmtPnl(-fees);
}

/* ── POSITIONS TABLE ──────────────────────────────────────────────────────── */
function renderPositions(positions){
  _openSyms=new Set((positions||[]).map(p=>p.symbol));
  const tb=g('pos-body'),cnt=g('pos-count');
  if(cnt)cnt.textContent=String((positions||[]).length);
  if(!positions||!positions.length){tb.innerHTML='<tr class="empty-row"><td colspan="15">No open positions</td></tr>';return;}
  tb.innerHTML=positions.map(p=>{
    const side=(p.side||'long').toLowerCase();
    const upnl=p.unrealised_pnl||0;
    const avgCost=p.avg_cost||0;
    const lp=p.last_price||avgCost;
    const upnlPct=avgCost>0?((lp-avgCost)/avgCost*(side==='short'?-1:1)*100):0;
    const rr=p.rr_ratio;
    const cc=p.candle_count||0;
    const holdCls=cc>120?'hold-danger':cc>60?'hold-warn':'';
    const openedTs=p.opened_ts||p.opened_at||'';
    const slPx=p.sl_price||0,tpPx=p.tp_price||0;
    const slPct=p.sl_pct!=null?((p.sl_pct>=0?'+':'')+p.sl_pct.toFixed(2)+'%'):'—';
    const tpPct=p.tp_pct!=null?((p.tp_pct>=0?'+':'')+p.tp_pct.toFixed(2)+'%'):'—';
    return`<tr class="row-${side}"><td class="left sym-cell">${p.symbol}</td><td class="left">${sideBadge(side)}</td><td>${Number(p.shares||0).toFixed(5)}</td><td>${fmtPrice(avgCost)}</td><td>${fmtPrice(lp)}</td><td style="color:var(--warn)">${fmt$(p.total_invested||0)}</td><td style="color:var(--accent);font-size:9px">${p.deployed_equity_pct!=null?p.deployed_equity_pct.toFixed(2)+'%':'—'}</td><td class="${pnlCls(upnl)}">${fmtPnl(upnl)}</td><td style="color:${pnlColor(upnlPct)};font-size:9px">${(upnlPct>=0?'+':'')+upnlPct.toFixed(2)}%</td><td class="left">${stratTag(p.strategy)}</td><td>${slPx>0?fmtPrice(slPx)+'<div style="font-size:7.5px;color:var(--t3)">'+slPct+'</div>':'—'}</td><td>${tpPx>0?fmtPrice(tpPx)+'<div style="font-size:7.5px;color:var(--t3)">'+tpPct+'</div>':'—'}</td><td>${rr!=null&&rr>0?`<span class="rr-val">1&#8758;${rr.toFixed(2)}</span>`:'<span style="color:var(--t3)">—</span>'}</td><td class="${holdCls}">${cc}</td><td class="pos-dur-cell left" data-ts="${openedTs}"><div class="time-cell"><span class="time-ts">${fmtTs(openedTs)}</span><span class="time-dur">${fmtDur(openedTs)}</span></div></td></tr>`;
  }).join('');
}

/* ── TRADES TABLE (with MFE / MAE) ───────────────────────────────────────── */
function renderTrades(trades){
  const tb=g('trades-body'),cnt=g('trades-count');
  if(cnt)cnt.textContent=String((trades||[]).length);
  if(!trades||!trades.length){tb.innerHTML='<tr class="empty-row"><td colspan="13">No trades yet</td></tr>';return;}
  tb.innerHTML=trades.map(t=>{
    const action=(t.action||'').toLowerCase();
    const side=action==='short'||action==='cover'||t.side==='short'?'short':'long';
    const net=t.net_pnl!=null?t.net_pnl:t.pnl;
    const gross=t.gross_pnl!=null?t.gross_pnl:null;
    const fee=t.fee_total;
    const alc=t.allocated_equity_pct;
    const ts=(t.ts||t.timestamp||'').slice(0,16).replace('T',' ');
    /* Fix 3: gross PnL sub-line under net for fee drag visibility */
    const netTd=net!=null
      ?`<div class="${pnlCls(net)}">${fmtPnl(net)}</div>${gross!=null&&fee!=null?`<div style="font-size:7.5px;color:var(--t3)">gross ${fmtPnl(gross)}</div>`:''}` 
      :'—';
    /* MFE / MAE */
    const mfe=t.mfe!=null?t.mfe:null;
    const mae=t.mae!=null?t.mae:null;
    const mfeTd=mfe!=null?`<span class="mfe-val">${fmtPnl(mfe)}</span>`:'<span class="mfe-dash">—</span>';
    const maeTd=mae!=null?`<span class="mae-val">${fmtPnl(mae)}</span>`:'<span class="mae-dash">—</span>';
    /* Fix 2 + Fix 4: row class for PnL glow; data-label on key cells for mobile card */
    const rowCls=net==null?'':net>0?'trade-win':'trade-loss';
    return`<tr class="${rowCls}">
      <td class="left" data-label="TIME" style="color:var(--t3);font-size:9px">${ts}</td>
      <td class="left sym-cell" data-label="SYMBOL">${t.symbol}</td>
      <td class="left" data-label="SIDE">${sideBadge(side)}</td>
      <td class="left" data-label="ACTION"><span class="chip chip-${action}">${action.toUpperCase()}</span></td>
      <td class="left">${stratTag(t.strategy)}</td>
      <td style="color:var(--t3);font-size:8.5px">${t.regime||'—'}</td>
      <td>${t.price?fmt$(t.price):'—'}</td>
      <td style="color:var(--t3);font-size:8.5px">${t.exec_price?fmt$(t.exec_price):'—'}</td>
      <td data-label="NET PNL">${netTd}</td>
      <td>${mfeTd}</td>
      <td>${maeTd}</td>
      <td data-label="FEES" style="color:var(--t3);font-size:8.5px">${fee!=null?fmt$(fee,4):'—'}</td>
      <td style="color:var(--accent);font-size:8.5px">${alc!=null?alc.toFixed(2)+'%':'—'}</td>
    </tr>`;
  }).join('');
}

/* ── TICKER / PRICES ──────────────────────────────────────────────────────── */
function buildTicker(prices){
  if(!prices||!prices.length)return;
  const all=[...prices,...prices];
  g('ticker-track').innerHTML=all.map(p=>{
    const cls=p.change>0?'t-up':p.change<0?'t-dn':'t-flat';
    const arr=p.change>0?'▲':p.change<0?'▼':'—';
    return`<div class="t-item"><span class="t-sym">${p.symbol}</span><span class="t-price">${fmtPrice(p.price)}</span><span class="${cls}">${arr}${Math.abs(p.change).toFixed(2)}%</span></div>`;
  }).join('');
}
function buildPricesGrid(prices){
  const el=g('prices-grid');if(!el)return;
  const sumEl=g('prices-summary');
  if(sumEl){
    const up=prices.filter(p=>p.change>0).length,dn=prices.filter(p=>p.change<0).length;
    sumEl.innerHTML=`<span style="color:${C.g}">▲${up}</span> <span style="color:${C.r}">▼${dn}</span> / ${prices.length}`;
  }
  el.innerHTML=prices.map(p=>{
    const hasp=_openSyms.has(p.symbol+'USDT');
    const cc=p.change>0?'pc-up':p.change<0?'pc-dn':'pc-flat';
    const arr=p.change>0?'▲ ':p.change<0?'▼ ':'';
    return`<div class="price-card${hasp?' active':''}"><div class="pc-sym">${p.symbol}${hasp?' ●':''}</div><div class="pc-price">${fmtPrice(p.price)}</div><div class="pc-chg ${cc}">${arr}${Math.abs(p.change).toFixed(2)}%</div></div>`;
  }).join('');
}

/* ── APPLY UPDATE ─────────────────────────────────────────────────────────── */
function applyUpdate(d){
  const p=d.portfolio||{};

  /* Timestamp header — show Vancouver time */
  const tsEl=g('h-ts');if(tsEl)tsEl.textContent=d.ts||vanNow()+' PT';
  const dot=g('h-dot'),tickLbl=g('h-tick');

  const b=d.brain||{};
  const regime=b.current_regime||'ranging';

  /* Fix 5: heartbeat indicator — green pulse healthy, orange when CB open,
     solid red on SSE disconnect (handled by es.onerror below). */
  if(dot){
    if(b.circuit_open){dot.className='brand-dot cb-open';}
    else{dot.className='brand-dot';}
    clearTimeout(dot._t);
  }
  if(tickLbl)tickLbl.textContent='TICK '+vanNow()+' PT';

  const hrEl=g('h-regime');
  if(hrEl){hrEl.textContent=regime.replace(/_/g,' ').toUpperCase();hrEl.style.color=(RC[regime]||C.b);hrEl.style.borderColor=(RC[regime]||C.b)+'40';}
  const hcbEl=g('h-cb');
  if(hcbEl){
    if(b.circuit_open){hcbEl.textContent='⚠ CB OPEN';hcbEl.className='hdr-pill pill-cb-open';}
    else{hcbEl.textContent='CB CLEAR';hcbEl.className='hdr-pill pill-cb-ok';}
  }

  const _eq=p.total_equity||0;
  const _eqPnl=p.daily_equity_pnl!=null?p.daily_equity_pnl:p.daily_pnl||0;
  const _eqDailyPct=p.starting_cash&&p.starting_cash>0?(_eqPnl/p.starting_cash*100):0;
  /* Fix 1 + Fix 6: HUD — today PnL, drawdown, profit factor, fees, AI sizing */
  const hudPnl=g('hud-pnl');
  if(hudPnl){hudPnl.textContent=fmtPnl(_eqPnl);hudPnl.style.color=pnlColor(_eqPnl);}
  const hudPct=g('hud-pnl-pct');
  if(hudPct){hudPct.textContent=(_eqDailyPct>=0?'+':'')+_eqDailyPct.toFixed(3)+'% today';}
  /* Max Drawdown */
  const hudDd=g('hud-dd');
  if(hudDd){
    const dd=p.drawdown_pct||0;
    hudDd.textContent=(dd<=0?'':'')+dd.toFixed(2)+'%';
    hudDd.style.color=dd<-5?'var(--neg)':dd<-2?'var(--warn)':'var(--pos)';
  }
  /* Profit Factor */
  const hudPf=g('hud-pf');
  if(hudPf){
    const pf=p.profit_factor;
    if(pf==null){hudPf.textContent='—';hudPf.style.color='var(--t3)';}
    else{
      hudPf.textContent=pf.toFixed(2)+'×';
      hudPf.style.color=pf>=1.5?'var(--pos)':pf>=1.0?'var(--warn)':'var(--neg)';
    }
  }
  /* Total Fees Paid */
  const hudFees=g('hud-fees');
  if(hudFees){
    hudFees.textContent=p.total_fees_paid!=null?fmt$(p.total_fees_paid,2):'—';
    hudFees.style.color='var(--warn)';
  }
  /* AI Sizing */
  const sm=b.current_size_mult!=null?b.current_size_mult:1.0;
  const smPct=Math.round(sm*100);
  const hudSz=g('hud-sizing');
  if(hudSz){
    if(smPct>=100){hudSz.textContent='100% — Nominal';hudSz.style.color='var(--pos)';}
    else{hudSz.textContent='Penalized to '+smPct+'%';hudSz.style.color=smPct<75?'var(--neg)':'var(--warn)';}
  }
  const hudFill=g('hud-ai-fill');
  if(hudFill){
    hudFill.style.width=smPct+'%';
    hudFill.style.background=smPct>=100?'var(--pos)':smPct<75?'var(--neg)':'var(--warn)';
  }

  /* Fix 1: equity always shows bright #00E5FF — red only confuses; peak shown as sub */
  const eq=p.total_equity||0,pk=p.peak_equity||0;
  const eqEl=g('kpi-eq');
  if(eqEl){eqEl.textContent=fmt$(eq);eqEl.className='kpi-value';eqEl.style.color='#00E5FF';}
  const pkEl=g('kpi-peak-sub');if(pkEl)pkEl.textContent='Peak  '+fmt$(pk);

  /* Cash */
  const cEl=g('kpi-cash');if(cEl)cEl.textContent=fmt$(p.cash||0);
  const cpEl=g('kpi-cash-pct');if(cpEl&&eq>0)cpEl.textContent=((p.cash||0)/eq*100).toFixed(1)+'% uninvested';

  /* Return */
  const rEl=g('kpi-ret');
  if(rEl){const rt=p.return_pct||0;rEl.textContent=(rt<0?'↓ ':'↑ ')+Math.abs(Number(rt).toFixed(3))+'%';rEl.className='kpi-value '+(rt>=0?'c-pos':'c-neg');}
  const rsEl=g('kpi-ret-start');if(rsEl&&p.starting_cash)rsEl.textContent='vs '+fmt$(p.starting_cash);

  /* Net PnL */
  const npEl=g('kpi-netpnl');
  if(npEl){const np=p.total_net_pnl||0;npEl.textContent=fmtPnl(np);npEl.className='kpi-value '+(np>=0?'c-pos':'c-neg');}
  const npSubEl=g('kpi-netpnl-sub');
  if(npSubEl&&p.realised_pnl_net!=null&&p.unrealised_pnl_total!=null){
    npSubEl.innerHTML=`<span style="color:${pnlColor(p.realised_pnl_net)}">R:${fmtPnl(p.realised_pnl_net)}</span> <span style="color:${pnlColor(p.unrealised_pnl_total)}">U:${fmtPnl(p.unrealised_pnl_total)}</span>`;
  }

  /* Drawdown */
  const ddEl=g('kpi-dd');
  if(ddEl){const dd=p.drawdown_pct||0;ddEl.textContent=fmtP(dd);ddEl.className='kpi-value '+(dd>=0?'c-pos':dd<-3?'c-neg':'c-warn');}
  const ddSubEl=g('kpi-dd-sub');if(ddSubEl&&pk)ddSubEl.textContent='peak '+fmt$(pk);

  /* Win rate — overall + long/short breakdown */
  const wrEl=g('kpi-wr');if(wrEl)wrEl.textContent=(p.win_rate||0).toFixed(1)+'%';
  const _lifetimeTrades=p.lifetime_trades||p.total_trades||0;
  const wrSubEl=g('kpi-wr-sub');if(wrSubEl)wrSubEl.textContent=_lifetimeTrades.toLocaleString()+' trades (lifetime)';
  const wrBar=g('wr-bar-fill');if(wrBar)wrBar.style.width=(p.win_rate||0).toFixed(1)+'%';
  const tEl=g('kpi-trades');if(tEl)tEl.textContent=String(_lifetimeTrades);
  /* Long / Short win rate breakdown */
  const lwrEl=g('kpi-long-wr');if(lwrEl)lwrEl.textContent=p.win_rate_long!=null?p.win_rate_long.toFixed(1)+'%':'—';
  const swrEl=g('kpi-short-wr');if(swrEl)swrEl.textContent=p.win_rate_short!=null?p.win_rate_short.toFixed(1)+'%':'—';
  /* Lifetime Trades KPI card */
  const ltEl=g('lifetime-trades');if(ltEl)ltEl.textContent=_lifetimeTrades.toLocaleString();
  const ltSubEl=g('lifetime-trades-sub');if(ltSubEl)ltSubEl.textContent='win rate '+((p.win_rate||0).toFixed(1))+'% · '+_lifetimeTrades.toLocaleString()+' total';
  const posSubEl=g('kpi-pos-sub');if(posSubEl)posSubEl.textContent=(d.positions||[]).length+' open';

  /* Profit factor */
  const pfEl=g('kpi-pf'),pfSub=g('kpi-pf-sub');
  if(pfEl&&d.trades){
    const nPv=t=>t.net_pnl!=null?t.net_pnl:t.pnl;
    const cl=(d.trades||[]).filter(t=>nPv(t)!=null);
    const gp=cl.reduce((s,t)=>nPv(t)>0?s+nPv(t):s,0);
    const gl=cl.reduce((s,t)=>nPv(t)<0?s+Math.abs(nPv(t)):s,0);
    if(!gl&&!gp){pfEl.textContent='—';pfEl.className='kpi-value';}
    else if(!gl){pfEl.textContent='MAX';pfEl.className='kpi-value c-pos';}
    else{
      const pf=gp/gl;pfEl.textContent=pf.toFixed(2);
      pfEl.className='kpi-value '+(pf>1.2?'c-pos':pf<1?'c-neg':'c-warn');
    }
    if(pfSub&&gl>0){
      pfSub.innerHTML=`<div class="pf-footer"><span class="pf-item"><em>GP</em><span style="color:var(--pos)">${fmt$(gp)}</span></span><span class="pf-divider">|</span><span class="pf-item"><em>GL</em><span style="color:var(--neg)">${fmt$(gl)}</span></span></div>`;
    }
  }

  /* Expectancy */
  const expEl=g('kpi-exp');
  if(expEl&&d.trades){
    const nPv=t=>t.net_pnl!=null?t.net_pnl:t.pnl;
    const cl=(d.trades||[]).filter(t=>nPv(t)!=null);
    const wins=cl.filter(t=>nPv(t)>0),losses=cl.filter(t=>nPv(t)<0);
    const aw=wins.length?wins.reduce((s,t)=>s+nPv(t),0)/wins.length:null;
    const al=losses.length?losses.reduce((s,t)=>s+nPv(t),0)/losses.length:null;
    if(aw!==null||al!==null){
      expEl.innerHTML=`<span style="color:${C.g}">${aw!==null?fmtPnl(aw):'—'}</span> / <span style="color:${C.r}">${al!==null?fmtPnl(al):'—'}</span>`;
    }else expEl.textContent='—';
  }

  /* Account strip */
  const aiEl=g('acct-id');
  if(aiEl&&p.starting_cash!=null&&p.total_net_pnl!=null&&p.total_equity!=null){
    const computed=Number((p.starting_cash+p.total_net_pnl).toFixed(2));
    const diff=Math.abs(computed-p.total_equity);
    const ok=diff<0.02;
    aiEl.textContent=fmt$(p.starting_cash)+' + ('+fmtPnl(p.total_net_pnl)+') = '+fmt$(computed)+(ok?' ✓ balanced':' ⚠ drift '+fmt$(diff));
    aiEl.className=ok?'acct-ok':'acct-fail';
  }
  const arEl=g('acct-realized');if(arEl&&p.realised_pnl_net!=null){arEl.textContent=fmtPnl(p.realised_pnl_net);arEl.style.color=pnlColor(p.realised_pnl_net);}
  const aoEl=g('acct-open');if(aoEl&&p.unrealised_pnl_total!=null){aoEl.textContent=fmtPnl(p.unrealised_pnl_total);aoEl.style.color=pnlColor(p.unrealised_pnl_total);}

  /* Session row — Fix 2: show equity-based daily PnL (includes floating PnL)
     which matches the equity chart instead of the realized-only closed-trade tally.
     The weekly breakdown table still uses realized-only figures for consistency. */
  const _eqDailyPnl=p.daily_equity_pnl!=null?p.daily_equity_pnl:p.daily_pnl;
  const sdpEl=g('sess-daily-pnl');
  if(sdpEl&&_eqDailyPnl!=null){
    sdpEl.textContent=fmtPnl(_eqDailyPnl);
    sdpEl.style.color=pnlColor(_eqDailyPnl);
    sdpEl.title='Equity-based (vs midnight Pacific). Realized only: '+fmtPnl(p.daily_pnl||0);
  }
  const sdtEl=g('sess-daily-trades');if(sdtEl)sdtEl.textContent=(p.daily_trades||0)+' closed trades';
  const sdwEl=g('sess-daily-wr');
  if(sdwEl&&p.daily_trades!=null){
    const dt=p.daily_trades||0,dw=p.daily_wins||0;
    const dwr=dt>0?Math.round(dw/dt*100):null;
    sdwEl.textContent=dwr!=null?(dwr+'%'):'—';
    sdwEl.style.color=dwr!=null?(dwr>=50?C.g:C.r):'var(--t3)';
    const sdwsEl=g('sess-daily-wr-sub');if(sdwsEl)sdwsEl.textContent=dw+' W / '+dt+' T';
  }
  const sunEl=g('sess-unr');if(sunEl&&p.unrealised_pnl_total!=null){sunEl.textContent=fmtPnl(p.unrealised_pnl_total);sunEl.style.color=pnlColor(p.unrealised_pnl_total);}
  const sunsEl=g('sess-unr-sub');if(sunsEl)sunsEl.textContent='across '+(d.positions||[]).length+' positions';
  const srEl=g('sess-realized');if(srEl&&p.realised_pnl_net!=null){srEl.textContent=fmtPnl(p.realised_pnl_net);srEl.style.color=pnlColor(p.realised_pnl_net);}

  /* Exposure bar */
  if(eq>0){
    const now=Date.now();
    if(now-_lastExpUpdate>1000){
      _lastExpUpdate=now;
      const expPct=Math.max(0,Math.min(100,(eq-(p.cash||0))/eq*100));
      const deployed=eq-(p.cash||0);
      const fill=g('exp-fill'),pctEl=g('exp-pct'),det=g('exp-detail');
      if(fill){fill.style.width=expPct.toFixed(1)+'%';fill.style.background=expPct>95?'var(--neg)':expPct>80?'var(--warn)':'var(--accent)';}
      if(pctEl){pctEl.textContent=expPct.toFixed(1)+'%';pctEl.style.color=expPct>95?'var(--neg)':expPct>80?'var(--warn)':'var(--t2)';}
      if(det)det.textContent=fmt$(deployed)+' / '+fmt$(eq);
      const deplEl=g('exp-deployed');if(deplEl)deplEl.textContent=fmt$(deployed);
      const eqValEl=g('exp-equity-val');if(eqValEl)eqValEl.textContent=fmt$(eq);
    }
  }

  /* Margin health */
  const mInfo=g('margin-info'),mHalt=g('margin-halt');
  if(mInfo&&d.margin_health){
    const mh=d.margin_health;
    const util=mh.utilization_pct!=null?mh.utilization_pct.toFixed(1)+'% util':'—';
    const ml=mh.margin_level!=null?' · ML '+mh.margin_level.toFixed(3):'';
    mInfo.textContent=(mh.source||'—')+' · '+util+ml+(mh.halt_reason?' · '+mh.halt_reason:'');
  }
  if(mHalt&&d.margin_health)mHalt.style.display=d.margin_health.halt_new_entries?'inline-flex':'none';

  /* Fee drag */
  renderFeeDrag(p);

  /* Time-based performance */
  if(p.weekly_breakdown&&p.weekly_breakdown.length){
    renderWeeklyBreakdown(p.weekly_breakdown,p.daily_pnl||0,p.weekly_pnl||0,p.daily_trades||0,p.daily_wins||0);
  }

  /* Charts + tables */
  /* Fix 1 — TF state preservation:
     SSE packets now carry equity_tf=null and equity_curve=[].
     On initial load (/api/data returns 24H data → renders once).
     On SSE updates, always re-fetch with the user's active TF (throttled to
     every 30 s so we don't flood the server).  This means the chart auto-
     updates on every TF the user has selected without resetting to 24H. */
  if(d.equity_curve&&d.equity_curve.length){
    if(!d.equity_tf||d.equity_tf===_activeTf){
      renderEquity(d.equity_curve);
    }else{
      /* TF mismatch (SSE stripped the curve) — refresh via API with active TF */
      const _now=Date.now();
      if(_now-_lastEqFetch>30000){
        _lastEqFetch=_now;
        fetch('/api/equity?tf='+_activeTf)
          .then(r=>r.json())
          .then(eqd=>{if(eqd.equity_curve&&eqd.equity_curve.length)renderEquity(eqd.equity_curve);})
          .catch(console.error);
      }
    }
  }else if(!d.equity_curve||!d.equity_curve.length){
    /* SSE stripped the curve — re-fetch with user's TF (throttled) */
    const _now=Date.now();
    if(_now-_lastEqFetch>30000){
      _lastEqFetch=_now;
      fetch('/api/equity?tf='+_activeTf)
        .then(r=>r.json())
        .then(eqd=>{if(eqd.equity_curve&&eqd.equity_curve.length)renderEquity(eqd.equity_curve);})
        .catch(console.error);
    }
  }
  if(b.regime_history&&b.regime_history.length)renderRegime(b.regime_history);
  /* cash_curve is [] in SSE (served by /api/cash); just re-render from memory */
  if(d.cash_curve&&d.cash_curve.length>=2){
    _cashData=d.cash_curve;
    renderCash(filterCashByTf(_cashData,_activeCashTf));
  }else if(_cashData.length>=2){
    /* Re-filter in case TF changed since last full render */
    renderCash(filterCashByTf(_cashData,_activeCashTf));
  }
  renderPositions(d.positions||[]);
  renderTrades(d.trades||[]);
  if(d.prices&&d.prices.length){_pricesData=d.prices;buildTicker(d.prices);buildPricesGrid(d.prices);}
}

const es=new EventSource('/stream');
es.addEventListener('update',e=>{try{applyUpdate(JSON.parse(e.data));g('h-dot').className='brand-dot';}catch(err){console.error('SSE parse error',err);}});
es.onerror=()=>{g('h-dot').className='brand-dot err';};
fetch('/api/data').then(r=>r.json()).then(applyUpdate).catch(console.error);
</script>
</body>
</html>"""


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
