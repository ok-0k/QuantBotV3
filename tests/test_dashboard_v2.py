"""
Dashboard v2 (audit round 5): server-side parts of the dashboard fixes --
auth on non-ASCII tokens, ledger tile, breaker drawdown, bot liveness,
lifetime trade count, experiment per-symbol-day metric, cleared-cost
threshold.
"""

import asyncio
from datetime import datetime, timezone

import pytest
from starlette.requests import Request

import dashboard


def _request(query: bytes = b"", headers: list | None = None) -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "query_string": query,
                    "headers": headers or [], "server": ("t", 80), "scheme": "http"})


def _call_middleware(req: Request):
    async def call_next(_req):
        return dashboard.JSONResponse({"ok": True})
    return asyncio.run(dashboard._require_token(req, call_next))


def test_auth_non_ascii_token_is_401_not_500(monkeypatch):
    monkeypatch.setattr(dashboard, "_AUTH_TOKEN", "s3cret")
    assert _call_middleware(_request(b"token=%C3%A9")).status_code == 401
    assert _call_middleware(_request(headers=[(b"authorization", "Bearer é".encode())])).status_code == 401
    assert _call_middleware(_request(b"token=s3cret")).status_code == 200


def test_auth_non_ascii_configured_token(monkeypatch):
    monkeypatch.setattr(dashboard, "_AUTH_TOKEN", "pässwörd")
    assert _call_middleware(_request(b"token=p%C3%A4ssw%C3%B6rd")).status_code == 200
    assert _call_middleware(_request(b"token=nope")).status_code == 401


# ── Experiment: net P&L per symbol-day (assignment unit = symbol) ─────────────

import json  # noqa: E402

import bot  # noqa: E402
import db  # noqa: E402
import insights  # noqa: E402
from config import SYMBOLS  # noqa: E402

A_SYM, B_SYM = SYMBOLS[0], SYMBOLS[1]


def _exp_fill(sym, arm, ts, pnl=None):
    """One decay_v2 journal row: an entry (pnl None) or a closed exit."""
    with db.get_db() as conn:
        conn.execute(
            "INSERT INTO trades (ts, symbol, action, status, pnl, trade_value, entry_features) "
            "VALUES (?,?,?,?,?,?,?)",
            (ts, sym, "short" if pnl is None else "cover", "filled", pnl,
             None if pnl is None else 100.0, json.dumps({"experiment": "decay_v2", "arm": arm})))


def _trade(sym, arm, day, pnl, hour=10):
    _exp_fill(sym, arm, f"{day}T{hour:02d}:00:00+00:00")
    _exp_fill(sym, arm, f"{day}T{hour:02d}:30:00+00:00", pnl)


def _experiment():
    conn = insights.connect()
    try:
        return insights.experiment(conn, "decay_v2")
    finally:
        conn.close()


def test_arm_symbols_match_bot_assignment_and_exclude_delisted():
    arms = insights.arm_symbols()
    for arm, syms in arms.items():
        assert all(bot._experiment_arm(s) == arm for s in syms)
    listed = [s for s in SYMBOLS if s not in insights.DELISTED_SYMBOLS]
    assert sorted(arms["A"] + arms["B"]) == sorted(listed)
    assert not insights.DELISTED_SYMBOLS & set(arms["A"] + arms["B"])


def test_per_symbol_day_means_include_zero_trade_symbols_and_gap_days(clean_db):
    n_a, n_b = (len(v) for v in (insights.arm_symbols()["A"], insights.arm_symbols()["B"]))
    _trade(A_SYM, "A", "2026-10-06", -4.0)
    _trade(B_SYM, "B", "2026-10-06", -1.0)
    _trade(A_SYM, "A", "2026-10-08", -2.0)          # 10-07: nobody traded -> zero day
    _trade("MATICUSDT", "B", "2026-10-08", 500.0)   # delisted: excluded
    psd = _experiment()["per_symbol_day"]
    assert psd["days"] == 3 and (psd["first_day"], psd["last_day"]) == ("2026-10-06", "2026-10-08")
    assert psd["symbols"] == {"A": n_a, "B": n_b}
    assert psd["mean"]["A"] == pytest.approx(round(-6.0 / n_a / 3, 3))
    assert psd["mean"]["B"] == pytest.approx(round(-1.0 / n_b / 3, 3))
    assert psd["net_per_symbol"]["A"] == pytest.approx(round(-6.0 / n_a, 3))
    assert psd["diff"] == pytest.approx(round((-1.0 / n_b - (-6.0 / n_a)) / 3, 3), abs=1e-3)
    assert psd["ci_lo"] <= psd["diff"] <= psd["ci_hi"]
    assert psd["rule"] == insights.DECISION_RULE
    assert "upper bound < 0" in psd["rule"]


def test_day_bootstrap_is_deterministic():
    vals = [0.3, -0.1, 0.8, 0.05, -0.4, 0.2]
    assert insights.day_bootstrap_ci(vals) == insights.day_bootstrap_ci(vals)
    lo, hi = insights.day_bootstrap_ci(vals)
    assert min(vals) <= lo < hi <= max(vals)
    assert insights.day_bootstrap_ci(vals, seed=1) != insights.day_bootstrap_ci(vals, seed=2)


def test_verdict_follows_the_preregistered_rule(clean_db):
    for i, (a, b) in enumerate([(-1.0, -9.0), (-2.0, -12.0), (-1.5, -10.0)]):
        day = f"2026-10-0{6 + i}"
        _trade(A_SYM, "A", day, a)
        _trade(B_SYM, "B", day, b)
    psd = _experiment()["per_symbol_day"]
    assert psd["ci_hi"] < 0 and psd["verdict"].startswith("Keep A")
    assert _experiment()["per_symbol_day"] == psd                # same data, same CI


def test_verdict_adopts_b_when_not_significantly_worse(clean_db):
    for i, (a, b) in enumerate([(-1.0, 2.0), (-2.0, -3.0), (1.0, -1.0)]):
        day = f"2026-10-0{6 + i}"
        _trade(A_SYM, "A", day, a)
        _trade(B_SYM, "B", day, b)
    exp = _experiment()
    psd = exp["per_symbol_day"]
    assert psd["ci_lo"] < 0 < psd["ci_hi"] and psd["verdict"].startswith("Adopt B")
    assert "V3_FIXES" in exp["note"]
    assert exp["arms"]["A"]["closed"] == 3                     # per-trade stats kept


def test_single_day_has_no_ci_yet(clean_db):
    _trade(A_SYM, "A", "2026-10-06", -1.0)
    psd = _experiment()["per_symbol_day"]
    assert psd["days"] == 1 and psd["ci_lo"] is None and psd["verdict"].startswith("not enough")


def test_experiment_report_prints_per_symbol_day_and_verdict(clean_db, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    for i, (a, b) in enumerate([(-1.0, 2.0), (-2.0, -3.0)]):
        day = f"2026-10-0{6 + i}"
        _trade(A_SYM, "A", day, a)
        _trade(B_SYM, "B", day, b)
    path = Path(__file__).resolve().parent.parent / "scripts" / "experiment_report.py"
    spec = importlib.util.spec_from_file_location("experiment_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["experiment_report.py", "decay_v2"])
    mod.main()
    out = capsys.readouterr().out
    assert "per symbol-day" in out and "day bootstrap, 2000 resamples" in out
    assert insights.DECISION_RULE in out and "Verdict: Adopt B" in out


def test_experiment_card_headlines_per_symbol_day():
    html = dashboard.DASHBOARD_HTML
    assert "e.per_symbol_day" in html and "B − A per symbol-day" in html
    assert html.index("B − A per symbol-day") < html.index("B − A per trade")   # per-trade is secondary
    assert "psd.rule" in html and "e.note" in html


# ── Insights: cleared-cost threshold (DASH-3) ─────────────────────────────────

def test_cleared_costs_threshold_counts_entry_slippage_once(clean_db):
    from config import FEE_GATE_ROUND_TRIP, SLIPPAGE_PCT
    thr = (FEE_GATE_ROUND_TRIP + SLIPPAGE_PCT) * 100
    assert insights.PEAK_BREAKEVEN_PCT == pytest.approx(thr)
    ts = datetime.now(timezone.utc).isoformat()
    with db.get_db() as conn:
        for peak in (thr + 0.01, (thr + insights.COST_PCT) / 2, thr - 0.05, 0.0):
            conn.execute("INSERT INTO trades (ts, symbol, action, status, pnl, trade_value, exit_reason, "
                         "peak_profit_pct) VALUES (?,?,?,?,?,?,?,?)",
                         (ts, A_SYM, "cover", "filled", -0.1, 100.0, "STOP_LOSS", peak))
    conn = insights.connect()
    try:
        w = insights.journal_window(conn, 7.0)
    finally:
        conn.close()
    assert w["cleared_cost_share"] == 50.0          # was 25% against the 0.36% round-trip
    assert w["cost_pct"] == pytest.approx(insights.COST_PCT, abs=1e-3)   # label unchanged
    assert w["peak_breakeven_pct"] == pytest.approx(thr, abs=1e-3)


# ── System health: ledger tile, breaker drawdown, bot liveness ───────────────

import time  # noqa: E402


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def test_ledger_tile_reads_bot_drift_not_a_tautology(clean_db):
    now = time.time()
    assert dashboard._ledger_check(now) == {"drift": None, "ok": None, "value": "—", "checked_age_s": None}
    db.set_portfolio_stat("ledger_drift", "0.73")
    db.set_portfolio_stat("ledger_checked_at", _iso(now - 120))
    lg = dashboard._ledger_check(now)
    assert lg["ok"] is True and lg["value"] == "Balanced" and lg["checked_age_s"] == pytest.approx(120, abs=1)
    db.set_portfolio_stat("ledger_drift", "-1.0")
    assert dashboard._ledger_check(now)["value"] == "Balanced"            # |x| <= $1
    db.set_portfolio_stat("ledger_drift", "-396.41")
    db.set_portfolio_stat("ledger_checked_at", str(now - 30))             # epoch seconds also accepted
    lg = dashboard._ledger_check(now)
    assert lg["ok"] is False and lg["value"] == "Drift -$396.41" and lg["checked_age_s"] == pytest.approx(30, abs=1)
    assert "p.starting_cash + p.total_net_pnl" not in dashboard.DASHBOARD_HTML


def test_breaker_drawdown_from_persisted_peak_and_distance_to_trip(clean_db):
    import brain
    assert dashboard._BREAKER_PEAK_KEY == brain._PEAK_EQUITY_BRAIN_KEY
    assert dashboard._breaker_state(9_000.0)["breaker_drawdown_pct"] is None   # not persisted yet
    db.save_brain_key(dashboard._BREAKER_PEAK_KEY, 10_000.0)
    s = dashboard._breaker_state(8_800.0)
    assert s["breaker_drawdown_pct"] == pytest.approx(-12.0)
    assert s["breaker_limit_pct"] == pytest.approx(dashboard.MAX_DRAWDOWN_PCT * 100)
    assert s["breaker_headroom_pct"] == pytest.approx(dashboard.MAX_DRAWDOWN_PCT * 100 - 12.0)
    assert s["breaker_trip_equity"] == pytest.approx(10_000 * (1 - dashboard.MAX_DRAWDOWN_PCT))
    assert dashboard._breaker_state(10_500.0)["breaker_drawdown_pct"] == 0.0   # bot ratchets the peak


def test_telemetry_peak_breaker_and_liveness(clean_db, monkeypatch):
    now = time.time()
    db.set_cash(9_000.0)
    db.set_equity_rebase(9_500.0, _iso(now - 400_000))
    db.save_brain_key(dashboard._BREAKER_PEAK_KEY, 10_003.0)
    rows = [(_iso(now - 500_000), 12_000.0),          # before the rebase: ignored
            (_iso(now - 390_000), 10_400.0)]          # post-rebase peak, older than 10k rows
    rows += [(_iso(now - 200 - 30 * i), 9_100.0) for i in range(10_001)]
    with db.get_db() as conn:
        conn.executemany("INSERT INTO equity_curve (ts, equity) VALUES (?,?)", rows)
    db.set_portfolio_stat("exit_monitor_last_cycle", str(now - 10))
    db.set_portfolio_stat("model_age_xgb_days", "12")
    db.set_portfolio_stat("model_age_sac_days", "45.5")
    monkeypatch.setattr(dashboard, "V3_FIXES", True)
    d = dashboard._build_telemetry()
    p, h = d["portfolio"], d["health"]
    assert p["peak_equity"] == pytest.approx(10_400.0)
    assert p["drawdown_pct"] == pytest.approx((9_000 - 10_400) / 10_400 * 100, abs=1e-3)
    assert p["breaker_peak_equity"] == pytest.approx(10_003.0)
    assert p["breaker_drawdown_pct"] == pytest.approx(-(10_003 - 9_000) / 10_003 * 100, abs=1e-3)
    assert h["ages_s"]["bot"] == pytest.approx(200, abs=5)
    assert h["ages_s"]["exit_monitor"] == pytest.approx(10, abs=5)
    assert h["ages_s"]["ws"] is None and h["ages_s"]["housekeeping"] is None      # not written yet
    assert h["stale"] == ["bot"]                                                    # > 90 s
    assert h["models_stale"] is True and h["model_age_sac_days"] == 45.5
    assert h["v3_fixes"] is True
    assert h["ledger"]["value"] == "—"


def test_liveness_fresh_bot_is_not_stale(clean_db):
    now = time.time()
    with db.get_db() as conn:
        conn.execute("INSERT INTO equity_curve (ts, equity) VALUES (?,?)", (_iso(now - 20), 9_000.0))
    db.set_portfolio_stat("ws_last_msg", _iso(now - 2).replace("+00:00", "Z"))
    lv = dashboard._liveness(now)
    assert lv["stale"] == [] and lv["ages_s"]["ws"] == pytest.approx(2, abs=1)
    assert lv["models_stale"] is False and lv["model_age_xgb_days"] is None
    assert "not reporting for" in dashboard.DASHBOARD_HTML


# ── Labels: lifetime trades and 7-day cards (DASH-6) ─────────────────────────

def test_lifetime_trades_is_portfolio_total_trades(clean_db):
    db.set_portfolio_stat("total_trades", "10459")
    with db.get_db() as conn:
        conn.execute("INSERT INTO trades (ts, symbol, action, status, pnl, net_pnl, trade_value) "
                     "VALUES (?,?,?,?,?,?,?)", (_iso(time.time() - 60), A_SYM, "cover", "filled", 1.0, 1.0, 100.0))
    p = dashboard._build_telemetry()["portfolio"]
    assert p["lifetime_trades"] == p["total_trades"] == 10459      # not 6629 + row count
    assert p["closed_7d"] == 1
    html = dashboard.DASHBOARD_HTML
    for label in ("Win Rate · 7d", "Profit Factor · 7d", "Fees Paid · 7d", "closed (7d)"):
        assert label in html
