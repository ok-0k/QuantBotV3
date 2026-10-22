"""
Money path, supporting pieces: the dashboard's position and equity
arithmetic (dashboard._build_telemetry, _position_risk_metrics), the margin
health record that can halt new entries (binance_margin), and the SAC weight
loader whose output sizes every entry (rl_agent._load_weights).
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import binance_margin as BM
import dashboard
import db
import rl_agent
from config import MARGIN_LEVEL_HALT_BELOW, STARTING_CASH, SYMBOLS

S_SYM, L_SYM = SYMBOLS[4], SYMBOLS[6]


# ── dashboard: per-position risk metrics ─────────────────────────────────────

def test_risk_metrics_short_and_long():
    s = dashboard._position_risk_metrics("short", 100.0, 101.0, 97.0)
    assert s == {"risk_per_share": 1.0, "reward_per_share": 3.0, "rr_ratio": 3.0,
                 "sl_pct": 1.0, "tp_pct": 3.0}
    lg = dashboard._position_risk_metrics("long", 100.0, 98.0, 104.0)
    assert (lg["risk_per_share"], lg["reward_per_share"], lg["rr_ratio"]) == (2.0, 4.0, 2.0)
    assert (lg["sl_pct"], lg["tp_pct"]) == (2.0, 4.0)


def test_risk_metrics_without_levels_report_none_not_infinity():
    m = dashboard._position_risk_metrics("short", 100.0, 0.0, 0.0)
    assert m["rr_ratio"] is None and m["sl_pct"] is None and m["tp_pct"] is None
    assert dashboard._position_risk_metrics("short", 0.0, 1.0, 0.0)["risk_per_share"] == 1.0   # entry floored


# ── dashboard: telemetry arithmetic ──────────────────────────────────────────

def _book(clean_db):
    """$10k: a 10-share short at $100 ($200 margin, mark $95) and a 10-share
    long at $100 (mark $101). Equity = 8,800 + (200 + 50) + 1,010 = 10,060."""
    clean_db.open_short(S_SYM, 10.0, 100.0, "QF", 103.0, 90.0, 200.0)
    clean_db.open_position(L_SYM, 10.0, 100.0, "QF", 98.0, 104.0)
    clean_db.set_cash(STARTING_CASH - 200.0 - 1000.0)
    clean_db.set_portfolio_stat(f"last_price_{S_SYM}", 95.0)
    clean_db.set_portfolio_stat(f"last_price_{L_SYM}", 101.0)


def test_telemetry_marks_both_sides_and_sums_equity(clean_db):
    _book(clean_db)
    d = dashboard._build_telemetry()
    p = d["portfolio"]
    assert p["cash"] == 8800.0
    assert p["total_equity"] == 10060.0
    assert p["total_net_pnl"] == 60.0 and p["return_pct"] == 0.6
    assert p["unrealised_pnl_total"] == 60.0
    pos = {r["symbol"]: r for r in d["positions"]}
    s, lg = pos[S_SYM], pos[L_SYM]
    assert (s["unrealised_pnl"], s["total_invested"]) == (50.0, 200.0)
    assert (s["potential_loss"], s["potential_gain"], s["rr_ratio"]) == (30.0, 100.0, 3.333)
    assert (s["stop_loss_total"], s["tp_total"]) == (1030.0, 900.0)
    assert (lg["unrealised_pnl"], lg["total_invested"]) == (10.0, 1000.0)
    assert (lg["potential_loss"], lg["potential_gain"], lg["rr_ratio"]) == (20.0, 40.0, 2.0)
    assert s["deployed_equity_pct"] == pytest.approx(round(100 * 200 / 10060, 3))
    assert lg["deployed_equity_pct"] == pytest.approx(round(100 * 1000 / 10060, 3))
    assert s["opened_epoch"] is not None


def test_telemetry_marks_at_cost_when_no_price_has_arrived(clean_db):
    clean_db.open_position(L_SYM, 10.0, 100.0, "QF", 0.0, 0.0)
    clean_db.set_cash(STARTING_CASH - 1000.0)
    row = dashboard._build_telemetry()["positions"][0]
    assert row["last_price"] == 100.0 and row["unrealised_pnl"] == 0.0
    assert row["potential_loss"] is None and row["potential_gain"] is None
    assert row["stop_loss_total"] is None and row["tp_total"] is None


def test_telemetry_tolerates_bad_stats_and_estimates_missing_fees(clean_db):
    clean_db.set_portfolio_stat("margin_health_json", "{broken")
    clean_db.set_portfolio_stat("total_trades", "many")
    clean_db.save_brain_key("adaptive_edge_profiles_v1", ["not", "a", "dict"])
    now = datetime.now(timezone.utc).isoformat()
    for pnl, fee in ((4.0, 0.25), (-1.0, 0.0)):          # the second predates stored fees
        db.log_trade({"symbol": S_SYM, "action": "cover", "status": "filled", "side": "short",
                      "pnl": pnl, "net_pnl": pnl, "gross_pnl": pnl, "fee_total": fee,
                      "trade_value": 100.0, "timestamp": now})
    d = dashboard._build_telemetry()
    p = d["portfolio"]
    assert d["margin_health"] == {}
    assert p["lifetime_trades"] == 0
    assert p["closed_7d"] == 2 and p["win_rate"] == 50.0 and p["profit_factor"] == 4.0
    assert p["total_fees_paid"] == pytest.approx(0.25 + 100.0 * dashboard.FEE_GATE_ROUND_TRIP)
    assert d["brain"]["current_size_mult"] == 1.0


def test_daily_equity_change_is_measured_from_the_last_point_before_midnight(clean_db):
    _book(clean_db)
    old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    with clean_db.get_db() as conn:
        conn.execute("INSERT INTO equity_curve (ts, equity) VALUES (?, ?)", (old, 9_900.0))
    d = dashboard._build_telemetry()                     # 24H window: the old point is outside it
    assert d["portfolio"]["daily_equity_pnl"] == 160.0
    assert len(d["equity_curve"]) == 2 and d["equity_curve"][0]["equity"] == 9_900.0
    w = dashboard._build_telemetry(equity_tf="1W")       # 1W reads the time range directly
    assert w["equity_tf"] == "1W" and [pt["equity"] for pt in w["equity_curve"]] == [9_900.0, 9_900.0]


# ── margin health ────────────────────────────────────────────────────────────

def test_available_usdt_sums_free_usdt_only():
    data = {"userAssets": [{"asset": "USDT", "free": "10.5"}, {"asset": "usdt", "free": "4.5"},
                           {"asset": "USDT", "free": "oops"}, {"asset": "BTC", "free": "9"}]}
    assert BM._estimate_available_usdt(data) == 15.0
    assert BM._estimate_available_usdt({"assets": [{"asset": "USDT", "free": "2"}]}) == 2.0
    assert BM._estimate_available_usdt({"userAssets": [{"asset": "USDT", "free": "0"}]}) is None
    assert BM._estimate_available_usdt({}) is None


def test_live_margin_level_below_floor_halts():
    h = BM.evaluate_margin_health(live_json={"marginLevel": str(MARGIN_LEVEL_HALT_BELOW - 0.01)},
                                  cash=9_000.0, total_equity=10_000.0)
    assert h["halt_new_entries"] is True and h["source"] == "binance_sapi"
    assert h["halt_reason"].startswith("margin_level=")
    assert h["utilization_pct"] == pytest.approx(10.0)


def test_synthetic_utilisation_halts_when_the_book_is_deployed():
    h = BM.evaluate_margin_health(live_json=None, cash=500.0, total_equity=10_000.0)
    assert h["halt_new_entries"] is True and h["source"] == "synthetic"
    assert h["halt_reason"] == "synthetic_util=95.0%>=90%"


def test_healthy_or_unknown_margin_does_not_halt():
    ok = BM.evaluate_margin_health(live_json={"marginLevel": "3.5", "userAssets": []},
                                   cash=9_000.0, total_equity=10_000.0)
    assert ok["halt_new_entries"] is False and ok["margin_level"] == 3.5
    blank = BM.evaluate_margin_health(live_json={"marginLevel": ""}, cash=0.0, total_equity=0.0)
    assert blank["halt_new_entries"] is False
    assert blank["margin_level"] is None and blank["utilization_pct"] is None
    bad = BM.evaluate_margin_health(live_json={"marginLevel": "n/a"}, cash=1.0, total_equity=1.0)
    assert bad["margin_level"] is None and bad["halt_new_entries"] is False


def test_live_halt_reason_wins_over_synthetic():
    h = BM.evaluate_margin_health(live_json={"marginLevel": "1.0"}, cash=0.0, total_equity=10_000.0)
    assert h["halt_reason"] == f"margin_level=1.0000<{MARGIN_LEVEL_HALT_BELOW}"


# ── SAC weights ──────────────────────────────────────────────────────────────

KEYS = ("W1", "b1", "W2", "b2", "W_mu", "b_mu", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta")


def _weights(path, b_mu=0.5, drop=None):
    rng = np.random.default_rng(0)
    w = {"W1": rng.normal(size=(13, 32)), "b1": np.zeros(32), "W2": rng.normal(size=(32, 32)),
         "b2": np.zeros(32), "W_mu": np.zeros((32, 1)), "b_mu": np.array([b_mu]),
         "ln1_gamma": np.ones(32), "ln1_beta": np.zeros(32), "ln2_gamma": np.ones(32),
         "ln2_beta": np.zeros(32)}
    if drop:
        w.pop(drop)
    np.savez(path, **w)
    return path


@pytest.fixture()
def actor(tmp_path):
    a = rl_agent.SACActorNumpy.__new__(rl_agent.SACActorNumpy)     # no default-path load
    a._trained = False
    return a


def test_missing_weights_use_the_bounded_heuristic(actor, tmp_path):
    assert actor._load_weights(tmp_path / "none.npz") is False
    state = np.zeros(13, dtype=np.float32)
    state[0] = 1.0
    assert actor.forward(state) == pytest.approx(0.5)                # clipped at +0.5
    state[0] = 0.6
    assert actor.forward(state) == pytest.approx(0.15)


def test_loaded_weights_drive_the_position_fraction(actor, tmp_path):
    assert actor._load_weights(_weights(tmp_path / "w.npz", b_mu=0.5)) is True
    assert actor._trained is True and actor.W1.dtype == np.float32
    assert actor.forward(np.ones(13, dtype=np.float32)) == pytest.approx(np.tanh(0.5), rel=1e-6)
    assert actor._load_weights(_weights(tmp_path / "big.npz", b_mu=50.0)) is True
    assert actor.forward(np.zeros(13, dtype=np.float32)) == pytest.approx(np.tanh(10.0))   # squash clip


@pytest.mark.parametrize("bad", ["missing_key", "corrupt"])
def test_broken_weights_fall_back_to_the_heuristic(actor, tmp_path, caplog, bad):
    actor._load_weights(_weights(tmp_path / "good.npz"))             # previously trained
    if bad == "missing_key":
        path = _weights(tmp_path / "bad.npz", drop="ln2_beta")
        text = "Missing key in SAC weights file"
    else:
        path = tmp_path / "bad.npz"
        path.write_bytes(b"not an npz archive")
        text = "Failed to load SAC weights"
    with caplog.at_level("ERROR"):
        assert actor._load_weights(path) is False
    assert actor._trained is False
    state = np.zeros(13, dtype=np.float32)
    assert actor.forward(state) == pytest.approx(-0.5)               # heuristic, not stale weights
    assert any(text in r.getMessage() for r in caplog.records)
