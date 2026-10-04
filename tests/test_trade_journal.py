"""
Trade journal: entry features, canonical exit reason, peak profit / max
drawdown, hold time — schema migration, helpers, and the live exit pipeline.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
from config import STARTING_CASH
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)

JOURNAL_COLS = (
    "entry_features", "exit_reason", "exit_detail",
    "peak_profit_pct", "max_drawdown_pct", "hold_time_seconds",
)


def _short_ensemble(symbol, price, ml_prob=0.05):
    return {
        "signal": "short", "symbol": symbol, "price": price,
        "regime": "trend_down", "on_fire": False, "position_size_mult": 1.0,
        "buy_weight": 0.0, "sell_weight": 0.6, "ml_prob": ml_prob,
        "breakdown": [], "time": "2026-10-04T00:00:00Z",
        "meta": {"short_score": 0.71, "adx": 27.5, "pdi": 14.0, "mdi": 29.0,
                 "rvol": 1.1, "ret5": -0.002, "ret20": -0.006,
                 "ema_spread": -0.001, "ml_component": 0.9, "struct_component": 0.5},
    }


def _cover_ensemble(symbol, price, exit_reason="TAKE_PROFIT", exit_detail="take_profit_short (x)"):
    return {
        "signal": "cover", "symbol": symbol, "price": price, "regime": "ranging",
        "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0,
        "sell_weight": 0.0, "breakdown": [], "time": "2026-10-04T00:00:00Z",
        "exit_reason": exit_reason, "exit_detail": exit_detail,
    }


@pytest.fixture()
def fresh(clean_db):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    return clean_db


def _trades(db, action):
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE action=? AND status='filled' ORDER BY id", (action,)
        ).fetchall()
    return [dict(r) for r in rows]


def _open_short(db, symbol="AAAUSDT"):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    price = candles[-1]["close"]
    bot._execute_trade(_short_ensemble(symbol, price), "QF_TR_S7_M95", candles, 0.5,
                       ZERO_STATE, STARTING_CASH)
    return candles, price


# ── Schema / migration ───────────────────────────────────────────────────────

def test_fresh_schema_has_journal_columns(fresh):
    with fresh.get_db() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
        pcols = {r[1] for r in conn.execute("PRAGMA table_info(positions)")}
    assert set(JOURNAL_COLS) <= cols
    assert "entry_features" in pcols


def test_migration_adds_columns_to_old_db_and_preserves_rows(fresh):
    """Simulate a pre-journal DB: drop the new columns, keep an old row, re-init."""
    with fresh.get_db() as conn:
        for c in JOURNAL_COLS:
            conn.execute(f"ALTER TABLE trades DROP COLUMN {c}")
        conn.execute("ALTER TABLE positions DROP COLUMN entry_features")
        conn.execute(
            "INSERT INTO trades (ts,symbol,action,side,status,pnl) "
            "VALUES ('2026-10-01T00:00:00+00:00','OLDUSDT','cover','short','filled',-1.5)"
        )
    fresh.init_db()
    fresh.init_db()   # idempotent
    with fresh.get_db() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
        assert set(JOURNAL_COLS) <= cols
        assert "entry_features" in {r[1] for r in conn.execute("PRAGMA table_info(positions)")}
        old = dict(conn.execute("SELECT * FROM trades WHERE symbol='OLDUSDT'").fetchone())
    assert old["pnl"] == -1.5
    assert all(old[c] is None for c in JOURNAL_COLS)


# ── Exit-reason classifier ───────────────────────────────────────────────────

@pytest.mark.parametrize("reason,avg,stop,short,expected", [
    ("take_profit_short ($1 <= $2)", 100, 101, True, "TAKE_PROFIT"),
    ("take_profit_tick_short (low=1)", 100, 101, True, "TAKE_PROFIT"),
    ("take_profit (x)", 100, 99, False, "TAKE_PROFIT"),
    ("stop_loss_short ($1 >= $2)", 100, 101, True, "STOP_LOSS"),
    ("stop_loss_tick_short (high=1)", 100, 100, True, "BREAKEVEN_STOP"),
    ("stop_loss_tick_short (high=1)", 100, 98, True, "TRAILING_STOP"),
    ("stop_loss (x)", 100, 99, False, "STOP_LOSS"),
    ("stop_loss (x)", 100, 102, False, "TRAILING_STOP"),
    ("stop_loss (x)", 100, 100, False, "BREAKEVEN_STOP"),
    ("hard_stop_survival_tick (pnl=$-20)", 100, 101, True, "HARD_STOP"),
    ("time_hold_exit (4.0h open)", 100, 101, True, "TIME_HOLD_EXIT"),
    ("max_hold_time (90 candles)", 100, 101, True, "MAX_HOLD"),
    ("circuit_breaker_flatten", 100, 101, True, "CIRCUIT_BREAKER"),
    ("something new", 100, 101, True, "OTHER"),
    (None, 100, 101, True, "OTHER"),
    ("stop_loss_short", 0, 0, True, "STOP_LOSS"),   # degenerate inputs
])
def test_classify_exit_reason(reason, avg, stop, short, expected):
    assert bot._classify_exit_reason(reason, avg, stop, short) == expected


# ── Excursion / hold-time math ───────────────────────────────────────────────

def _opened(seconds_ago):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def test_exit_fields_short_excursions_and_hold():
    pos = {"avg_cost": 100.0, "mfe_price": 97.0, "mae_price": 101.5,
           "opened_ts": _opened(600), "entry_features": '{"a": 1}'}
    f = bot._exit_journal_fields(pos, True, {"exit_reason": "TAKE_PROFIT", "exit_detail": "d"})
    assert f["peak_profit_pct"] == pytest.approx(3.0)
    assert f["max_drawdown_pct"] == pytest.approx(-1.5)
    assert 598 <= f["hold_time_seconds"] <= 603
    assert f["entry_features"] == '{"a": 1}'
    assert (f["exit_reason"], f["exit_detail"]) == ("TAKE_PROFIT", "d")


def test_exit_fields_long_excursions():
    pos = {"avg_cost": 100.0, "mfe_price": 104.0, "mae_price": 98.0, "opened_ts": _opened(5)}
    f = bot._exit_journal_fields(pos, False, {})
    assert f["peak_profit_pct"] == pytest.approx(4.0)
    assert f["max_drawdown_pct"] == pytest.approx(-2.0)


def test_exit_fields_clamped_when_price_never_went_our_way():
    # short that only ever traded above entry: peak profit 0, not negative
    pos = {"avg_cost": 100.0, "mfe_price": 100.4, "mae_price": 102.0}
    f = bot._exit_journal_fields(pos, True, {})
    assert f["peak_profit_pct"] == 0.0
    assert f["max_drawdown_pct"] == pytest.approx(-2.0)


def test_exit_fields_unknown_is_none_not_zero():
    f = bot._exit_journal_fields({"avg_cost": 100.0}, True, {})
    assert f["peak_profit_pct"] is None
    assert f["max_drawdown_pct"] is None
    assert f["hold_time_seconds"] is None
    assert f["entry_features"] is None


# ── Pipeline: entry -> tracking -> exit ──────────────────────────────────────

def test_short_entry_captures_features_and_seeds_excursions(fresh):
    _open_short(fresh)
    pos = fresh.get_all_positions()[0]
    assert pos["mfe_price"] == pos["mae_price"] == pytest.approx(pos["avg_cost"])
    feats = json.loads(pos["entry_features"])
    assert feats["ml_tier"] == 95 and feats["ml_down"] == pytest.approx(0.95)
    assert feats["regime"] == "trend_down"
    assert feats["short_score"] == 0.71 and feats["adx"] == 27.5
    assert feats["stop_pct"] > 0 and feats["tp_pct"] > 0
    assert feats["equity"] == STARTING_CASH
    # the entry row carries the same snapshot
    entry = _trades(fresh, "short")[0]
    assert json.loads(entry["entry_features"]) == feats


def test_cover_journals_everything(fresh):
    candles, price = _open_short(fresh)
    pos = fresh.get_all_positions()[0]
    entry = pos["avg_cost"]
    fresh.update_mfe_mae("AAAUSDT", entry * 1.01, entry * 0.97, True)    # squeeze +1%, dip -3%
    fresh.update_mfe_mae("AAAUSDT", entry * 1.002, entry * 0.99, True)   # inside range: no change
    with fresh.get_db() as conn:                                          # opened 20 minutes ago
        conn.execute("UPDATE positions SET opened_ts=? WHERE symbol='AAAUSDT'", (_opened(1200),))

    bot._execute_trade(_cover_ensemble("AAAUSDT", entry * 0.97), "QF", candles, None,
                       ZERO_STATE, STARTING_CASH)

    assert fresh.get_all_positions() == []
    row = _trades(fresh, "cover")[0]
    assert row["exit_reason"] == "TAKE_PROFIT"
    assert row["exit_detail"] == "take_profit_short (x)"
    assert row["peak_profit_pct"] == pytest.approx(3.0, abs=1e-3)
    assert row["max_drawdown_pct"] == pytest.approx(-1.0, abs=1e-3)
    assert 1195 <= row["hold_time_seconds"] <= 1210
    assert json.loads(row["entry_features"])["ml_tier"] == 95


def test_exit_without_reason_still_journals_numbers(fresh):
    """Back-compat: callers that don't pass exit_reason (tests, manual) still work."""
    candles, price = _open_short(fresh)
    ens = _cover_ensemble("AAAUSDT", price)
    ens.pop("exit_reason"); ens.pop("exit_detail")
    bot._execute_trade(ens, "QF", candles, None, ZERO_STATE, STARTING_CASH)
    row = _trades(fresh, "cover")[0]
    assert row["exit_reason"] is None
    assert row["hold_time_seconds"] is not None and row["peak_profit_pct"] is not None


def test_tick_exit_path_stamps_canonical_reason(fresh):
    """End-to-end through _tick_exit_check: a TP touch is journaled as TAKE_PROFIT."""
    candles, price = _open_short(fresh)
    pos = fresh.get_all_positions()[0]
    bot._candles_cache["AAAUSDT"] = candles
    crash = pos["avg_cost"] * 0.90        # far below TP and well clear of fees
    tick = {"open": pos["avg_cost"], "high": pos["avg_cost"], "low": crash,
            "close": crash, "volume": 1.0, "time": candles[-1]["time"] + 60_000}

    async def run():
        await bot._tick_exit_check("AAAUSDT", tick, asyncio.get_running_loop())
    asyncio.run(run())

    assert fresh.get_all_positions() == []
    row = _trades(fresh, "cover")[0]
    assert row["exit_reason"] == "TAKE_PROFIT"
    assert row["peak_profit_pct"] == pytest.approx(10.0, abs=1e-3)
    assert row["max_drawdown_pct"] == 0.0
    assert row["hold_time_seconds"] is not None and row["hold_time_seconds"] < 30
    assert json.loads(row["entry_features"])["regime"] == "trend_down"
