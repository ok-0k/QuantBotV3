"""
Ledger & booking: RL transition logging (LEDGER-9 / ENTRY-09 / EXIT-04),
atomic trade booking (LEDGER-6 / ACR-05 / EXIT-07) and the ledger anchor +
reconciliation (LEDGER-5 / LEDGER-1 / ACR-10 / IMPR-1 / OPS-2).
"""

import asyncio
import importlib.util
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

import bot
import config
from config import STARTING_CASH
from conftest import make_candles


ROOT = Path(__file__).resolve().parent.parent
ZERO_STATE = np.zeros(13, dtype=np.float32)
ENTRY_STATE = np.arange(1, 14, dtype=np.float32) / 100.0     # 0.01 .. 0.13


def _ensemble(action, price, symbol="AAAUSDT", ml_prob=None):
    ens = {
        "signal": action, "symbol": symbol, "price": price,
        "regime": "trend_down" if action == "short" else "ranging",
        "on_fire": False, "position_size_mult": 1.0,
        "buy_weight": 0.0, "sell_weight": 0.0, "breakdown": [],
        "time": "2026-07-11T00:00:00Z",
    }
    if ml_prob is not None:
        ens["ml_prob"] = ml_prob
    return ens


@pytest.fixture()
def fresh(clean_db):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    return clean_db


def _rl_rows(db):
    with db.get_db() as conn:
        rows = conn.execute("SELECT * FROM rl_experience ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def _open_short_via_bot(candles, symbol="QFUSDT", sac=0.5):
    ens = _ensemble("short", candles[-1]["close"], symbol=symbol, ml_prob=0.03)
    bot._execute_trade(ens, "QF", candles, sac, ENTRY_STATE, STARTING_CASH)
    pos = bot.get_short_position(symbol)
    assert pos is not None, "short entry did not book"
    return pos


def _load_script(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [mod.__name__, *argv])
    mod.main()


def _book(db):
    """Everything an atomic booking must move together (or not at all)."""
    with db.get_db() as conn:
        n_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    return {
        "cash": db.get_cash(),
        "positions": [(p["symbol"], p["shares"], p["margin_reserved"])
                      for p in db.get_all_positions()],
        "realised_pnl": db.get_portfolio_stat("realised_pnl"),
        "realised_pnl_net": db.get_portfolio_stat("realised_pnl_net"),
        "total_trades": db.get_portfolio_stat("total_trades"),
        "n_trades": n_trades,
    }


# ── db.transaction() (LEDGER-6) ──────────────────────────────────────────────

def test_nested_get_db_inside_transaction_does_not_commit_early(fresh):
    fresh.set_cash(1000.0)
    other = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
    try:
        def seen_by_other():
            return float(other.execute(
                "SELECT value FROM portfolio WHERE key='cash'").fetchone()[0])

        with fresh.transaction():
            fresh.set_cash(1234.0)                     # nested get_db()
            with fresh.transaction():                  # nested transaction joins
                fresh.set_portfolio_stat("total_trades", 7)
            assert fresh.get_cash() == pytest.approx(1234.0)   # own writes visible
            assert seen_by_other() == pytest.approx(1000.0)    # not committed yet
        assert seen_by_other() == pytest.approx(1234.0)        # one commit at exit
    finally:
        other.close()

    with pytest.raises(RuntimeError):
        with fresh.transaction():
            fresh.set_cash(1.0)
            fresh.set_portfolio_stat("total_trades", 99)
            raise RuntimeError("boom")
    assert fresh.get_cash() == pytest.approx(1234.0)
    assert fresh.get_portfolio_stat("total_trades") == "7"
    # Back to per-helper autocommit afterwards.
    fresh.set_cash(55.0)
    other = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
    try:
        assert float(other.execute(
            "SELECT value FROM portfolio WHERE key='cash'").fetchone()[0]) == 55.0
    finally:
        other.close()


# ── Ledger anchor + reconciliation (LEDGER-5 / LEDGER-1 / ACR-10 / IMPR-1) ───

def _drifted_books(db, offset=396.41):
    """Open short (margin 30) + realised -50, with an unrecorded top-up `offset`."""
    db.open_short("BBBUSDT", 3.0, 50.0, "QF", 55.0, 45.0, margin_reserved=30.0)
    db.set_portfolio_stat("realised_pnl_net", -50.0)
    db.set_cash(STARTING_CASH - 50.0 - 30.0 + offset)
