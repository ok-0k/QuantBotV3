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


# ── RL experience logging (LEDGER-9 / ENTRY-09 / EXIT-04) ────────────────────

def test_cover_rl_row_uses_stored_entry_state_and_action(fresh):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    _open_short_via_bot(candles, sac=0.5)
    # Another open short so total equity != cash (next_state must use equity).
    fresh.open_short("LEDGERXUSDT", 1.0, 40.0, "QF", 44.0, 36.0, margin_reserved=8.0)
    fresh.set_cash(fresh.get_cash() - 8.0)

    cov = _ensemble("cover", candles[-1]["close"] * 0.95, symbol="QFUSDT")
    bot._execute_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)

    rows = _rl_rows(fresh)
    assert len(rows) == 1
    assert json.loads(rows[0]["state"]) == pytest.approx(ENTRY_STATE.tolist())
    assert rows[0]["action"] == pytest.approx(0.5)
    cash = fresh.get_cash()
    next_state = json.loads(rows[0]["next_state"])
    assert next_state[1] == pytest.approx(cash / (cash + 8.0), rel=1e-5)


def test_sell_rl_row_uses_stored_entry_state_and_action(fresh):
    candles = make_candles(n=120, start_price=110.0, amplitude=0.4)
    fresh.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0,
                        entry_state=ENTRY_STATE.tolist())
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET entry_features=? WHERE symbol='AAAUSDT'",
                     (json.dumps({"sac_fraction": 0.3}),))
    fresh.set_cash(STARTING_CASH - 200.0)

    bot._execute_trade(_ensemble("sell", 110.0), "TEST", candles, None,
                       ZERO_STATE, STARTING_CASH)

    rows = _rl_rows(fresh)
    assert len(rows) == 1
    assert json.loads(rows[0]["state"]) == pytest.approx(ENTRY_STATE.tolist())
    assert rows[0]["action"] == pytest.approx(0.3)


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


# ── Atomic booking (LEDGER-6 / ACR-05 / EXIT-07) ─────────────────────────────

def test_cover_failing_mid_booking_rolls_back_everything(fresh, monkeypatch):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    _open_short_via_bot(candles)
    cov = _ensemble("cover", candles[-1]["close"] * 0.95, symbol="QFUSDT")
    prepared = bot._prepare_trade(cov, "QF", candles, None, ZERO_STATE, STARTING_CASH)
    assert prepared is not None
    fill = bot._exec_adapter.execute(prepared.order_intent)
    coid = prepared.order_intent.client_order_id
    before = _book(fresh)

    real_log_trade = bot.log_trade

    def boom(_rec):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(bot, "log_trade", boom)       # last write in the booking
    with pytest.raises(sqlite3.OperationalError):
        bot._commit_trade(prepared, fill)

    assert _book(fresh) == before                     # cash, short, stats untouched
    assert coid in [o["client_order_id"] for o in fresh.get_open_orders()]
    assert _rl_rows(fresh) == []

    # The retry books exactly once.
    monkeypatch.setattr(bot, "log_trade", real_log_trade)
    bot._commit_trade(prepared, fill)
    assert fresh.get_all_positions() == []
    assert fresh.get_order(coid)["state"] == "FILLED"
    margin = before["positions"][0][2]
    net = float(fresh.get_portfolio_stat("realised_pnl_net"))
    assert fresh.get_cash() == pytest.approx(before["cash"] + margin + net, rel=1e-9)


def test_sell_failing_mid_booking_rolls_back_everything(fresh, monkeypatch):
    candles = make_candles(n=120, start_price=110.0, amplitude=0.4)
    fresh.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0)
    fresh.set_cash(STARTING_CASH - 200.0)
    prepared = bot._prepare_trade(_ensemble("sell", 110.0), "TEST", candles, None,
                                  ZERO_STATE, STARTING_CASH)
    fill = bot._exec_adapter.execute(prepared.order_intent)
    before = _book(fresh)

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(bot, "update_order", boom)    # after cash + close_position
    with pytest.raises(sqlite3.OperationalError):
        bot._commit_trade(prepared, fill)

    assert _book(fresh) == before
    assert prepared.order_intent.client_order_id in [
        o["client_order_id"] for o in fresh.get_open_orders()]


def test_short_entry_failing_mid_booking_rolls_back(fresh, monkeypatch):
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    ens = _ensemble("short", candles[-1]["close"], symbol="QFUSDT", ml_prob=0.03)
    prepared = bot._prepare_trade(ens, "QF", candles, 0.5, ENTRY_STATE, STARTING_CASH)
    fill = bot._exec_adapter.execute(prepared.order_intent)
    before = _book(fresh)

    def boom(_rec):
        raise OSError("No space left on device")

    monkeypatch.setattr(bot, "log_trade", boom)
    with pytest.raises(OSError):
        bot._commit_trade(prepared, fill)

    assert _book(fresh) == before                     # no margin reserved, no cash debit
    assert prepared.order_intent.client_order_id in [
        o["client_order_id"] for o in fresh.get_open_orders()]


# ── Ledger anchor + reconciliation (LEDGER-5 / LEDGER-1 / ACR-10 / IMPR-1) ───

def _drifted_books(db, offset=396.41):
    """Open short (margin 30) + realised -50, with an unrecorded top-up `offset`."""
    db.open_short("BBBUSDT", 3.0, 50.0, "QF", 55.0, 45.0, margin_reserved=30.0)
    db.set_portfolio_stat("realised_pnl_net", -50.0)
    db.set_cash(STARTING_CASH - 50.0 - 30.0 + offset)


def test_ledger_drift_formula_and_capital_adjustments(fresh):
    _drifted_books(fresh)
    assert fresh.ledger_drift()["drift"] == pytest.approx(396.41)
    fresh.set_portfolio_stat("capital_adjustments", 396.41)
    led = fresh.ledger_drift()
    assert led["drift"] == pytest.approx(0.0, abs=1e-6)
    assert led["deployed"] == pytest.approx(30.0)


def test_boot_check_records_drift_and_points_to_anchor_not_heal(fresh, caplog):
    _drifted_books(fresh)
    with caplog.at_level(logging.INFO):
        bot._report_cash_invariant()
    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) == 1 and "drifted" in warns[0]
    assert "set_ledger_anchor.py" in warns[0]
    assert "heal_equity" not in warns[0]
    assert float(fresh.get_portfolio_stat("ledger_drift")) == pytest.approx(396.41)
    assert fresh.get_portfolio_stat("ledger_checked_at")

    fresh.set_portfolio_stat("capital_adjustments", 396.41)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        bot._report_cash_invariant()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert float(fresh.get_portfolio_stat("ledger_drift")) == 0.0


def test_periodic_check_warns_only_when_drift_moves(fresh, caplog):
    _drifted_books(fresh)
    bot._report_cash_invariant()                        # boot: records +396.41
    caplog.clear()
    with caplog.at_level(logging.INFO):
        bot._report_cash_invariant(periodic=True)       # same known offset
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    fresh.set_cash(fresh.get_cash() + 5.0)              # a $5 leak
    with caplog.at_level(logging.INFO):
        bot._report_cash_invariant(periodic=True)
    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) == 1 and "+396.41" in warns[0]
    assert float(fresh.get_portfolio_stat("ledger_drift")) == pytest.approx(401.41)


def test_housekeeping_loop_writes_ledger_drift(fresh, monkeypatch):
    _drifted_books(fresh, offset=12.5)

    class _OneCycle:
        """First wait times out (one housekeeping pass), second returns."""
        calls = 0

        def is_set(self):
            return self.calls >= 2

        async def wait(self):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.TimeoutError
            return True

    monkeypatch.setattr(bot, "_shutdown_event", _OneCycle())
    assert fresh.get_portfolio_stat("ledger_drift") is None
    asyncio.run(bot.housekeeping_loop())
    assert float(fresh.get_portfolio_stat("ledger_drift")) == pytest.approx(12.5)
    assert fresh.get_portfolio_stat("ledger_checked_at")


def test_set_ledger_anchor_confirm_zeroes_drift(fresh, monkeypatch, capsys):
    _drifted_books(fresh)
    cash_before = fresh.get_cash()
    script = _load_script("set_ledger_anchor", "scripts/set_ledger_anchor.py")

    _run_main(script, monkeypatch)                      # dry-run writes nothing
    assert fresh.get_portfolio_stat("capital_adjustments") is None
    for argv in (["--confirm"], ["--confirm", "--amount", "396.39"]):
        with pytest.raises(SystemExit) as exc:          # no blind plug, no near-miss
            _run_main(script, monkeypatch, *argv)
        assert exc.value.code == 1
        assert fresh.get_portfolio_stat("capital_adjustments") is None

    _run_main(script, monkeypatch, "--confirm", "--amount", "396.41")
    assert fresh.ledger_drift()["drift"] == pytest.approx(0.0, abs=1e-6)
    assert fresh.get_capital_adjustments() == pytest.approx(396.41)
    assert fresh.get_cash() == pytest.approx(cash_before)        # books untouched
    assert len(fresh.get_all_positions()) == 1
    backups = sorted((config.DATA_DIR / "backups").glob("ledger-anchor-pre-*.json"))
    assert backups and json.loads(backups[-1].read_text())["old_capital_adjustments"] == 0.0

    # A later unrecorded change is folded on top of the existing anchor.
    fresh.set_cash(fresh.get_cash() + 10.0)
    _run_main(script, monkeypatch, "--confirm", "--amount", "10.00")
    assert fresh.get_capital_adjustments() == pytest.approx(406.41)
    assert fresh.ledger_drift()["drift"] == pytest.approx(0.0, abs=1e-6)
    assert "drift now" in capsys.readouterr().out


def test_set_cash_balance_records_capital_adjustment(fresh, monkeypatch):
    # Flat, balanced books after heavy losses: cash 43.75, realised -9956.25.
    fresh.set_portfolio_stat("realised_pnl_net", 43.75 - STARTING_CASH)
    fresh.set_cash(43.75)
    assert fresh.ledger_drift()["drift"] == pytest.approx(0.0, abs=1e-6)
    script = _load_script("set_cash_balance", "set_cash_balance.py")

    _run_main(script, monkeypatch, "--amount", "10000", "--confirm")

    assert fresh.get_cash() == pytest.approx(10_000.0)
    assert fresh.get_capital_adjustments() == pytest.approx(10_000.0 - 43.75)
    assert fresh.ledger_drift()["drift"] == pytest.approx(0.0, abs=1e-6)


def test_heal_equity_refuses_with_open_positions(fresh, monkeypatch, capsys):
    fresh.open_short("BBBUSDT", 3.0, 50.0, "QF", 55.0, 45.0, margin_reserved=30.0)
    fresh.set_cash(STARTING_CASH - 30.0)
    fresh.record_equity(STARTING_CASH)
    script = _load_script("heal_equity", "heal_equity.py")

    with pytest.raises(SystemExit) as exc:
        _run_main(script, monkeypatch, "--confirm")
    assert exc.value.code == 1
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 30.0)
    out = capsys.readouterr().out
    assert "REFUSING TO RUN" in out and "set_ledger_anchor.py" in out

