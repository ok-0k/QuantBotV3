"""Tests for V4 startup reconciliation and the cash-invariant report."""

import logging

import pytest

import bot
from config import STARTING_CASH


def test_stuck_paper_order_marked_orphaned(clean_db):
    """A PENDING_NEW row left by a crash resolves to ORPHANED (never executed)."""
    clean_db.try_create_order("qb-stuck", "AAAUSDT", "buy", "long",
                              1_700_000_000_000, 2.0, 100.0, mode="paper")
    assert len(clean_db.get_open_orders()) == 1

    bot._reconcile_orders_on_boot()

    assert clean_db.get_open_orders() == []
    row = clean_db.get_order("qb-stuck")
    assert row["state"] == "ORPHANED"
    assert "not executed" in row["note"]


def test_reconcile_noop_on_clean_journal(clean_db, caplog):
    with caplog.at_level(logging.INFO):
        bot._reconcile_orders_on_boot()
    assert any("journal clean" in r.message for r in caplog.records)


def test_cash_invariant_balanced_books(clean_db, caplog):
    # cash = STARTING - deployed; realised_net = 0 -> drift 0
    clean_db.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0)
    clean_db.set_cash(STARTING_CASH - 200.0)
    with caplog.at_level(logging.INFO):
        bot._report_cash_invariant()
    assert any("drift=$+0.00" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING and "drifted" in r.message
                   for r in caplog.records)


def test_cash_invariant_detects_drift(clean_db, caplog):
    clean_db.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 130.0)
    clean_db.set_cash(STARTING_CASH - 200.0 + 50.0)   # $50 unexplained
    with caplog.at_level(logging.WARNING):
        bot._report_cash_invariant()
    assert any("drifted" in r.message for r in caplog.records)


def test_cash_invariant_counts_short_margin(clean_db, caplog):
    clean_db.open_short("BBBUSDT", 3.0, 50.0, "QF", 55.0, 45.0, margin_reserved=30.0)
    clean_db.set_cash(STARTING_CASH - 30.0)
    with caplog.at_level(logging.INFO):
        bot._report_cash_invariant()
    assert any("drift=$+0.00" in r.message for r in caplog.records)
