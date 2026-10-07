"""Tests for the V4 order-intent journal (duplicate-order protection core)."""

import pytest


@pytest.fixture()
def orders_db(clean_db):
    with clean_db.get_db() as conn:
        conn.execute("DELETE FROM orders")
    return clean_db


def test_try_create_order_first_claim_wins(orders_db):
    ok = orders_db.try_create_order(
        "qb-abc123", "AAAUSDT", "buy", "long", 1_700_000_000_000,
        2.0, 100.0, mode="paper",
    )
    assert ok is True
    row = orders_db.get_order("qb-abc123")
    assert row["state"] == "PENDING_NEW"
    assert row["symbol"] == "AAAUSDT"
    assert row["mode"] == "paper"


def test_try_create_order_duplicate_is_rejected(orders_db):
    assert orders_db.try_create_order(
        "qb-abc123", "AAAUSDT", "buy", "long", 1, 2.0, 100.0, mode="paper")
    # Same client_order_id (same symbol+action+candle) -> duplicate blocked
    assert orders_db.try_create_order(
        "qb-abc123", "AAAUSDT", "buy", "long", 1, 2.0, 100.0, mode="paper") is False
    # Journal still shows exactly one row
    assert len(orders_db.get_open_orders()) == 1


def test_update_order_lifecycle_to_filled(orders_db):
    orders_db.try_create_order("qb-x", "AAAUSDT", "buy", "long", 1, 2.0, 100.0, mode="paper")
    orders_db.update_order("qb-x", state="FILLED", filled_qty=2.0,
                           avg_fill_price=100.015, note="paper fill")
    row = orders_db.get_order("qb-x")
    assert row["state"] == "FILLED"
    assert row["filled_qty"] == pytest.approx(2.0)
    assert row["avg_fill_price"] == pytest.approx(100.015)
    # Terminal rows disappear from the reconciliation queue
    assert orders_db.get_open_orders() == []


def test_open_orders_lists_only_non_terminal(orders_db):
    orders_db.try_create_order("qb-1", "A", "buy", "long", 1, 1.0, 1.0, mode="paper")
    orders_db.try_create_order("qb-2", "B", "short", "short", 1, 1.0, 1.0, mode="paper")
    orders_db.try_create_order("qb-3", "C", "buy", "long", 1, 1.0, 1.0, mode="paper")
    orders_db.update_order("qb-2", state="REJECTED", note="gate")
    orders_db.update_order("qb-3", state="PARTIALLY_FILLED", filled_qty=0.4)
    open_ids = {r["client_order_id"] for r in orders_db.get_open_orders()}
    assert open_ids == {"qb-1", "qb-3"}   # PARTIALLY_FILLED is non-terminal


def test_orphaned_is_terminal(orders_db):
    orders_db.try_create_order("qb-z", "A", "buy", "long", 1, 1.0, 1.0, mode="paper")
    orders_db.update_order("qb-z", state="ORPHANED", note="startup reconcile")
    assert orders_db.get_open_orders() == []
