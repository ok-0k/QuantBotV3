"""Characterization tests for db.py — position lifecycle, cash, trade log."""

import pytest

from config import STARTING_CASH


def test_seed_portfolio_starting_cash(clean_db):
    assert clean_db.get_cash() == pytest.approx(STARTING_CASH)


def test_set_get_cash_roundtrip(clean_db):
    clean_db.set_cash(1234.56)
    assert clean_db.get_cash() == pytest.approx(1234.56)


def test_open_long_position(clean_db):
    clean_db.open_position("AAAUSDT", 2.0, 100.0, "TEST", 95.0, 110.0)
    positions = clean_db.get_all_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p["side"] == "long"
    assert p["shares"] == pytest.approx(2.0)
    assert p["avg_cost"] == pytest.approx(100.0)
    assert clean_db.open_position_count() == 1
    assert clean_db.open_short_count() == 0


def test_long_averaging_weighted_cost_and_tightest_stop(clean_db):
    clean_db.open_position("AAAUSDT", 1.0, 100.0, "TEST", 95.0, 110.0)
    clean_db.open_position("AAAUSDT", 1.0, 200.0, "TEST", 90.0, 220.0)
    p = clean_db.get_position("AAAUSDT")
    assert p["shares"] == pytest.approx(2.0)
    assert p["avg_cost"] == pytest.approx(150.0)
    assert p["stop_price"] == pytest.approx(95.0)   # max() keeps tightest stop
    assert p["tp_price"] == pytest.approx(220.0)


def test_close_position_hard_deletes(clean_db):
    clean_db.open_position("AAAUSDT", 1.0, 100.0, "TEST", 95.0, 110.0)
    clean_db.close_position("AAAUSDT")
    assert clean_db.get_all_positions() == []
    assert clean_db.open_position_count() == 0


def test_open_short_uses_margin_not_shares_for_count(clean_db):
    clean_db.open_short("BBBUSDT", 3.0, 50.0, "QF", 55.0, 45.0, margin_reserved=30.0)
    assert clean_db.open_short_count() == 1
    assert clean_db.open_position_count() == 0
    p = clean_db.get_short_position("BBBUSDT")
    assert p is not None
    assert p["shares"] == pytest.approx(3.0)
    assert p["margin_reserved"] == pytest.approx(30.0)
    clean_db.close_short("BBBUSDT")
    assert clean_db.get_short_position("BBBUSDT") is None


def test_open_short_twice_rejected(clean_db):
    """C3: re-shorting an already-open symbol must be rejected, not silently
    overwrite the row (which used to orphan the first short's margin_reserved
    from cash forever and reset its stop/opened_ts tracking)."""
    clean_db.open_short("BBBUSDT", 3.0, 50.0, "QF", 55.0, 45.0, margin_reserved=30.0)
    with pytest.raises(ValueError):
        clean_db.open_short("BBBUSDT", 2.0, 60.0, "QF", 66.0, 54.0, margin_reserved=24.0)
    # Original short must be completely untouched.
    p = clean_db.get_short_position("BBBUSDT")
    assert p["shares"] == pytest.approx(3.0)
    assert p["avg_cost"] == pytest.approx(50.0)
    assert p["margin_reserved"] == pytest.approx(30.0)
    assert clean_db.open_short_count() == 1


def test_entry_state_json_roundtrip(clean_db):
    state = [0.1, 0.2, 0.3]
    clean_db.open_position("AAAUSDT", 1.0, 100.0, "TEST", 95.0, 110.0, entry_state=state)
    assert clean_db.get_entry_state("AAAUSDT") == state


def test_mfe_mae_ratchet_long(clean_db):
    clean_db.open_position("AAAUSDT", 1.0, 100.0, "TEST", 95.0, 110.0)
    clean_db.update_mfe_mae("AAAUSDT", 105.0, 99.0, is_short=False)
    clean_db.update_mfe_mae("AAAUSDT", 103.0, 97.0, is_short=False)  # mfe holds, mae drops
    p = clean_db.get_position("AAAUSDT")
    assert p["mfe_price"] == pytest.approx(105.0)
    assert p["mae_price"] == pytest.approx(97.0)


def test_mfe_mae_ratchet_short_inverted(clean_db):
    clean_db.open_short("BBBUSDT", 1.0, 100.0, "QF", 105.0, 90.0, margin_reserved=20.0)
    clean_db.update_mfe_mae("BBBUSDT", 101.0, 96.0, is_short=True)
    clean_db.update_mfe_mae("BBBUSDT", 104.0, 98.0, is_short=True)
    p = clean_db.get_position("BBBUSDT")
    assert p["mfe_price"] == pytest.approx(96.0)    # lowest low = best cover
    assert p["mae_price"] == pytest.approx(104.0)   # highest high = worst squeeze


def test_trade_log_recent_trades_excludes_entry_legs(clean_db):
    clean_db.log_trade({"symbol": "AAAUSDT", "action": "buy", "status": "filled"})
    clean_db.log_trade({
        "symbol": "AAAUSDT", "action": "sell", "status": "filled",
        "pnl": 5.0, "net_pnl": 4.4, "gross_pnl": 5.6, "fee_total": 1.2,
    })
    clean_db.log_trade({"symbol": "AAAUSDT", "action": "buy", "status": "skipped",
                        "reason": "ml_gate_blocked"})
    recent = clean_db.get_recent_trades()
    assert len(recent) == 1                     # only the exit leg with net_pnl
    assert recent[0]["action"] == "sell"
    assert clean_db.get_filled_trade_count() == 2


def test_increment_candle_count(clean_db):
    clean_db.open_position("AAAUSDT", 1.0, 100.0, "TEST", 95.0, 110.0)
    assert clean_db.increment_candle_count("AAAUSDT") == 1
    assert clean_db.increment_candle_count("AAAUSDT") == 2


def test_brain_key_roundtrip_and_default(clean_db):
    clean_db.save_brain_key("k1", {"a": [1, 2], "b": "x"})
    assert clean_db.load_brain_key("k1") == {"a": [1, 2], "b": "x"}
    assert clean_db.load_brain_key("missing", default="fallback") == "fallback"


def test_equity_rebase_roundtrip_and_default(clean_db):
    assert clean_db.get_equity_rebase() == (None, None)
    clean_db.set_equity_rebase(500.0, "2026-09-30T00:00:00+00:00")
    baseline, ts = clean_db.get_equity_rebase()
    assert baseline == pytest.approx(500.0)
    assert ts == "2026-09-30T00:00:00+00:00"


def test_record_equity_uses_rebase_baseline_when_set(clean_db):
    """record_equity()'s return_pct must use STARTING_CASH until a rebase is
    declared, then switch to the rebase baseline -- without a rebase being
    present, existing behaviour must stay byte-for-byte identical."""
    clean_db.record_equity(9_000.0)
    assert float(clean_db.get_portfolio_stat("return_pct")) == pytest.approx(
        (9_000.0 - STARTING_CASH) / STARTING_CASH * 100
    )

    clean_db.set_equity_rebase(50.0, "2026-09-30T00:00:00+00:00")
    clean_db.record_equity(55.0)
    assert float(clean_db.get_portfolio_stat("return_pct")) == pytest.approx(
        (55.0 - 50.0) / 50.0 * 100
    )


def test_candle_upsert_replaces_same_ts(clean_db):
    c = {"time": 1_700_000_000_000, "open": 1.0, "high": 2.0, "low": 0.5,
         "close": 1.5, "volume": 10.0}
    clean_db.upsert_candle("AAAUSDT", c)
    c2 = dict(c, close=1.7)
    clean_db.upsert_candle("AAAUSDT", c2)
    rows = clean_db.get_candles("AAAUSDT")
    assert len(rows) == 1
    assert rows[0]["close"] == pytest.approx(1.7)


def test_zombie_seed_rows_hidden_from_active_positions(clean_db):
    # _seed_portfolio inserts shares=0 placeholder rows; they must not count.
    assert clean_db.get_all_positions() == []
    assert clean_db.open_position_count() == 0
