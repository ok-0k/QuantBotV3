"""
Money path, booking: bot._prepare_trade / bot._commit_trade outcomes that the
paper adapter never produces — rejected and failed orders, fills away from the
requested price, partial covers, cash consumed during the exchange
round-trip — plus the sizing and journal refusals of the short entry, and the
reward and RL-log side effects of exits. Each test asserts cash, position,
order-journal state and the trade row.
"""

import json

import numpy as np
import pytest

import bot
import db
from config import (FEE_GATE_ROUND_TRIP, MICRO_WIN_REWARD_PENALTY, SHORT_MARGIN_PCT,
                    SLIPPAGE_PCT, STARTING_CASH)
from conftest import make_candles
from execution.base import Fill, make_client_order_id

ZERO_STATE = np.zeros(13, dtype=np.float32)
SYM = "AAAUSDT"


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    monkeypatch.setattr(bot, "V3_FIXES", True)
    return clean_db


class Adapter:
    """Scripted exchange: fill(intent) -> Fill; records every intent."""
    name = "paper"

    def __init__(self, fill):
        self.fill, self.intents = fill, []

    def execute(self, intent):
        self.intents.append(intent)
        return self.fill(intent)


def _use(monkeypatch, fill):
    ad = Adapter(fill)
    monkeypatch.setattr(bot, "_exec_adapter", ad)
    return ad


def _candles(price=100.0):
    return make_candles(n=120, start_price=price, drift=-0.05, amplitude=0.4)


def _ens(action, price, ml_prob=None, regime="trend_down"):
    if ml_prob is None:
        ml_prob = 0.03 if action == "short" else 0.97
    return {"signal": action, "symbol": SYM, "price": price, "regime": regime, "on_fire": False,
            "position_size_mult": 1.0, "buy_weight": 0.5, "sell_weight": 0.5, "ml_prob": ml_prob,
            "breakdown": [], "time": "2026-10-21T00:00:00Z"}


def _exit(action, price):
    e = _ens(action, price)
    e.pop("ml_prob")
    return e


def _last_row(db):
    with db.get_db() as conn:
        return dict(conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 1").fetchone())


def _order(action, candles):
    return db.get_order(make_client_order_id(SYM, action, candles[-1]["time"]))


def _open_short(db, shares=10.0, entry=100.0, margin=200.0, **feats):
    db.open_short(SYM, shares, entry, "QF", entry * 1.03, entry * 0.9, margin,
                  entry_features=feats or None)
    db.set_cash(STARTING_CASH - margin)


# ── entries the exchange did not execute ─────────────────────────────────────

@pytest.mark.parametrize("action", ["short", "buy"])
@pytest.mark.parametrize("status, journal", [("REJECTED", "REJECTED"), ("FAILED", "FAILED"),
                                             ("EXPIRED", "FAILED")])
def test_unexecuted_entry_books_nothing_and_closes_the_journal(fresh, monkeypatch, action, status, journal):
    _use(monkeypatch, lambda i: Fill(status=status, note="exchange said no"))
    c = _candles()
    bot._execute_trade(_ens(action, c[-1]["close"]), "QF", c, 0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)
    row = _last_row(fresh)
    assert (row["status"], row["reason"]) == ("skipped", f"execution_{status.lower()}")
    o = _order(action, c)
    assert o["state"] == journal and o["note"] == "exchange said no"
    assert int(fresh.get_portfolio_stat("total_trades", "0")) == 0


# ── entries filled away from the requested price ─────────────────────────────

@pytest.mark.parametrize("action, side, move", [("short", "short", 0.99), ("buy", "long", 1.01)])
def test_entry_filled_away_from_limit_rederives_levels_at_the_fill(fresh, monkeypatch, action, side, move):
    calls = []
    real_st, real_rr = bot.brain.get_stop_take, bot.apply_dynamic_rr

    def spy_st(entry, *a, **k):
        out = real_st(entry, *a, **k)
        calls.append(("st", entry))
        return out

    def spy_rr(**k):
        out = real_rr(**k)
        calls.append(("rr", k["entry"], out))
        return out
    monkeypatch.setattr(bot.brain, "get_stop_take", spy_st)
    monkeypatch.setattr(bot, "apply_dynamic_rr", spy_rr)
    ad = _use(monkeypatch, lambda i: Fill(status="FILLED", qty=i.qty, price=i.limit_price * move))
    c = _candles()
    bot._execute_trade(_ens(action, c[-1]["close"]), "QF", c, 0.5, ZERO_STATE, STARTING_CASH)

    intent = ad.intents[0]
    fill_px = intent.limit_price * move
    pos = fresh.get_position(SYM)
    assert pos["side"] == side and pos["avg_cost"] == pytest.approx(fill_px)
    # prepare derived levels at the limit, commit re-derived them at the fill
    assert [x[1] for x in calls if x[0] == "st"] == [pytest.approx(intent.limit_price), pytest.approx(fill_px)]
    stop, tp = calls[-1][2]
    assert (pos["stop_price"], pos["tp_price"]) == (pytest.approx(stop), pytest.approx(tp))
    value = intent.qty * fill_px
    if side == "short":
        assert tp < fill_px < stop
        assert pos["margin_reserved"] == pytest.approx(value * SHORT_MARGIN_PCT)
        assert fresh.get_cash() == pytest.approx(STARTING_CASH - value * SHORT_MARGIN_PCT)
        feats = json.loads(pos["entry_features"])
        assert feats["initial_stop"] == pytest.approx(stop) and feats["initial_tp"] == pytest.approx(tp)
        assert feats["stop_pct"] == pytest.approx(round((stop - fill_px) / fill_px * 100, 4))
    else:
        assert stop < fill_px < tp
        assert fresh.get_cash() == pytest.approx(STARTING_CASH - value)
    assert _last_row(fresh)["exec_price"] == pytest.approx(fill_px)


@pytest.mark.parametrize("action, text", [("short", "SHORT AAAUSDT committed with cash"),
                                          ("buy", "BUY AAAUSDT committed with cash")])
def test_fill_is_booked_even_if_cash_was_spent_during_the_round_trip(fresh, monkeypatch, caplog, action, text):
    """The order already executed: booking must not be skipped. Cash goes
    negative and the warning names the cause."""
    def fill(i):
        fresh.set_cash(5.0)                               # other symbols spent it meanwhile
        return Fill(status="FILLED", qty=i.qty, price=i.limit_price)
    ad = _use(monkeypatch, fill)
    c = _candles()
    with caplog.at_level("WARNING"):
        bot._execute_trade(_ens(action, c[-1]["close"]), "QF", c, 0.5, ZERO_STATE, STARTING_CASH)
    i = ad.intents[0]
    debit = i.qty * i.limit_price * (SHORT_MARGIN_PCT if action == "short" else 1.0)
    assert fresh.get_position(SYM) is not None
    assert fresh.get_cash() == pytest.approx(5.0 - debit)
    assert any(text in r.getMessage() for r in caplog.records)


# ── short entry refusals inside _prepare_trade ───────────────────────────────

def test_short_without_margin_cash_is_refused(fresh):
    fresh.set_cash(5.0)
    c = _candles()
    bot._execute_trade(_ens("short", c[-1]["close"]), "QF", c, 0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == [] and fresh.get_cash() == pytest.approx(5.0)
    assert _last_row(fresh)["reason"] == "insufficient_margin"


def test_short_whose_target_cannot_clear_fees_is_refused(fresh, caplog):
    """Quiet tape (target at its ~3% floor) on $70 of equity, where the 15%
    order cap sizes a $10.50 short: the target grosses about $0.33, under the
    $0.35 minimum expected net edge."""
    c = make_candles(n=120, start_price=100.0, drift=0.0, amplitude=0.0005)
    with caplog.at_level("INFO"):
        bot._execute_trade(_ens("short", c[-1]["close"]), "QF", c, 0.5, ZERO_STATE, 70.0)
    assert fresh.get_all_positions() == [] and fresh.get_cash() == pytest.approx(STARTING_CASH)
    assert _last_row(fresh)["reason"] == "edge_too_small"
    assert any("SHORT AAAUSDT blocked — projected TP net edge" in r.getMessage() for r in caplog.records)


def test_second_short_signal_on_the_same_candle_is_a_duplicate(fresh):
    c = _candles()
    coid = make_client_order_id(SYM, "short", c[-1]["time"])
    assert db.try_create_order(coid, SYM, "short", "short", c[-1]["time"], 1.0, 99.0, "paper")
    bot._execute_trade(_ens("short", c[-1]["close"]), "QF", c, 0.5, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == [] and fresh.get_cash() == pytest.approx(STARTING_CASH)
    assert _last_row(fresh)["reason"] == "duplicate_order_blocked"
    o = db.get_order(coid)
    assert o["state"] == "PENDING_NEW" and o["req_qty"] == 1.0          # the first claim, untouched


def test_unknown_action_is_journaled_and_moves_no_money(fresh):
    c = _candles()
    bot._execute_trade(_exit("flip", c[-1]["close"]), "QF", c, None, ZERO_STATE, STARTING_CASH)
    assert fresh.get_cash() == pytest.approx(STARTING_CASH)
    assert _last_row(fresh)["reason"] == "unknown_action:flip"


# ── covers ───────────────────────────────────────────────────────────────────

def test_unexecuted_cover_keeps_the_short_and_its_margin(fresh, monkeypatch, caplog):
    _open_short(fresh)
    _use(monkeypatch, lambda i: Fill(status="FAILED", note="timeout"))
    c = _candles(95.0)
    with caplog.at_level("CRITICAL"):
        bot._execute_trade(_exit("cover", 95.0), "QF", c, None, ZERO_STATE, STARTING_CASH)
    pos = fresh.get_short_position(SYM)
    assert pos["shares"] == 10.0 and pos["margin_reserved"] == 200.0
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 200.0)
    assert _last_row(fresh)["reason"] == "execution_failed"
    assert _order("cover", c)["state"] == "FAILED"
    assert any("COVER AAAUSDT DID NOT EXECUTE" in r.getMessage() for r in caplog.records)


def test_unexecuted_cover_leaves_a_previous_attempts_journal_row_alone(fresh, monkeypatch):
    """Exits are fail-open on the journal: a crashed earlier attempt holds the
    id, the cover still goes out, and its failure does not overwrite that row."""
    _open_short(fresh)
    c = _candles(95.0)
    coid = make_client_order_id(SYM, "cover", c[-1]["time"])
    assert db.try_create_order(coid, SYM, "cover", "short", c[-1]["time"], 10.0, 95.0, "paper")
    ad = _use(monkeypatch, lambda i: Fill(status="FAILED", note="timeout"))
    bot._execute_trade(_exit("cover", 95.0), "QF", c, None, ZERO_STATE, STARTING_CASH)
    assert len(ad.intents) == 1                                       # the exit was attempted
    assert db.get_order(coid)["state"] == "PENDING_NEW"
    assert fresh.get_short_position(SYM)["shares"] == 10.0


def test_partial_cover_releases_margin_pro_rata(fresh, monkeypatch, caplog):
    _open_short(fresh, shares=10.0, entry=100.0, margin=200.0)
    _use(monkeypatch, lambda i: Fill(status="PARTIALLY_FILLED", qty=4.0, price=i.limit_price))
    with caplog.at_level("WARNING"):
        bot._execute_trade(_exit("cover", 95.0), "QF", _candles(95.0), None, ZERO_STATE, STARTING_CASH)
    exec_px = 95.0 * (1 + SLIPPAGE_PCT)
    gross = 4.0 * (100.0 - exec_px)
    net = gross - (4.0 * 100.0 + 4.0 * exec_px) / 2.0 * FEE_GATE_ROUND_TRIP
    pos = fresh.get_short_position(SYM)
    assert pos["shares"] == pytest.approx(6.0)
    assert pos["margin_reserved"] == pytest.approx(120.0)                    # 60% stays reserved
    assert fresh.get_cash() == pytest.approx(STARTING_CASH - 200.0 + 80.0 + net, rel=1e-9)
    assert float(fresh.get_portfolio_stat("realised_pnl")) == pytest.approx(gross)
    row = _last_row(fresh)
    assert row["status"] == "filled" and row["shares"] == 4.0 and row["net_pnl"] == pytest.approx(round(net, 2))
    assert any("COVER AAAUSDT PARTIAL fill" in r.getMessage() for r in caplog.records)


def _price_for_net(entry, shares, target_net, cover):
    """Raw exit price whose booked net P&L is target_net (net is linear in price)."""
    def net(raw):
        px = raw * (1 + SLIPPAGE_PCT) if cover else raw * (1 - SLIPPAGE_PCT)
        gross = shares * ((entry - px) if cover else (px - entry))
        return bot.net_realized_pnl(gross, shares * entry, shares * px)[1]
    a, b = entry * 0.9, entry * 1.1
    na, nb = net(a), net(b)
    return a + (target_net - na) * (b - a) / (nb - na)


@pytest.mark.parametrize("action", ["cover", "sell"])
def test_micro_win_is_penalised_in_the_reward_not_in_cash(fresh, monkeypatch, action):
    rewards = []
    monkeypatch.setattr(bot.brain, "reward", lambda **k: rewards.append(k))
    if action == "cover":
        _open_short(fresh, shares=10.0, entry=100.0, margin=200.0)
    else:
        fresh.open_position(SYM, 10.0, 100.0, "QF", 95.0, 110.0)
        fresh.set_cash(STARTING_CASH - 1000.0)
    cover = action == "cover"
    raw = _price_for_net(100.0, 10.0, 0.02, cover=cover)
    cash0 = fresh.get_cash()
    bot._execute_trade(_exit(action, raw), "QF", _candles(raw), None, ZERO_STATE, STARTING_CASH)
    px = raw * (1 + SLIPPAGE_PCT) if cover else raw * (1 - SLIPPAGE_PCT)
    gross = 10.0 * ((100.0 - px) if cover else (px - 100.0))
    net = bot.net_realized_pnl(gross, 1000.0, 10.0 * px)[1]
    assert net == pytest.approx(0.02) and 0 < net < bot.MICRO_WIN_USD
    assert fresh.get_cash() == pytest.approx(cash0 + (200.0 if cover else 1000.0) + net)   # cash gets the full net
    shaped = bot._shaped_reward_v2(gross, 0, 1000.0)
    assert rewards[0]["pnl"] == pytest.approx(shaped - MICRO_WIN_REWARD_PENALTY)
    assert rewards[0]["side"] == ("short" if action == "cover" else "long")


@pytest.mark.parametrize("action", ["cover", "sell"])
def test_rl_log_failure_never_undoes_the_booking(fresh, monkeypatch, caplog, action):
    def boom(**k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(bot, "log_rl_experience", boom)
    if action == "cover":
        _open_short(fresh)
    else:
        fresh.open_position(SYM, 10.0, 100.0, "QF", 95.0, 110.0)
        fresh.set_cash(STARTING_CASH - 1000.0)
    with caplog.at_level("WARNING"):
        bot._execute_trade(_exit(action, 95.0), "QF", _candles(95.0), None, ZERO_STATE, STARTING_CASH)
    assert fresh.get_all_positions() == []
    row = _last_row(fresh)
    assert row["status"] == "filled"
    assert fresh.get_cash() == pytest.approx(STARTING_CASH + row["net_pnl"], abs=0.01)   # margin/cost back + net
    assert any(f"log_rl_experience failed for {action.upper()}" in r.getMessage() for r in caplog.records)


def test_rl_transition_reads_entry_state_and_tolerates_corrupt_snapshots(fresh, monkeypatch):
    rewards = []
    monkeypatch.setattr(bot.brain, "reward", lambda **k: rewards.append(k))
    stored = [float(i) / 10 for i in range(13)]
    _open_short(fresh, sac_fraction=0.42)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET entry_state=? WHERE symbol=?", (json.dumps(stored), SYM))
    bot._execute_trade(_exit("cover", 95.0), "QF", _candles(95.0), None, ZERO_STATE, STARTING_CASH)
    assert np.allclose(rewards[0]["state"], stored) and rewards[0]["action"] == pytest.approx(0.42)

    _open_short(fresh)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET entry_state=?, entry_features=? WHERE symbol=?",
                     ("not json", "[1, 2]", SYM))
    bot._execute_trade(_exit("cover", 94.0), "QF", _candles(94.0), None, ZERO_STATE, STARTING_CASH)
    assert np.array_equal(rewards[1]["state"], ZERO_STATE) and rewards[1]["action"] == 0.0
    assert fresh.get_all_positions() == []
