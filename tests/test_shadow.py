"""
Shadow learner (shadow.py + the bot hooks): refused shorts are captured in
memory only, resolved on closed 1m candles with the bot's own exit rules and
cost booking, never across a missing candle, and judged only at the two fixed
evaluation dates. Demotion only ever removes permission to trade.
"""

import asyncio
import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
import shadow
from accounting_v2 import net_realized_pnl
from config import (HARD_STOP_LOSS_USD, MAX_HOLD_OPEN_SECONDS, SHADOW_BORROW_DAILY,
                    SHADOW_MAX_HOLD_HOURS, SHADOW_NOTIONAL_USD, SLIPPAGE_PCT, SYMBOLS)
from conftest import make_candles

SYM = SYMBOLS[0]
NAME = "QF_RA_S7_M87"
M = shadow.MINUTE_MS
T_START = 1_790_000_040_000 // M * M          # a fixed minute (ms)
RULES = bot._SHADOW_RULES


# ── fixtures / helpers ───────────────────────────────────────────────────────

@pytest.fixture()
def fresh(clean_db, monkeypatch):
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    monkeypatch.setattr(bot, "_symbol_last_trade", {})
    monkeypatch.setattr(bot, "_entry_block_journaled", {})
    monkeypatch.setattr(bot, "_entry_in_flight", {})
    monkeypatch.setattr(bot, "_evidence_blocklist", set())
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", True)
    monkeypatch.setattr(bot, "SHADOW_LEARNER", True)
    monkeypatch.setattr(shadow, "SHADOW_LEARNER", True)
    return clean_db


def _short_ens(symbol, price, strategy=NAME):
    return {"signal": "short", "symbol": symbol, "price": price, "regime": "ranging",
            "on_fire": False, "position_size_mult": 1.0, "buy_weight": 0.0, "sell_weight": 0.6,
            "ml_prob": 0.05, "breakdown": [{"signal": "sell", "strategy": strategy, "alloc": 0.7}],
            "time": "2026-10-21T00:00:00Z", "meta": {"short_score": 0.7}}


@pytest.fixture()
def harness(fresh, monkeypatch):
    async def _no_margin(*a, **k):
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", _no_margin)
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: 0.5)

    def evaluate(symbol=SYM, strategy=NAME):
        candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
        monkeypatch.setattr(bot.brain, "get_ensemble_signal",
                            lambda *a, **k: _short_ens(symbol, candles[-1]["close"], strategy))

        async def run():
            await bot._evaluate_and_trade(symbol, candles, 0.05, asyncio.get_running_loop())
        t = time.perf_counter()
        asyncio.run(run())
        return candles, time.perf_counter() - t
    return evaluate


def _rows(db, sql="SELECT * FROM trades ORDER BY id", args=()):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(sql, args)]


def bars(spec, start=T_START, base=100.0):
    """spec: list of (open, high, low, close) tuples; flat 'base' bars if a float."""
    out = []
    for i, k in enumerate(spec):
        o, h, l, c = (k, k, k, k) if isinstance(k, (int, float)) else k
        out.append({"time": start + i * M, "open": o, "high": h, "low": l, "close": c, "volume": 1.0})
    return out


def history(n=40, base=100.0, rng=0.5, start=T_START - 40 * M):
    """Closed candles before the trade, with an ATR(14) of about `rng`."""
    return [{"time": start + i * M, "open": base, "high": base + rng / 2, "low": base - rng / 2,
             "close": base, "volume": 1.0} for i in range(n)]


def trade(entry=100.0, stop0=103.5, tp0=90.0, opened_ms=None):
    o = opened_ms if opened_ms is not None else T_START - 1_000
    return {"entry": entry, "stop": stop0, "tp": tp0, "stop0": stop0, "tp0": tp0,
            "opened_ms": o, "last_ms": T_START - M}


def run_series(t, series, now_ms=None, rules=RULES):
    full = history() + series
    return shadow.resolve_trade(t, full, rules, now_ms if now_ms is not None else series[-1]["time"] + 10 * M)


# ── capture: in memory only, never changes the real decision ─────────────────

def test_refused_short_is_captured_with_the_real_entry_levels(harness, fresh):
    candles, _ = harness()
    rows = _rows(fresh)
    assert [(r["status"], r["reason"]) for r in rows] == [("skipped", "no_validated_edge")]
    q = list(shadow._queue)
    assert len(q) == 1 and q[0]["symbol"] == SYM and q[0]["strategy"] == NAME
    price = candles[-1]["close"]
    entry = price * (1 - SLIPPAGE_PCT)
    conf = 1.0 * bot.dynamic_conviction_size_mult(0.05, "short")
    stop, tp = bot.brain.get_stop_take(entry, candles, False, side="short", regime="ranging",
                                       confidence_mult=conf)
    stop, tp = bot.apply_dynamic_rr(entry=entry, stop=stop, take=tp, atr=bot._wilder_atr(candles),
                                    side="short")
    assert q[0]["entry"] == pytest.approx(entry)
    assert (q[0]["stop0"], q[0]["tp0"]) == pytest.approx((stop, tp))
    assert q[0]["last_ms"] == q[0]["opened_ms"] // M * M - M          # resolution starts at the entry minute
    assert _rows(fresh, "SELECT * FROM shadow_trades") == []          # nothing written yet


def test_capture_failure_cannot_change_the_decision(harness, fresh, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("shadow broke")
    monkeypatch.setattr(shadow, "capture", boom)
    harness()
    assert fresh.get_all_positions() == []
    assert [r["reason"] for r in _rows(fresh)] == ["no_validated_edge"]


def test_capture_never_touches_the_database(harness, fresh, monkeypatch):
    calls = []
    real = shadow.process
    monkeypatch.setattr(shadow, "process", lambda *a, **k: calls.append(1) or real(*a, **k))
    _, elapsed = harness()
    assert calls == [] and len(shadow._queue) == 1


def test_a_slow_shadow_database_cannot_delay_a_real_decision(harness, fresh, monkeypatch):
    def slow_job(*a, **k):
        time.sleep(1.5)
        return {}, [], []
    monkeypatch.setattr(bot, "_shadow_job", slow_job)
    candles = make_candles(n=120, start_price=100.0, drift=-0.05, amplitude=0.4)
    monkeypatch.setattr(bot.brain, "get_ensemble_signal",
                        lambda *a, **k: _short_ens(SYM, candles[-1]["close"]))

    async def _no_margin(*a, **k):
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", _no_margin)
    monkeypatch.setattr(bot, "compute_position_fraction", lambda state: 0.5)
    done = {}

    async def entry():
        await asyncio.sleep(0.05)                    # housekeeping is already in the thread
        t = time.perf_counter()
        await bot._evaluate_and_trade(SYM, candles, 0.05, asyncio.get_running_loop())
        done["entry"] = time.perf_counter() - t

    async def main():
        await asyncio.gather(bot._shadow_housekeeping(timeout=5), entry())
    asyncio.run(main())
    assert done["entry"] < 1.0                        # not held behind the 1.5 s shadow job
    assert [r["reason"] for r in _rows(fresh)] == ["no_validated_edge"]


def test_shadow_timeout_raises_out_of_housekeeping_only(fresh, monkeypatch):
    monkeypatch.setattr(bot, "_shadow_job", lambda *a, **k: time.sleep(0.5) or ({}, [], []))
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(bot._shadow_housekeeping(timeout=0.05))


def test_flag_off_captures_nothing(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "SHADOW_LEARNER", False)
    monkeypatch.setattr(shadow, "SHADOW_LEARNER", False)
    harness()
    assert len(shadow._queue) == 0
    assert [r["reason"] for r in _rows(fresh)] == ["no_validated_edge"]


def test_no_capture_when_the_entry_is_allowed(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "ENTRY_EVIDENCE_GATE", False)
    harness()
    assert fresh.get_short_position(SYM) is not None
    assert len(shadow._queue) == 0


def test_housekeeping_snapshot_excludes_the_live_minute(fresh, monkeypatch):
    seen = {}

    def job(candles, now_ms):
        seen.update(candles=candles, now_ms=now_ms)
        return {}, [], []
    monkeypatch.setattr(bot, "_shadow_job", job)
    now = int(time.time() * 1000) // M * M
    live = [{"time": now - 2 * M, "open": 1, "high": 1, "low": 1, "close": 1},
            {"time": now - M, "open": 1, "high": 1, "low": 1, "close": 1},
            {"time": now, "open": 1, "high": 1, "low": 1, "close": 1}]          # still forming
    monkeypatch.setattr(bot, "_candles_cache", {SYM: live})
    shadow.capture({"x": 1})
    asyncio.run(bot._shadow_housekeeping(timeout=5))
    times = [c["time"] for c in seen["candles"][SYM]]
    assert now not in times and (now - 2 * M) in times
    assert len(shadow._queue) == 1                  # only process() drains, under its lock
    assert json.loads(fresh.get_portfolio_stat(shadow.STATS_KEY))["at"]


# ── evidence gate: runtime demotion blocklist ────────────────────────────────

def test_blocklisted_validated_name_is_refused(harness, fresh, monkeypatch):
    monkeypatch.setattr(bot, "VALIDATED_STRATEGIES", frozenset({NAME}))
    monkeypatch.setattr(bot, "_evidence_blocklist", {NAME})
    harness()
    assert fresh.get_short_position(SYM) is None
    assert [r["reason"] for r in _rows(fresh)] == ["no_validated_edge"]
    monkeypatch.setattr(bot, "_evidence_blocklist", set())
    bot._symbol_last_trade.clear()
    harness()
    assert fresh.get_short_position(SYM) is not None


def _cover(db, name, pnl, i):
    db.log_trade({"symbol": SYM, "action": "cover", "strategy": name, "regime": "ranging",
                  "price": 100.0, "exec_price": 100.0, "timestamp": f"2026-10-21T00:{i % 60:02d}:00+00:00",
                  "side": "short", "status": "filled", "shares": 1.0, "trade_value": 100.0,
                  "pnl": pnl, "net_pnl": pnl, "entry_features": {"strategy": name}})


@pytest.mark.parametrize("n, pnl, demoted", [(50, -0.2, True), (49, -0.2, False), (50, 0.3, False), (50, 0.0, True)])
def test_demotion_floor(fresh, n, pnl, demoted):
    for i in range(n):
        _cover(fresh, NAME, pnl, i)
    with fresh.get_db() as conn:
        assert shadow.demotion_check(conn, frozenset({NAME})) == ([NAME] if demoted else [])
        assert shadow.blocklist(conn) == ([NAME] if demoted else [])
        assert shadow.demotion_check(conn, frozenset({NAME})) == []        # already blocklisted
    with fresh.get_db() as conn:
        assert shadow.demotion_check(conn, frozenset()) == []


def test_housekeeping_publishes_the_blocklist_to_the_gate(fresh, monkeypatch):
    monkeypatch.setattr(bot, "VALIDATED_STRATEGIES", frozenset({NAME}))
    for i in range(50):
        _cover(fresh, NAME, -1.0, i)
    monkeypatch.setattr(bot, "_candles_cache", {})
    asyncio.run(bot._shadow_housekeeping(timeout=10))
    assert bot._evidence_blocklist == {NAME}


# ── resolution on closed candles ─────────────────────────────────────────────

def test_stop_hit_exits_at_the_stop_or_the_open_if_it_gapped():
    t = trade()
    assert run_series(t, bars([(100, 103.6, 99.9, 101)])) == ("stop_loss_short", 103.5, T_START)
    t = trade()
    assert run_series(t, bars([(104, 104.5, 103.9, 104.2)])) == ("stop_loss_short", 104, T_START)


def test_stop_beats_take_profit_in_the_same_candle():
    t = trade()
    assert run_series(t, bars([(100, 104, 89, 95)]))[0] == "stop_loss_short"


def test_take_profit_fills_at_the_target_when_fees_clear():
    t = trade()
    assert run_series(t, bars([(100, 100.2, 89.5, 92)])) == ("take_profit_short", 90.0, T_START)


def test_fee_gated_take_profit_does_not_exit():
    t = trade(tp0=99.9)                         # 0.1% target: cannot clear 0.20% fees + slippage
    assert run_series(t, bars([(100, 100.1, 99.8, 99.95)] + [100.0] * 3)) is None
    assert t["last_ms"] == T_START + 3 * M


def test_trail_only_matters_when_the_target_is_fee_gated(monkeypatch):
    gated = shadow.Rules(trail=RULES.trail, breakeven=RULES.breakeven, decay=RULES.decay,
                         clears_fees=lambda *a: False, atr=RULES.atr)
    t = trade()
    # low 89.5 passes the 90 target (fee-gated here): stop trails to 89.5 + ATR ~0.5 = ~90,
    # and the close at 92 is above the moved stop -> stop on the mark
    out = run_series(t, bars([(100, 100.2, 89.5, 92)]), rules=gated)
    assert out[0] == "stop_loss_short" and out[1] == 92


def test_break_even_moves_the_stop_to_entry_then_it_exits_there():
    t = trade(stop0=103.5, tp0=80.0)
    # ATR incl. this candle ~0.72: wick 4.5 ATR down (to <= ~96.75), close >= 1.5 ATR in profit
    out = run_series(t, bars([(100, 100.1, 96.5, 98.0), (98, 100.3, 97.9, 100.1)]))
    assert out == ("stop_loss_short", 100.0, T_START + M)


def test_entry_minute_is_tested_for_the_stop_only():
    entry_minute = T_START
    t = trade(opened_ms=entry_minute + 20_000)
    t["last_ms"] = entry_minute - M
    # entry minute touches the target and wicks below it: never a take-profit
    assert run_series(t, bars([(100, 100.2, 89.0, 99.5)])) is None
    assert (t["stop"], t["tp"]) == (103.5, 90.0)                             # no level moves either
    t = trade(opened_ms=entry_minute + 20_000)
    t["last_ms"] = entry_minute - M
    assert run_series(t, bars([(100, 103.8, 99.0, 101)])) == ("stop_loss_short", 103.5, entry_minute)


def test_designed_decay_matches_the_bot_helper():
    t = trade(opened_ms=T_START - 59 * M)        # 60 min old after the first candle
    assert run_series(t, bars([100.0])) is None
    exp_tp, exp_sl = bot.designed_decay_moves(100.0, 90.0, 103.5, 60.0, 103.5, 90.0, True)
    assert (t["tp"], t["stop"]) == pytest.approx((exp_tp, exp_sl))


def test_losing_trade_is_closed_at_four_hours_and_a_winner_is_not():
    age0 = T_START - int(MAX_HOLD_OPEN_SECONDS * 1000) + M
    t = trade(opened_ms=age0)
    assert run_series(t, bars([100.5])) == ("time_hold_exit", 100.5, T_START)
    t = trade(opened_ms=age0)
    assert run_series(t, bars([99.5])) is None


def test_shadow_cap_closes_after_max_hold():
    t = trade(opened_ms=T_START - int(SHADOW_MAX_HOLD_HOURS * 3_600_000) + M)
    assert run_series(t, bars([99.5])) == ("shadow_max_hold", 99.5, T_START)


def test_survival_stop_on_a_huge_adverse_close():
    t = trade(stop0=200.0, tp0=50.0)
    px = 100.0 - HARD_STOP_LOSS_USD / (SHADOW_NOTIONAL_USD / 100.0) + 0.5      # beyond the $15 floor
    assert run_series(t, bars([(100, px, 100, px)])) == ("hard_stop_survival", px, T_START)


def test_never_resolves_across_a_missing_candle():
    t = trade()
    series = bars([100.0]) + bars([100.0, 103.9], start=T_START + 2 * M)   # T_START+M missing
    now = T_START + 2 * M + 60_000
    assert run_series(t, series, now_ms=now) is None                         # inside the grace
    assert t["last_ms"] == T_START                                           # stopped before the hole
    out = run_series(t, series, now_ms=T_START + M + shadow.GAP_GRACE_MS + 1)
    assert out == ("gap_stop", 103.5, T_START + M)                           # closed at the stop


def test_no_candles_at_all_closes_at_the_stop_after_30_minutes():
    t = trade()
    assert shadow.resolve_trade(t, [], RULES, T_START + 10 * M) is None
    assert shadow.resolve_trade(t, [], RULES, T_START + shadow.NO_DATA_CLOSE_MS + M)[0] == "gap_stop"


def test_rules_run_in_the_tick_path_order():
    calls = []
    spy = shadow.Rules(
        trail=lambda *a: calls.append("trail") or 102.0,
        breakeven=lambda *a: calls.append(("be", a[2])) or (101.0, False),
        decay=lambda *a: calls.append(("decay", a[4])) or (None, 100.9),
        clears_fees=lambda *a: True,
        atr=lambda c: 0.5)
    t = trade()
    assert shadow.step(t, bars([100.0])[0], history(), spy) is None
    assert calls == ["trail", ("be", 102.0), ("decay", 101.0)]
    assert t["stop"] == 100.9


# ── cost booking: the bot's own ──────────────────────────────────────────────

def test_booking_uses_slippage_fees_and_borrow():
    entry, exit_px, hold, stop0 = 99.92, 95.0, 720.0, 103.4
    net, r = shadow.book_short(entry, exit_px, hold, stop0)
    shares = SHADOW_NOTIONAL_USD / entry
    fill = exit_px * (1 + SLIPPAGE_PCT)
    _, exp = net_realized_pnl((entry - fill) * shares, shares * entry, shares * fill)
    exp -= shares * entry * SHADOW_BORROW_DAILY * 0.5
    assert net == pytest.approx(exp)
    assert r == pytest.approx(exp / ((stop0 - entry) * shares))


def test_a_stopped_trade_loses_more_than_one_r_after_costs():
    net, r = shadow.book_short(100.0, 103.5, 30, 103.5)
    assert r < -1.0 and net < 0


# ── process(): inserts, dedupe, T0, resolution ───────────────────────────────

def _rec(ts="2026-10-21T00:00:00+00:00", strategy=NAME, symbol=SYM, entry=100.0, stop0=103.5, tp0=90.0):
    return {"opened_ts": ts, "opened_ms": T_START - 1_000, "symbol": symbol, "strategy": strategy,
            "regime": "ranging", "entry": entry, "stop0": stop0, "tp0": tp0, "last_ms": T_START - M}


def test_process_inserts_dedupes_sets_t0_once_and_resolves(fresh):
    s = shadow.process({SYM: history() + bars([100.0])}, RULES, T_START + 3 * M, fresh,
                       records=[_rec(), _rec(ts="2026-10-21T00:01:00+00:00"), _rec(strategy="QF_TR_S9_M99")])
    assert (s["opened"], s["duplicates"], s["errors"]) == (2, 1, 0)
    assert fresh.get_portfolio_stat(shadow.T0_KEY) == "2026-10-21T00:00:00+00:00"
    shadow.process({}, RULES, T_START, fresh, records=[_rec(ts="2026-10-22T00:00:00+00:00", symbol=SYMBOLS[1])])
    assert fresh.get_portfolio_stat(shadow.T0_KEY) == "2026-10-21T00:00:00+00:00"   # written once
    s = shadow.process({SYM: history() + bars([100.0, (100, 103.6, 99, 101)])}, RULES,
                       T_START + 5 * M, fresh, records=[])
    assert s["closed"] == 2
    rows = _rows(fresh, "SELECT * FROM shadow_trades WHERE symbol=? ORDER BY id", (SYM,))
    assert all(r["status"] == "closed" and r["exit_reason"] == "stop_loss_short" for r in rows)
    assert all(r["r_net"] < -1 and r["hold_min"] > 0 for r in rows)


def test_process_survives_a_bad_row_and_counts_it(fresh):
    with fresh.get_db() as conn:
        conn.execute("INSERT INTO shadow_trades (opened_ts, opened_ms, symbol, strategy, entry, stop0, tp0, "
                     "stop, tp, last_ms) VALUES ('x', 0, ?, 'N', 0, 0, 0, 0, 0, 0)", (SYM,))
    s = shadow.process({SYM: [{"time": M, "open": "bad"}]}, RULES, 10 ** 13, fresh, records=[])
    assert s["errors"] >= 1


def test_process_refuses_to_run_twice_at_once_and_keeps_the_queue(fresh, monkeypatch):
    monkeypatch.setattr(shadow, "SHADOW_LEARNER", True)
    shadow.capture(_rec())
    assert shadow._lock.acquire(blocking=False)
    try:
        assert shadow.process({}, RULES, T_START, fresh).get("skipped_busy") == 1
        assert len(shadow._queue) == 1                                  # nothing drained, nothing lost
    finally:
        shadow._lock.release()
    s = shadow.process({}, RULES, T_START, fresh)
    assert (s["queued"], s["opened"], len(shadow._queue)) == (1, 1, 0)


def test_queue_overflow_is_counted(fresh, monkeypatch):
    from collections import deque
    monkeypatch.setattr(shadow, "SHADOW_LEARNER", True)
    monkeypatch.setattr(shadow, "_queue", deque(maxlen=3))
    shadow.take_dropped()
    for i in range(5):
        shadow.capture(_rec(symbol=SYMBOLS[i]))
    s = shadow.process({}, RULES, T_START, fresh)
    assert (s["queued"], s["dropped"]) == (3, 2)
    assert shadow.take_dropped() == 0


def test_real_housekeeping_drains_and_inserts(fresh, monkeypatch):
    monkeypatch.setattr(bot, "_candles_cache", {})
    shadow.capture(_rec())
    asyncio.run(bot._shadow_housekeeping(timeout=10))
    assert len(shadow._queue) == 0
    assert len(_rows(fresh, "SELECT * FROM shadow_trades")) == 1
    assert json.loads(fresh.get_portfolio_stat(shadow.STATS_KEY))["opened"] == 1


def test_blocklist_is_loaded_at_boot(fresh, monkeypatch):
    fresh.set_portfolio_stat(shadow.BLOCKLIST_KEY, json.dumps([NAME]))
    monkeypatch.setattr(bot, "_evidence_blocklist", set())
    bot._load_evidence_blocklist()
    assert bot._evidence_blocklist == {NAME}


# ── learning statistics and the fixed evaluation dates ───────────────────────

def _closed_rows(db, name, per_day, days, start=datetime(2026, 10, 21, tzinfo=timezone.utc),
                 r=0.2, sym_cycle=None, spread=0.5):
    rng = np.random.default_rng(7)
    syms = sym_cycle or SYMBOLS[:10]
    with db.get_db() as conn:
        for d in range(days):
            for k in range(per_day):
                ts = (start + timedelta(days=d, minutes=k)).isoformat()
                conn.execute(
                    "INSERT INTO shadow_trades (opened_ts, opened_ms, symbol, strategy, entry, stop0, tp0, stop, "
                    "tp, last_ms, status, closed_ts, r_net) VALUES (?,?,?,?,100,103,90,103,90,0,'closed',?,?)",
                    (ts, 0, syms[k % len(syms)], name, ts, float(r + rng.normal(0, spread))))


def test_name_stats_day_clustered():
    rows = [{"symbol": "A", "opened_ts": f"2026-10-{21 + d}T00:00", "r_net": v}
            for d, v in enumerate([1.0, 0.0, 2.0, 1.0])]
    rows += [{"symbol": "B", "opened_ts": "2026-10-24T05:00", "r_net": 3.0}]   # 2 trades on one day
    s = shadow.name_stats(rows)
    assert s["n"] == 5 and s["days"] == 4 and s["mean_r"] == pytest.approx(1.4)  # per-trade mean
    r = [1.0, 0.0, 2.0, 1.0, 3.0]
    d = ["21", "22", "23", "24", "24"]
    assert s["t"] == pytest.approx(shadow.cluster_t(r, d))
    assert s["half1"] == 0.5 and s["half2"] == pytest.approx(2.0)             # trade means per half
    assert s["max_symbol_share"] == pytest.approx(4.0 / 7.0)


def test_cluster_t_matches_the_research_formula():
    import pandas  # noqa: F401  (research.smc needs it)
    from research.smc import cluster_t as smc_cluster_t
    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(5, 400))
        r = rng.normal(0.05, 1.0, n)
        day = rng.integers(0, max(2, n // 7), n)
        assert shadow.cluster_t(list(r), list(day)) == pytest.approx(smc_cluster_t(r, day), rel=1e-9)
    assert math.isnan(shadow.cluster_t([1.0], ["a"])) and math.isnan(shadow.cluster_t([1.0, 2.0], ["a", "a"]))


def test_bonferroni_bar_from_the_frozen_name_count():
    assert shadow.bonferroni_t(1) == 3.6                                  # floor
    assert shadow.bonferroni_t(150) == pytest.approx(3.6)                 # z(1.67e-4) = 3.59 -> floor
    assert shadow.bonferroni_t(400) == pytest.approx(3.836, abs=1e-3)     # z(0.05 / 800)


@pytest.mark.parametrize("field, value, ok", [
    ("n", 299, False), ("days", 19, False), ("mean_r", 0.099, False), ("t", 3.59, False),
    ("half1", -0.01, False), ("half2", 0.0, False), ("max_symbol_share", 0.26, False),
    (None, None, True)])
def test_candidate_rule_thresholds(field, value, ok):
    s = {"n": 300, "days": 20, "mean_r": 0.10, "t": 3.6, "half1": 0.1, "half2": 0.1, "max_symbol_share": 0.25}
    if field:
        s[field] = value
    assert shadow.passes(s) is ok


def _journal_signal(db, name, ts):
    db.log_trade({"symbol": SYM, "action": "short", "strategy": name, "regime": "ranging", "price": 100.0,
                  "exec_price": 100.0, "timestamp": ts, "side": "short", "status": "skipped",
                  "shares": 0.0, "trade_value": 0.0, "reason": "no_validated_edge"})


def test_evaluation_only_at_fixed_dates_and_only_for_names_frozen_at_t0(fresh, monkeypatch):
    t0 = datetime(2026, 10, 21, tzinfo=timezone.utc)
    fresh.set_portfolio_stat(shadow.T0_KEY, t0.isoformat())
    _journal_signal(fresh, NAME, (t0 - timedelta(days=3)).isoformat())          # frozen at T0
    _journal_signal(fresh, "OLD_NAME", (t0 - timedelta(days=20)).isoformat())   # outside the lookback
    _closed_rows(fresh, NAME, per_day=10, days=50, r=0.6, spread=0.3)
    _closed_rows(fresh, "LATE_NAME", per_day=10, days=50, r=0.6, spread=0.3)    # first seen after T0
    _closed_rows(fresh, "OLD_NAME", per_day=10, days=50, r=0.6, spread=0.3)

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return t0 + timedelta(days=60)
    monkeypatch.setattr(shadow, "datetime", _Now)
    with fresh.get_db() as conn:
        assert shadow.eligible_names(conn, t0) == {NAME}
        assert shadow.evaluate(conn, t0 + timedelta(days=30)) is None             # not an evaluation date
        assert shadow.evaluate(conn, t0 + timedelta(days=112)) is None            # in the future
        e = shadow.evaluate(conn, t0 + timedelta(days=56))
        assert e["candidates"] == [NAME]
        assert e["n_names"] == 1 and e["t_bar"] == 3.6
        assert shadow.passes(e["stats"]["LATE_NAME"])                             # passes, still excluded
        s = shadow.summary(conn, now=t0 + timedelta(days=60))
    assert s["next_eval"] == (t0 + timedelta(days=112)).isoformat()
    assert [x["at"] for x in s["evaluations"]] == [(t0 + timedelta(days=56)).isoformat()]
    assert {x["name"] for x in s["running"]} == {NAME, "LATE_NAME", "OLD_NAME"}


def test_noise_names_do_not_become_candidates(fresh, monkeypatch):
    t0 = datetime(2026, 10, 21, tzinfo=timezone.utc)
    fresh.set_portfolio_stat(shadow.T0_KEY, t0.isoformat())
    names = [f"N{i}" for i in range(40)]
    for n in names:
        _journal_signal(fresh, n, (t0 - timedelta(days=1)).isoformat())
        _closed_rows(fresh, n, per_day=8, days=56, r=0.0, spread=1.0)

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return t0 + timedelta(days=57)
    monkeypatch.setattr(shadow, "datetime", _Now)
    with fresh.get_db() as conn:
        assert shadow.evaluate(conn, t0 + timedelta(days=56))["candidates"] == []


def test_summary_before_t0_and_counts(fresh):
    with fresh.get_db() as conn:
        s = shadow.summary(conn)
    assert s["t0"] is None and s["open"] == 0 and s["closed"] == 0 and s["running"] == []


# ── visibility: insights payload, dashboard card, CLI report ─────────────────

def test_insights_snapshot_carries_the_shadow_summary(fresh, monkeypatch):
    import insights
    s = insights.snapshot()["shadow"]
    assert s["enabled"] is True and s["closed"] == 0 and s["t0"] is None
    monkeypatch.setattr(shadow, "summary", lambda conn: (_ for _ in ()).throw(RuntimeError("old db")))
    assert insights.snapshot()["shadow"] is None


def test_dashboard_has_the_learning_card():
    import dashboard
    html = dashboard.DASHBOARD_HTML
    assert 'id="learning-grid"' in html and "function renderLearning" in html
    assert "renderLearning(d.shadow)" in html


def test_shadow_report_script_runs(fresh, capsys):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "shadow_report.py"
    spec = importlib.util.spec_from_file_location("shadow_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    assert "no shadow trade yet" in capsys.readouterr().out
    start = (datetime.now(timezone.utc) - timedelta(days=10)).replace(hour=0, minute=0, second=0, microsecond=0)
    fresh.set_portfolio_stat(shadow.T0_KEY, start.isoformat())
    _closed_rows(fresh, NAME, per_day=5, days=8, start=start)
    mod.main()
    out = capsys.readouterr().out
    assert "40 closed" in out and NAME in out and "NOT evidence" in out
