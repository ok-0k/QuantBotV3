"""
Exit paths: what the trade journal records on every exit (on by default),
and the V3_FIXES exit behaviour (on by default since decay_v2 ended; these
tests pin it explicitly either way): stops/targets only ever tighten, break-even and trailing stops survive
time decay, tick-path order survival -> stop -> take-profit, a stop moved
within the minute triggers on the mark only, and exit rewards credit the
regime the position was entered in.

Each V3 behaviour is also asserted with the flag off where the legacy result
differs, so the flag demonstrably gates it.
"""

import asyncio
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import bot
from config import (DECAY_HALFLIFE_CANDLES, DECAY_SL_TIGHTEN_STRENGTH, SLIPPAGE_PCT,
                    STARTING_CASH, SYMBOLS)
from conftest import make_candles

ZERO_STATE = np.zeros(13, dtype=np.float32)
A_SYM = SYMBOLS[0]        # experiment arm A (legacy decay)
B_SYM = SYMBOLS[1]        # experiment arm B (designed decay)
ENTRY, SL0, TP0 = 100.0, 100.55, 97.4


@pytest.fixture()
def fresh(clean_db, monkeypatch):
    monkeypatch.setattr(bot, "V3_FIXES", False)              # legacy path, whatever the default or env says
    bot.brain._edge_profiles.clear()
    bot.brain.ml_probs.clear()
    bot.brain.circuit_open = False
    bot.brain._peak_equity = 0.0
    bot._tick_minute_stop.clear()
    return clean_db


@pytest.fixture()
def v3(fresh, monkeypatch):                                  # after fresh, so it wins
    monkeypatch.setattr(bot, "V3_FIXES", True)


def _ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _place_short(db, sym, arm="B", age_min=1, count=0, regime="ranging", stop=SL0, tp=TP0):
    """A 1-share short at $100 (survival floor -$15 is a 15% move away)."""
    feats = {"experiment": "decay_v2", "arm": arm, "initial_stop": SL0,
             "initial_tp": TP0, "regime": regime}
    db.open_short(sym, 1.0, ENTRY, "QF", stop, tp, 20.0, entry_features=feats)
    with db.get_db() as conn:
        conn.execute("UPDATE positions SET opened_ts=?, candle_count=? WHERE symbol=?",
                     (_ago(age_min), count, sym))
    return db.get_short_position(sym)


def _history(sym):
    """Quiet 1m history at $100 (ATR ~0.06) in the cache; returns (candles, atr)."""
    candles = make_candles(n=120, start_price=ENTRY, drift=0.0, amplitude=0.04)
    bot._candles_cache[sym] = candles
    return candles, bot._wilder_atr(candles)


def _tick(sym, candles, high, low, close, minute=1):
    candle = {"open": close, "high": high, "low": low, "close": close,
              "volume": 1.0, "time": candles[-1]["time"] + minute * 60_000}

    async def run():
        await bot._tick_exit_check(sym, candle, asyncio.get_running_loop())
    asyncio.run(run())


def _stop(db, sym):
    return db.get_short_position(sym)["stop_price"]


def _covers(db):
    with db.get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE action='cover' AND status='filled' ORDER BY id")]


def _be_candle(atr):
    """Deep, confirmed favourable move: arms break-even, touches nothing else."""
    return dict(high=ENTRY - 0.5 * atr, low=ENTRY - 5.0 * atr, close=ENTRY - 2.0 * atr)


def _designed_target_sl(age_min):
    lam = 1 - math.exp(-age_min / DECAY_HALFLIFE_CANDLES)
    return SL0 + lam * DECAY_SL_TIGHTEN_STRENGTH * (ENTRY - SL0)


def _spy_stop_writes(monkeypatch):
    """Records (new_stop, row_written) for every update_stop_price call from bot."""
    writes = []
    real = bot.update_stop_price

    def spy(sym, new_stop, only_tighten=False):
        ok = real(sym, new_stop, only_tighten=only_tighten)
        writes.append((new_stop, ok))
        return ok
    monkeypatch.setattr(bot, "update_stop_price", spy)
    return writes


# ── db: optional directional guard ───────────────────────────────────────────

def test_update_stop_price_only_tighten_guard(fresh):
    _place_short(fresh, B_SYM)
    assert fresh.update_stop_price(B_SYM, 100.7, only_tighten=True) is False   # loosen: refused
    assert _stop(fresh, B_SYM) == SL0
    assert fresh.update_stop_price(B_SYM, 100.2, only_tighten=True) is True    # tighten: written
    assert _stop(fresh, B_SYM) == 100.2
    assert fresh.update_stop_price(B_SYM, 100.9) is True                       # default: unguarded
    assert _stop(fresh, B_SYM) == 100.9


@pytest.mark.parametrize("side", ["long", None])
def test_update_stop_price_guard_treats_non_short_as_long(fresh, side):
    _place_short(fresh, B_SYM)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET side=?, stop_price=95.0, tp_price=110.0 WHERE symbol=?",
                     (side, B_SYM))
    get = lambda col: fresh.get_position(B_SYM)[col]
    assert fresh.update_stop_price(B_SYM, 94.0, only_tighten=True) is False
    assert fresh.update_stop_price(B_SYM, 96.0, only_tighten=True) is True and get("stop_price") == 96.0
    assert fresh.update_tp_price(B_SYM, 111.0, only_tighten=True) is False
    assert fresh.update_tp_price(B_SYM, 108.0, only_tighten=True) is True and get("tp_price") == 108.0


def test_update_tp_price_only_tightens_toward_entry_and_fills_empty(fresh):
    _place_short(fresh, B_SYM)
    assert fresh.update_tp_price(B_SYM, 97.0, only_tighten=True) is False      # further away
    assert fresh.update_tp_price(B_SYM, 98.0, only_tighten=True) is True       # toward entry
    assert fresh.get_short_position(B_SYM)["tp_price"] == 98.0
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET stop_price=NULL, tp_price=0 WHERE symbol=?", (B_SYM,))
    assert fresh.update_stop_price(B_SYM, 101.0, only_tighten=True) is True    # none stored
    assert fresh.update_tp_price(B_SYM, 96.0, only_tighten=True) is True


# ── Exit journal (on by default): position regime, mark and stored levels ───

def _spy_reward(monkeypatch):
    calls = []
    real = bot.brain.reward

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)
    monkeypatch.setattr(bot.brain, "reward", spy)
    return calls


def _tp_exit(fresh, monkeypatch):
    """Ranging-entered short; the brain last evaluated a trend_down symbol."""
    _place_short(fresh, B_SYM, regime="ranging")
    candles, _ = _history(B_SYM)
    monkeypatch.setattr(bot.brain, "current_regime", "trend_down")
    calls = _spy_reward(monkeypatch)
    _tick(B_SYM, candles, high=ENTRY, low=TP0 - 0.1, close=TP0 - 0.05)
    return _covers(fresh)[0], calls


def test_exit_row_journals_position_regime_reward_unchanged_by_default(fresh, monkeypatch):
    row, calls = _tp_exit(fresh, monkeypatch)
    assert row["exit_reason"] == "TAKE_PROFIT"
    assert row["regime"] == "ranging"                       # the position's own regime
    assert calls[-1]["regime"] == "trend_down"              # V3 off: legacy global value
    assert calls[-1]["next_state"][9] == 1.0                # SAC exit state: global bits


def test_exit_without_entry_regime_keeps_global_regime(fresh, monkeypatch):
    _place_short(fresh, B_SYM)
    with fresh.get_db() as conn:
        conn.execute("UPDATE positions SET entry_features=NULL WHERE symbol=?", (B_SYM,))
    candles, _ = _history(B_SYM)
    monkeypatch.setattr(bot.brain, "current_regime", "trend_up")
    _tick(B_SYM, candles, high=ENTRY, low=TP0 - 0.1, close=TP0 - 0.05)
    assert _covers(fresh)[0]["regime"] == "trend_up"


def test_tick_exit_detail_records_mark_and_stored_levels(fresh):
    _place_short(fresh, B_SYM)
    candles, atr = _history(B_SYM)
    mark, low = TP0 - 0.05, TP0 - 0.1
    _tick(B_SYM, candles, high=ENTRY, low=low, close=mark)
    detail = _covers(fresh)[0]["exit_detail"]
    head, tail = detail.split(" | ")
    assert head.startswith("take_profit_tick_short (low=")
    # the stop as stored at exit: the trail had just moved it to low + ATR
    assert tail == f"mark={mark:.10g} stop={low + atr:.10g} tp={TP0:.10g}"


def _one_monitor_cycle(monkeypatch, candles, price, sym=B_SYM):
    """One exit_monitor cycle with `sym`'s mark at `price` (test_exit_monitor pattern)."""
    last = dict(candles[-1], close=price, high=max(price, candles[-1]["high"]),
                low=min(price, candles[-1]["low"]))
    bot._candles_cache[sym] = candles[:-1] + [last]

    async def run():
        ev = asyncio.Event()
        monkeypatch.setattr(bot, "_shutdown_event", ev)
        monkeypatch.setattr(bot, "CHECK_EVERY_SECS", 0.01)
        monkeypatch.setattr(bot, "record_equity", lambda eq: ev.set())
        await asyncio.wait_for(bot.exit_monitor(asyncio.get_running_loop()), 10)
    asyncio.run(run())


def test_monitor_exit_journals_position_regime_and_detail(fresh, monkeypatch):
    stop = ENTRY * 1.004
    _place_short(fresh, B_SYM, regime="ranging", stop=stop)
    candles, _ = _history(B_SYM)
    monkeypatch.setattr(bot.brain, "current_regime", "trend_up")
    price = stop * 1.001
    _one_monitor_cycle(monkeypatch, candles, price)
    row = _covers(fresh)[0]
    assert row["regime"] == "ranging"
    assert row["exit_reason"] == "TIGHTENED_STOP"
    assert row["exit_detail"].startswith("stop_loss_short (")
    assert row["exit_detail"].endswith(f" | mark={price:.10g} stop={stop:.10g} tp={TP0:.10g}")


@pytest.mark.parametrize("reason,stop,short,initial", [
    ("take_profit_tick_short (low=$1 <= tp=$2)", 101.0, True, 101.0),
    ("stop_loss_tick_short (high=$1 >= stop=$2)", 100.0, True, 101.0),
    ("stop_loss_tick_short (mark=$1 >= stop=$2)", 99.0, True, 101.0),
    ("stop_loss_short ($1 >= $2)", 100.5, True, 101.0),
    ("stop_loss_tick (low=$1 <= stop=$2)", 99.0, False, 99.0),
    ("hard_stop_survival_tick (pnl=$-20)", 101.0, True, 101.0),
    ("time_hold_exit (4.1h open, in_red=$-1)", 101.0, True, 101.0),
    ("max_hold_time (900 candles)", 101.0, True, 101.0),
])
def test_exit_detail_suffix_leaves_classification_unchanged(fresh, reason, stop, short, initial):
    _place_short(fresh, B_SYM)
    detail = bot._exit_detail(reason, B_SYM, 100.25)
    assert detail.startswith(reason + " | mark=100.25 stop=")
    assert (bot._classify_exit_reason(detail, 100.0, stop, short, initial)
            == bot._classify_exit_reason(reason, 100.0, stop, short, initial))


# ── V3: break-even / trailing stops survive time decay ───────────────────────

@pytest.mark.parametrize("sym,arm,age,count", [(B_SYM, "B", 60, 0), (A_SYM, "A", 60, 60)])
def test_v3_break_even_persists_across_ticks(fresh, v3, monkeypatch, sym, arm, age, count):
    stale = _place_short(fresh, sym, arm=arm, age_min=age, count=count)
    candles, atr = _history(sym)
    writes = _spy_stop_writes(monkeypatch)
    _tick(sym, candles, **_be_candle(atr))
    assert _stop(fresh, sym) == ENTRY                       # decay did not overwrite it
    _tick(sym, candles, **_be_candle(atr))
    _tick(sym, candles, **_be_candle(atr), minute=2)
    assert writes == [(ENTRY, True)]                        # armed once, never re-armed
    assert _stop(fresh, sym) == ENTRY
    bot._apply_ml_time_decay(sym, stale, candles)           # stale snapshot: guard refuses
    assert _stop(fresh, sym) == ENTRY
    assert fresh.get_short_position(sym) is not None


def test_v3_off_legacy_decay_overwrites_break_even(fresh, monkeypatch):
    """Known bug, kept while V3_FIXES is off: decay is fed the row read before
    break-even moved, and writes its own loss-side level straight over it."""
    assert bot.V3_FIXES is False
    _place_short(fresh, B_SYM, arm="B", age_min=60)
    candles, atr = _history(B_SYM)
    writes = _spy_stop_writes(monkeypatch)
    _tick(B_SYM, candles, **_be_candle(atr))
    _tick(B_SYM, candles, **_be_candle(atr))
    assert _stop(fresh, B_SYM) == pytest.approx(_designed_target_sl(60), abs=1e-3)
    assert [w for w, _ in writes].count(ENTRY) == 2         # break-even re-armed every tick


@pytest.mark.parametrize("v3_on", [True, False])
def test_trailing_stop_persists_only_with_v3(fresh, monkeypatch, v3_on):
    monkeypatch.setattr(bot, "V3_FIXES", v3_on)
    tp = ENTRY * (1 - 0.0021)                               # fee-gated target, so no TP exit
    _place_short(fresh, B_SYM, arm="B", age_min=60, tp=tp)
    candles, atr = _history(B_SYM)
    low = tp - 2.0 * atr
    trail = low + atr
    tick = dict(high=ENTRY - 0.5 * atr, low=low, close=trail - 0.1 * atr)
    _tick(B_SYM, candles, **tick)
    _tick(B_SYM, candles, **tick)
    assert fresh.get_short_position(B_SYM) is not None
    if v3_on:
        assert _stop(fresh, B_SYM) == pytest.approx(trail, rel=1e-12)
    else:
        assert _stop(fresh, B_SYM) == pytest.approx(_designed_target_sl(60), abs=1e-3)


def test_v3_monitor_rereads_position_before_decay(fresh, v3, monkeypatch):
    """A break-even stored while exit_monitor awaited (cycle snapshot now
    stale) must survive the monitor's decay."""
    _place_short(fresh, B_SYM, arm="B", age_min=60)
    candles, _ = _history(B_SYM)

    async def moves_stop_meanwhile(*_a, **_k):
        fresh.update_stop_price(B_SYM, ENTRY)               # tick path arms break-even
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", moves_stop_meanwhile)
    _one_monitor_cycle(monkeypatch, candles, ENTRY * 0.999)
    assert _stop(fresh, B_SYM) == ENTRY


# ── V3: tick-path exit order ─────────────────────────────────────────────────

@pytest.mark.parametrize("v3_on", [True, False])
def test_fee_gated_take_profit_falls_through_to_the_stop(fresh, monkeypatch, v3_on):
    monkeypatch.setattr(bot, "V3_FIXES", v3_on)
    tp = ENTRY * (1 - 0.0021)                               # touch is fee-gated
    stop = ENTRY * 1.003
    _place_short(fresh, B_SYM, stop=stop, tp=tp)
    candles, _ = _history(B_SYM)
    mark = ENTRY * 1.0035
    _tick(B_SYM, candles, high=ENTRY * 1.004, low=tp, close=mark)
    if v3_on:
        row = _covers(fresh)[0]
        assert row["exit_reason"] == "TIGHTENED_STOP"
        assert row["exec_price"] == pytest.approx(mark * (1 + SLIPPAGE_PCT), rel=1e-9)
    else:                                                   # legacy: gated TP hides the stop
        assert fresh.get_short_position(B_SYM) is not None and _covers(fresh) == []


@pytest.mark.parametrize("v3_on", [True, False])
def test_survival_exits_at_mark_even_when_tp_touched(fresh, monkeypatch, v3_on):
    monkeypatch.setattr(bot, "V3_FIXES", v3_on)
    _place_short(fresh, B_SYM)
    candles, _ = _history(B_SYM)
    mark = ENTRY * 1.16                                     # open loss -$16 < -$15 floor
    _tick(B_SYM, candles, high=ENTRY * 1.17, low=TP0 - 0.4, close=mark)
    row = _covers(fresh)[0]
    if v3_on:
        assert row["exit_reason"] == "HARD_STOP"
        assert row["exec_price"] == pytest.approx(mark * (1 + SLIPPAGE_PCT), rel=1e-9)
        assert row["exit_detail"].startswith("hard_stop_survival_tick")
    else:                                                   # legacy: booked at the TP price
        assert row["exit_reason"] == "TAKE_PROFIT"


def test_v3_stop_and_take_profit_in_one_candle_takes_the_stop(fresh, v3):
    tp = ENTRY * 0.99                                       # clears fees: would book
    _place_short(fresh, B_SYM, stop=ENTRY * 1.003, tp=tp)
    candles, _ = _history(B_SYM)
    _tick(B_SYM, candles, high=ENTRY * 1.004, low=tp * 0.999, close=ENTRY)
    assert _covers(fresh)[0]["exit_reason"] == "TIGHTENED_STOP"


def test_v3_mid_candle_break_even_triggers_on_mark_only(fresh, v3):
    _place_short(fresh, B_SYM)
    candles, atr = _history(B_SYM)
    hi = ENTRY + 1.0 * atr                                  # printed before break-even existed
    _tick(B_SYM, candles, high=hi, low=ENTRY - 5.0 * atr, close=ENTRY - 2.0 * atr)
    assert _stop(fresh, B_SYM) == ENTRY and _covers(fresh) == []
    # Later ticks in the same minute: the stored high still counts only
    # against the stop in place when the minute began.
    _tick(B_SYM, candles, high=hi, low=ENTRY - 5.0 * atr, close=ENTRY - 1.0 * atr)
    assert _covers(fresh) == []
    # The mark crossing the break-even stop does trigger it, at the mark.
    mark = ENTRY + 0.2 * atr
    _tick(B_SYM, candles, high=hi, low=ENTRY - 5.0 * atr, close=mark)
    row = _covers(fresh)[0]
    assert row["exit_reason"] == "BREAKEVEN_STOP"
    assert row["exec_price"] == pytest.approx(mark * (1 + SLIPPAGE_PCT), rel=1e-9)


def test_v3_persisted_break_even_uses_the_next_minutes_high(fresh, v3):
    _place_short(fresh, B_SYM)
    candles, atr = _history(B_SYM)
    _tick(B_SYM, candles, high=ENTRY + 1.0 * atr, low=ENTRY - 5.0 * atr, close=ENTRY - 2.0 * atr)
    _tick(B_SYM, candles, high=ENTRY + 0.5 * atr, low=ENTRY - 1.0 * atr,
          close=ENTRY - 0.5 * atr, minute=2)
    row = _covers(fresh)[0]                                 # wick through BE: stop fills at BE
    assert row["exit_reason"] == "BREAKEVEN_STOP"
    assert row["exec_price"] == pytest.approx(ENTRY * (1 + SLIPPAGE_PCT), rel=1e-9)


def test_v3_off_mid_candle_break_even_hit_by_earlier_high(fresh):
    assert bot.V3_FIXES is False
    _place_short(fresh, B_SYM)
    candles, atr = _history(B_SYM)
    _tick(B_SYM, candles, high=ENTRY + 1.0 * atr, low=ENTRY - 5.0 * atr, close=ENTRY - 2.0 * atr)
    assert _covers(fresh)[0]["exit_reason"] == "BREAKEVEN_STOP"


# ── V3: stop-integrity property over a tick sequence, per arm (TI1) ──────────

STOP_CODES = {"STOP_LOSS", "TIGHTENED_STOP", "BREAKEVEN_STOP", "TRAILING_STOP"}


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("tp", [TP0, ENTRY * (1 - 0.0021)])     # far / fee-gated (trail acts)
@pytest.mark.parametrize("sym,arm,count", [(A_SYM, "A", 60), (B_SYM, "B", 0)])
def test_v3_stop_integrity_over_tick_sequence(fresh, v3, monkeypatch, sym, arm, count, tp, seed):
    """~10 minutes of 2 s ticks (cumulative per-minute high/low, like Binance
    klines) on a 60-minute-old short, with an exit_monitor cycle every 15
    ticks. Invariants: no stop level written is ever undone (within a call
    or across calls); once at or past entry it stays there; an open
    position's mark is below its stop; a stop exit fired on a level stored
    since the minute began, or on the mark."""
    import random

    async def no_margin(*_a, **_k):
        return {}
    monkeypatch.setattr(bot, "_refresh_margin_health", no_margin)
    _place_short(fresh, sym, arm=arm, age_min=60, count=count, tp=tp)
    candles, _ = _history(sym)
    t0 = candles[-1]["time"]
    rng, px = random.Random(seed), ENTRY
    writes = _spy_stop_writes(monkeypatch)
    minute = hi = lo = s_minute = None
    for i in range(300):
        pos = fresh.get_short_position(sym)
        if pos is None:
            break
        s_before = pos["stop_price"]
        px *= 1 + rng.gauss(-0.0002 if i < 75 else 0.0002, 0.0001)   # down, then back up
        m = i // 30
        if m != minute:
            minute, hi, lo, s_minute = m, px, px, s_before
        hi, lo = max(hi, px), min(lo, px)
        if i % 15 == 14:
            _one_monitor_cycle(monkeypatch, candles, px, sym)
            high_seen = px
        else:
            candle = {"open": px, "high": hi, "low": lo, "close": px, "volume": 1.0,
                      "time": t0 + (m + 1) * 60_000}
            bot._candles_cache[sym] = candles

            async def run():
                await bot._tick_exit_check(sym, candle, asyncio.get_running_loop())
            asyncio.run(run())
            high_seen = hi
        after = fresh.get_short_position(sym)
        if after is not None:
            s = after["stop_price"]
            assert s <= s_before + 1e-12, f"stop loosened at tick {i}"
            assert s <= min([SL0] + [w for w, ok in writes if ok]) + 1e-12, \
                f"a stop written earlier was undone at tick {i}"
            if s_before <= ENTRY:
                assert s <= ENTRY, f"break-even/trail undone at tick {i}"
            assert px < s, f"open with mark {px} >= stop {s} at tick {i}"
        else:
            row = _covers(fresh)[-1]
            if row["exit_reason"] in STOP_CODES:
                assert px >= s_before or high_seen >= s_minute, f"phantom stop exit at tick {i}"


# ── V3: exit reward and SAC exit state use the position regime ──────────────

def test_v3_reward_and_sac_exit_state_use_position_regime(fresh, v3, monkeypatch):
    row, calls = _tp_exit(fresh, monkeypatch)
    assert row["regime"] == "ranging"
    assert calls[-1]["regime"] == "ranging"
    assert (calls[-1]["next_state"][9], calls[-1]["next_state"][10]) == (0.0, 0.0)
    assert "short:ranging" in bot.brain._edge_profiles
    assert "short:trend_down" not in bot.brain._edge_profiles
