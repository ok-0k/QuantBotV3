"""
SMC playbook study (research/PREREG_SMC.md): detection, timing, no-lookahead,
fill model, costs, limits and decision rules, on SYNTHETIC data only. No real
market data is read here and no real-data result is computed.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research import smc as M

TICK = 0.01


# ── helpers ──────────────────────────────────────────────────────────────────

def bars_from(o, h, l, c, t0=0, tf=5):
    n = len(c)
    t = t0 + tf * M.MIN_MS * np.arange(n, dtype=np.int64)
    a = lambda x: np.asarray(x, dtype=float)
    return M.Bars(t, a(o), a(h), a(l), a(c), tf, np.full(n, 1e5))


def flat_day(D, price=99.70, start_min=9 * 60, end_min=13 * 60):
    """5m bars on NY day D, every bar o=h=l=c=price."""
    t0 = M.ny_ms(D, start_min)
    n = (M.ny_ms(D, end_min) - t0) // M.BASE_MS
    p = np.full(n, price)
    return bars_from(p.copy(), p.copy(), p.copy(), p.copy(), t0)


def at(b, D, minute):
    return (M.ny_ms(D, minute) - int(b.t[0])) // M.BASE_MS


def set_bar(b, i, o=None, h=None, l=None, c=None):
    for arr, v in ((b.o, o), (b.h, h), (b.l, l), (b.c, c)):
        if v is not None:
            arr[i] = v


D0 = M.day_num(date(2024, 3, 4))          # a Monday (EST)


def long_cand(D=D0, place_min=9 * 60 + 40, L=99.50, S=99.00, T1=100.00, T2=101.00, bot=99.30, side=1):
    return M.Candidate("XUSDT", 5, side, D, M.ny_ms(D, place_min), 0, L, S, T1, T2, bot, TICK,
                       9, "A+", True, True, False, ("PD",), 0, 0)


def random_walk(start, days, seed, base=100.0, sigma=0.0025):
    """Fat-tailed (Student-t, 3 dof) 5m walk, more volatile 09:30-11:00 NY."""
    rng = np.random.default_rng(seed)
    n = days * 288
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t = t0 + M.BASE_MS * np.arange(n, dtype=np.int64)
    _, minute = M.ny_day_minute(t)
    vol = sigma * np.where((minute >= 570) & (minute < 660), 2.0, 1.0)
    c = np.round(base * np.exp(np.cumsum(rng.standard_t(3, n) * 0.6 * vol)), 2)
    o = np.r_[base, c[:-1]]
    up = np.round(np.abs(rng.normal(0, 0.5, n)) * vol * c, 2)
    dn = np.round(np.abs(rng.normal(0, 0.5, n)) * vol * c, 2)
    h = np.round(np.maximum(o, c) + up, 2)
    l = np.round(np.minimum(o, c) - dn, 2)
    return bars_from(o, h, l, c, t0)


def reflect(b, mid):
    """Mirror image about `mid` (stays positive): high <-> low."""
    return M.Bars(b.t, np.round(2 * mid - b.o, 2), np.round(2 * mid - b.l, 2),
                  np.round(2 * mid - b.h, 2), np.round(2 * mid - b.c, 2), b.tf, b.qv)


# ── grid, calendar, time ─────────────────────────────────────────────────────

def test_grid_is_the_8_preregistered_configs():
    g = M.grid()
    assert len(g) == 8 and len({c.name for c in g}) == 8
    assert {c.tf for c in g} == {5, 15} and {c.mgmt for c in g} == {"pbe", "tp2"}
    assert {c.bias for c in g} == {True, False}
    assert M.DECISION_SIDES == ("short",)                    # "long" and "both" are report-only
    assert M.PASS["oos_t"] == 2.50 and M.PASS["day_share"] == 0.25


def test_trading_days_are_nyse_weekdays():
    assert all(d.weekday() < 5 for d in M.NYSE_HOLIDAYS)
    days = M.trading_days()
    ds = [M.EPOCH + pd.Timedelta(days=x) for x in days]
    assert all(d.weekday() < 5 for d in ds)
    assert not set(ds) & M.NYSE_HOLIDAYS
    assert min(ds) == date(2023, 1, 3) and max(ds) == date(2026, 10, 6)


def test_ny_time_is_dst_aware():
    fri, mon = M.day_num(date(2024, 3, 8)), M.day_num(date(2024, 3, 11))   # DST starts 2024-03-10
    assert pd.Timestamp(M.ny_ms(fri, 570), unit="ms", tz="UTC").hour == 14   # 09:30 EST = 14:30 UTC
    assert pd.Timestamp(M.ny_ms(mon, 570), unit="ms", tz="UTC").hour == 13   # 09:30 EDT = 13:30 UTC
    t = np.array([M.ny_ms(fri, 570), M.ny_ms(mon, 570), M.ny_ms(mon, 0)], dtype=np.int64)
    day, minute = M.ny_day_minute(t)
    assert list(day) == [fri, mon, mon] and list(minute) == [570, 570, 0]


def test_daily_and_4h_bars_follow_ny_wall_clock_across_dst():
    b = random_walk("2024-03-01", 20, seed=1)
    day, minute = M.ny_day_minute(b.t)
    d1 = M.htf_bars(b, day, minute, "1d")
    sun = M.day_num(date(2024, 3, 10))                   # 23-hour NY day
    k = int(np.flatnonzero(d1.key == sun)[0])
    assert d1.t_close[k] - d1.t_close[k - 1] == 23 * M.HOUR_MS
    assert np.isfinite(d1.h[k])                          # complete although only 276 bars
    assert not np.isfinite(d1.h[0]) and not np.isfinite(d1.h[-1])   # grid-edge blocks invalid
    h4 = M.htf_bars(b, day, minute, "4h")
    closes = pd.to_datetime(h4.t_close[1:30], unit="ms", utc=True).tz_convert(M.NY)
    assert set(closes.hour) <= {0, 4, 8, 12, 16, 20}


def test_missing_5m_bar_removes_the_15m_bar_only():
    b = random_walk("2024-01-01", 2, seed=2)
    b.c[4] = np.nan
    b15 = M.resample(b, 15)
    assert np.isnan(b15.c[1]) and np.isfinite(b15.c[0]) and np.isfinite(b15.c[2])
    assert b15.h[0] == np.max(b.h[0:3]) and b15.o[0] == b.o[0] and b15.c[0] == b.c[2]


# ── swings, structure, ATR ──────────────────────────────────────────────────

def test_swing_needs_strictly_higher_high_and_present_neighbours():
    h = np.array([1, 2, 3, 5, 3, 2, 1, 2, 5, 5, 2, 1.0])
    sh, sl = M.swings(h, h - 0.5, 2)
    assert sh[3] and not sh[8] and not sh[9]            # equal highs are not swings
    h2 = h.copy()
    h2[5] = np.nan
    assert not M.swings(h2, h2 - 0.5, 2)[0][3]          # a missing neighbour kills the swing


def test_swing_is_usable_only_after_the_confirmation_candle_closes():
    # swing high at i=3 (h=10) is confirmed at the close of candle 5 (N=2).
    h = [5, 6, 7, 10, 8, 7, 9, 12, 13]
    l = [4, 5, 6, 8, 6, 5, 6, 8, 11]
    c = [4.5, 5.5, 6.5, 9, 7, 6, 8.5, 11, 12]
    o = [4.2, 5.2, 6.2, 7, 8, 7, 6, 9, 11.5]
    st = M.structure(bars_from(o, h, l, c), 2)
    assert list(st.brk[:7]) == [0] * 7                  # 9 > 8.5 is no break; ref only exists from candle 6
    assert st.brk[7] == 1 and st.ref[7] == 10           # first CLOSE above 10
    assert st.brk[8] == 0                               # ref consumed, no newer swing high yet


def test_wick_beyond_a_swing_is_not_a_break_and_mss_vs_bos():
    h = [5, 6, 7, 10, 8, 7, 11, 9, 12]
    l = [4, 5, 6, 8, 6, 5, 6, 7, 9]
    c = [4.5, 5.5, 6.5, 9, 7, 6, 9.9, 8, 11]
    o = [4.2, 5.2, 6.2, 7, 8, 7, 6, 9.5, 8]
    st = M.structure(bars_from(o, h, l, c), 2)
    assert st.brk[6] == 0                                # wick to 11, close 9.9 <= 10
    assert st.brk[8] == 1 and st.prev[8] == 0            # from no state: not an MSS (prev != -1)


def test_atr_is_wilder_and_atr_prev_excludes_the_candle():
    n = 30
    o = np.full(n, 100.0)
    h, l, c = o + 1, o - 1, o.copy()
    a, ap = M.atr(bars_from(o, h, l, c))
    assert np.isnan(a[12]) and a[13] == pytest.approx(2.0) and a[29] == pytest.approx(2.0)
    h2 = h.copy()
    h2[20] = 111.0                                       # TR 12 on candle 20
    a2, ap2 = M.atr(bars_from(o, h2, l, c))
    assert ap2[20] == pytest.approx(2.0) and a2[20] == pytest.approx(2.0 + 10 / 14)
    assert ap2[21] == pytest.approx(a2[20])


# ── displacement, FVG, sweep ────────────────────────────────────────────────

def _flat(n, p=100.0):
    return [np.full(n, p) for _ in range(4)]


def test_displacement_single_and_runs():
    o, h, l, c = _flat(6)
    ap = np.full(6, 1.0)
    o[5], c[5], h[5], l[5] = 100, 101.5, 101.9, 99.9         # body 1.5 ATR: single
    b = bars_from(o, h, l, c)
    assert M.disp_leg(b, ap, 5, 0) == [5]
    c[5], h[5] = 101.4, 101.4
    assert M.disp_leg(bars_from(o, h, l, c), ap, 5, 0) is None   # body 1.4 < 1.5
    o2, h2, l2, c2 = _flat(6)
    for k, (oo, cc, hh) in zip((3, 4, 5), ((100, 101, 101.2), (101, 102, 102.25), (102, 103, 103.1))):
        o2[k], c2[k], h2[k], l2[k] = oo, cc, hh, oo - 0.1
    b2 = bars_from(o2, h2, l2, c2)
    assert M.disp_leg(b2, ap, 5, 0) == [3, 4, 5]
    assert M.disp_leg(b2, ap, 5, 4) == [4, 5]                    # leg may not start before start_min
    h2[4] = 102.3                                                # upper wick 0.3 > 0.25 x body
    assert M.disp_leg(bars_from(o2, h2, l2, c2), ap, 5, 0) is None


def test_fvg_detection_min_height_and_ce():
    o, h, l, c = _flat(5)
    h[1], l[3] = 100.2, 100.6                                    # gap [100.2, 100.6], height 0.4
    o[2], c[2], h[2], l[2] = 100.1, 100.9, 100.95, 100.05
    o[3], c[3], h[3] = 100.7, 100.8, 100.9
    b = bars_from(o, h, l, c)
    f = M.leg_fvgs(b, np.full(5, 1.0), [2], 3)
    assert f == [(2, pytest.approx(100.2), pytest.approx(100.6))]
    assert 0.5 * (f[0][1] + f[0][2]) == pytest.approx(100.4)
    assert M.leg_fvgs(b, np.full(5, 2.0), [2], 3) == []         # 0.4 < 0.25 x 2.0
    assert M.leg_fvgs(b, np.full(5, 1.0), [2], 2) == []         # third candle not closed yet


def test_sweep_reclaim_windows_and_breakout():
    ap = np.full(8, 1.0)
    level = 100.0

    def mk(closes, lows=None, opens=None):
        n = len(closes)
        c = np.array(closes, float)
        o = np.array(opens if opens else [100.5] * n, float)
        l = np.array(lows if lows else [min(x, 100.5) - 0.2 for x in closes], float)
        h = np.maximum(o, c) + 0.05
        return bars_from(o, h, l, c)

    assert M.classify_sweep(mk([100.3, 100.5, 100.5]), ap, 0, level) == 0          # same-candle reclaim
    assert M.classify_sweep(mk([99.9, 100.2, 100.5]), ap, 0, level) == 1
    assert M.classify_sweep(mk([99.9, 99.95, 100.1]), ap, 0, level) == 2           # held 2, no displacement
    assert M.classify_sweep(mk([99.9, 99.9, 99.9, 100.5]), ap, 0, level) is None   # no reclaim in time
    # breakout: two closes below, the first a bearish displacement candle (body 1.6 ATR)
    b = mk([98.9, 98.95, 100.2], lows=[98.8, 98.85, 98.9], opens=[100.5, 98.9, 98.95])
    assert M.classify_sweep(b, ap, 0, level) is None
    assert M.classify_sweep(mk([100.0, 100.0, 100.0]), ap, 0, level) is None       # close AT the level


# ── ticks and order parameters ──────────────────────────────────────────────

def test_tick_inference_and_rounding_are_mirror_symmetric():
    b = random_walk("2024-01-01", 3, seed=3)
    assert set(M.infer_ticks(b).values()) == {0.01}
    b2 = M.Bars(b.t, np.round(b.o, 1), np.round(b.h, 1), np.round(b.l, 1), np.round(b.c, 1), 5, b.qv)
    assert set(M.infer_ticks(b2).values()) == {0.1}
    assert M.floor_tick(100.037, 0.01) == 100.03 and M.ceil_tick(100.031, 0.01) == 100.04
    assert M.floor_tick(100.03, 0.01) == 100.03                  # exact multiples are kept
    for x in (100.037, 99.995, 0.12345):
        assert M.floor_tick(-x, 0.01) == -M.ceil_tick(x, 0.01)


def test_stop_buffer_is_max_of_two_ticks_and_tenth_atr():
    sweep_low = 99.00
    assert M.floor_tick(sweep_low - max(2 * TICK, 0.1 * 0.05), TICK) == 98.98   # ticks bind
    assert M.floor_tick(sweep_low - max(2 * TICK, 0.1 * 1.234), TICK) == 98.87  # ATR binds, floored


# ── fill model ───────────────────────────────────────────────────────────────

def _sim(b, cand, mgmt="tp2", tfc=None):
    return M.simulate(b, b.c if tfc is None else tfc, cand, mgmt)


def gross(cand, oc):
    return M.net_r(cand, oc, M.COSTS["zero"])


def test_limit_touch_does_not_fill_and_order_expires_after_60_minutes():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.50)                          # touches the limit only
    set_bar(b, at(b, D0, 640), l=99.40)                          # opens exactly at placement + 60 min
    oc = _sim(b, long_cand())
    assert not oc.filled and oc.reason == "expired" and oc.t_end == M.ny_ms(D0, 640)


def test_limit_fills_when_traded_one_tick_through_then_time_stop():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.49)
    c = long_cand()
    oc = _sim(b, c)
    assert oc.filled and oc.t_fill == M.ny_ms(D0, 585)
    assert oc.reason == "time_stop" and oc.t_end == M.ny_ms(D0, 590 + 45)   # 45 min after the fill bar's close
    assert gross(c, oc) == pytest.approx((99.70 - 99.50) / 0.5)


def test_time_stop_disarmed_at_half_r_then_session_exit_at_noon():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.49)
    set_bar(b, at(b, D0, 600), h=99.75)                          # +0.5R reached
    oc = _sim(b, long_cand())
    assert oc.reason == "session_end" and oc.t_end == M.ny_ms(D0, 720)


def test_tp1_reached_first_cancels_even_in_the_fill_bar():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), h=100.00, l=99.40)
    oc = _sim(b, long_cand())
    assert not oc.filled and oc.reason == "tp1_first"


def test_fvg_close_through_cannot_beat_the_fill_but_cancels_when_alone():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.20, c=99.25)                  # through the limit AND closes below the FVG
    assert _sim(b, long_cand()).filled
    b2 = flat_day(D0)
    tfc = b2.c.copy()
    tfc[at(b2, D0, 585)] = 99.20                                 # entry-TF close below the FVG bottom
    oc = _sim(b2, long_cand(), tfc=tfc)
    assert not oc.filled and oc.reason == "fvg_invalid"


def test_order_cancelled_at_killzone_end():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 660), l=99.00)
    oc = _sim(b, long_cand(place_min=630))
    assert not oc.filled and oc.reason == "killzone_end" and oc.t_end == M.ny_ms(D0, 660)


def test_stop_in_the_fill_bar_and_gap_through_the_stop():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=98.99)
    c = long_cand()
    oc = _sim(b, c)
    assert oc.filled and oc.reason == "stop" and gross(c, oc) == pytest.approx(-1.0)
    b2 = flat_day(D0)
    set_bar(b2, at(b2, D0, 585), l=99.49)
    set_bar(b2, at(b2, D0, 590), o=98.90, h=98.95, l=98.80, c=98.85)   # gaps through the stop
    oc2 = _sim(b2, c)
    assert oc2.reason == "stop" and gross(c, oc2) == pytest.approx((98.90 - 99.50) / 0.5)


def test_stop_and_target_in_the_same_bar_counts_the_stop():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.49)
    set_bar(b, at(b, D0, 600), h=101.50, l=98.90)
    c = long_cand()
    oc = _sim(b, c)
    assert oc.reason == "stop" and gross(c, oc) == pytest.approx(-1.0)


def test_tp2_needs_a_trade_one_tick_through():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.49)
    set_bar(b, at(b, D0, 600), h=99.80)                          # disarm the time stop
    set_bar(b, at(b, D0, 610), h=101.00)                         # touch only
    c = long_cand()
    assert _sim(b, c).reason == "session_end"
    set_bar(b, at(b, D0, 615), h=101.01)
    oc = _sim(b, c)
    assert oc.reason == "tp2" and gross(c, oc) == pytest.approx(3.0)


def test_partial_then_breakeven_and_breakeven_in_the_tp1_bar():
    c = long_cand()
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.49)
    set_bar(b, at(b, D0, 600), h=100.01)                         # TP1 through by a tick
    set_bar(b, at(b, D0, 620), l=99.50)                          # back to breakeven
    oc = _sim(b, c, "pbe")
    assert oc.reason == "breakeven" and gross(c, oc) == pytest.approx(0.5)
    b2 = flat_day(D0)
    set_bar(b2, at(b2, D0, 585), l=99.49)
    set_bar(b2, at(b2, D0, 600), h=100.01, l=99.50)              # TP1 and BE in one bar: BE after TP1
    assert _sim(b2, c, "pbe").reason == "breakeven"
    b3 = flat_day(D0)
    set_bar(b3, at(b3, D0, 585), l=99.49)
    set_bar(b3, at(b3, D0, 600), h=101.01, l=99.60)              # TP1 and TP2 in one bar
    oc3 = _sim(b3, c, "pbe")
    assert oc3.reason == "tp2" and gross(c, oc3) == pytest.approx(0.5 * 1 + 0.5 * 3)
    assert gross(c, _sim(b3, c, "tp2")) == pytest.approx(3.0)


def test_short_side_is_the_mirror_of_the_long_side():
    b = flat_day(D0)
    set_bar(b, at(b, D0, 585), l=99.49)
    set_bar(b, at(b, D0, 600), h=100.01)
    set_bar(b, at(b, D0, 630), h=101.01)
    cl = long_cand()
    mid = 100.0
    actual_short = reflect(b, mid)                               # what a short sees in real prices
    cs = long_cand(L=99.50 - 2 * mid, S=99.00 - 2 * mid, T1=100.00 - 2 * mid, T2=101.00 - 2 * mid,
                   bot=99.30 - 2 * mid, side=-1)
    M.attach_outcomes([cs], actual_short, {(5, -1): actual_short.flip()})
    for mg in ("pbe", "tp2"):
        ol = _sim(b, cl, mg)
        assert cs.out[mg].reason == ol.reason and cs.out[mg].t_end == ol.t_end
    assert cs.r[("pbe", "zero")] == pytest.approx(gross(cl, _sim(b, cl, "pbe")))


# ── costs ────────────────────────────────────────────────────────────────────

def test_net_r_fees_slippage_and_borrow():
    base = M.COSTS["base"]
    c = long_cand(L=100.0, S=99.0, T1=101.0, T2=102.0)
    tp = M.Outcome(True, "tp2", 0, 0, ((1.0, 102.0, False),), 1)
    assert M.net_r(c, tp, base) == pytest.approx(2.0 - 0.001 * (100 + 102))
    sl = M.Outcome(True, "stop", 0, 0, ((1.0, 99.0, True),), 1)
    x = 99.0 * (1 - 0.0008)
    assert M.net_r(c, sl, base) == pytest.approx((x - 100) - 0.001 * (100 + x))
    s = long_cand(L=-100.0, S=-101.0, T1=-99.0, T2=-98.0, side=-1)         # short at 100, stop 101
    ss = M.Outcome(True, "stop", 0, 0, ((1.0, -101.0, True),), 3)
    xs = 101.0 * (1 + 0.0008)
    assert M.net_r(s, ss, base) == pytest.approx(-(xs - 100) - 0.001 * (100 + xs) - 0.0003 / 24 * 3 * 100)
    assert M.net_r(s, ss, M.COSTS["2x"]) < M.net_r(s, ss, base) < M.net_r(s, ss, M.COSTS["zero"])
    assert M.COSTS["2x"] == M.Cost(0.002, 0.0016, 0.0006)


# ── statistics and decision rules ───────────────────────────────────────────

def test_cluster_t():
    assert M.cluster_t(np.array([1, 2, 3, 4.0]), np.array([1, 1, 2, 2])) == pytest.approx(2.5)
    r = np.random.default_rng(0).normal(0.1, 1, 50)
    plain = r.mean() / (r.std(ddof=1) / np.sqrt(r.size))
    assert M.cluster_t(r, np.arange(50)) == pytest.approx(plain)


def _summ(exp, n=200, t=3.0, share=0.1, exp_w=None, day_share=0.1):
    return {"exp": exp, "n": n, "t": t, "max_sym_share": share,
            "exp_w": exp if exp_w is None else exp_w, "max_day_share": day_share}


def test_verdict_needs_every_rule():
    good = {"IS": _summ(0.05), "OOS": _summ(0.15), "OOS_H1": _summ(0.1), "OOS_H2": _summ(0.2)}
    port = {"exp": 0.05}
    v, failed = M.verdict(good, {"exp": 0.01}, port)
    assert v.startswith("CANDIDATE") and failed == []
    for k, bad, fail in (("IS", _summ(-0.01), "is_exp_pos"), ("OOS", _summ(0.09), "oos_exp_min"),
                         ("OOS", _summ(0.2, t=2.49), "oos_t"), ("OOS", _summ(0.2, n=99), "oos_n"),
                         ("OOS_H2", _summ(-0.1), "oos_h2_pos"), ("OOS", _summ(0.2, share=0.3), "sym_share"),
                         ("OOS", _summ(0.2, exp_w=-0.01), "oos_exp_w_pos"),
                         ("OOS", _summ(0.2, day_share=0.26), "day_share"),
                         ("OOS", _summ(0.2, share=np.nan, day_share=np.nan), "day_share")):
        v, failed = M.verdict({**good, k: bad}, {"exp": 0.01}, port)
        assert v == "rejected" and fail in failed
    assert M.verdict({**good, "OOS": _summ(0.2, t=2.50)}, {"exp": 0.01}, port)[1] == []
    assert "oos_2x_nonneg" in M.verdict(good, {"exp": -0.001}, port)[1]
    assert "port_exp_pos" in M.verdict(good, {"exp": 0.01}, {"exp": 0.0})[1]
    assert "port_exp_pos" in M.verdict(good, {"exp": 0.01}, M.summarize([]))[1]   # no portfolio trades


def _trade(sym, day, r, w=1.0, grade="A+"):
    return M.Trade(sym, day, -1, grade, w, r, r + 0.3, "tp2", day, day, day)


def test_summarize_day_share_and_risk_weighted_expectancy():
    tr = [_trade("A", 1, 3.0), _trade("B", 1, 1.0), _trade("A", 2, -1.0, w=0.5, grade="B"),
          _trade("C", 3, 1.0, w=0.5, grade="B")]
    s = M.summarize(tr)
    assert s["total_r"] == pytest.approx(4.0)
    assert s["max_day_share"] == pytest.approx(4.0 / 4.0) and s["top_day"] == 1     # day 1 = all of the net R
    assert s["max_sym_share"] == pytest.approx(2.0 / 4.0)
    assert s["exp_w"] == pytest.approx((3 + 1 - 0.5 + 0.5) / 3.0)
    assert s["exp"] == pytest.approx(1.0)
    neg = M.summarize([_trade("A", 1, 1.0, w=0.5, grade="B"), _trade("B", 2, -0.8)])
    assert neg["exp"] > 0 and neg["exp_w"] < 0                                     # A+ losers outweigh


def test_periods():
    assert M.period_of(M.day_num(date(2025, 3, 31))) == "IS"
    assert M.period_of(M.day_num(date(2025, 4, 1))) == "OOS_H1"
    assert M.period_of(M.day_num(date(2026, 1, 2))) == "OOS_H2"
    assert M.period_of(M.day_num(date(2022, 12, 30))) == "warmup"


# ── gates and limits ─────────────────────────────────────────────────────────

def _ctx(bias=1, neutral_from=2 ** 62):
    return M.DayCtx(D0, True, bias=bias, neutral_from=neutral_from)


def test_gate_bias_rules():
    on, off = M.Config(5, "tp2", True), M.Config(5, "tp2", False)
    a, b = long_cand(), long_cand()
    b.grade = "B"
    assert M.gate(a, on, _ctx(1)) == 1.0 and M.gate(b, on, _ctx(1)) == 0.5
    assert M.gate(a, on, _ctx(-1)) is None
    assert M.gate(a, on, _ctx(0)) == 0.5 and M.gate(b, on, _ctx(0)) is None     # Neutral: A+ at half risk
    t = a.t_place
    assert M.gate(b, on, _ctx(1, neutral_from=t)) is None                         # 15m MSS made it Neutral
    assert M.gate(b, on, _ctx(1, neutral_from=t + 1)) == 0.5
    assert M.gate(a, off, _ctx(-1)) == 1.0 and M.gate(b, off, _ctx(-1)) == 0.5
    b.grade = None
    assert M.gate(b, off, _ctx(1)) is None


def _fake(sym, day, place_min, r, end_min, filled=True, grade="A+", side=1):
    c = long_cand(D=day, place_min=place_min, side=side)
    c.sym, c.grade = sym, grade
    oc = M.Outcome(filled, "stop" if r < 0 else "tp2", M.ny_ms(day, end_min), M.ny_ms(day, place_min + 5))
    c.out = {"tp2": oc, "pbe": oc}
    c.r = {(mg, k): r for mg in ("tp2", "pbe") for k in M.COSTS}
    return c


def test_walk_one_at_a_time_two_per_day_and_daily_stop():
    cfg = M.Config(5, "tp2", False)
    ctxs = {"A": {D0: _ctx()}, "B": {D0: _ctx()}}
    cands = [_fake("A", D0, 575, 1.0, 600), _fake("A", D0, 590, 1.0, 610),     # 2nd overlaps the 1st
             _fake("A", D0, 605, -1.0, 620), _fake("A", D0, 625, 1.0, 640)]    # 3rd fill -> cap
    tr = M.walk(cands, ctxs, cfg, (1,), "base")
    assert [x.t_place for x in tr] == [M.ny_ms(D0, 575), M.ny_ms(D0, 605)]
    unfilled = [_fake("A", D0, 575, 1.0, 590, filled=False), _fake("A", D0, 595, 1.0, 600),
                _fake("A", D0, 605, 1.0, 610), _fake("A", D0, 615, 1.0, 620)]
    assert len(M.walk(unfilled, ctxs, cfg, (1,), "base")) == 2                  # cancelled orders don't count
    big = [_fake("A", D0, 575, -2.1, 590), _fake("A", D0, 600, 1.0, 610)]
    assert len(M.walk(big, ctxs, cfg, (1,), "base")) == 1                       # day stop at -2U
    port = [_fake("A", D0, 575, 1.0, 640), _fake("B", D0, 580, 1.0, 640), _fake("A", D0, 650, 1.0, 660)]
    assert len(M.walk(port, ctxs, cfg, (1,), "base", scope="symbol")) == 3
    assert len(M.walk(port, ctxs, cfg, (1,), "base", scope="portfolio")) == 2


def test_walk_counts_refusals_and_logs_placed_orders(monkeypatch):
    cfg = M.Config(5, "tp2", False)
    ctxs = {"A": {D0: _ctx()}}
    cands = [_fake("A", D0, 575, -0.5, 590), _fake("A", D0, 580, 1.0, 600),
             _fake("A", D0, 595, -0.5, 600), _fake("A", D0, 605, 1.0, 610)]
    st, log = {}, []
    M.walk(cands, ctxs, cfg, (1,), "base", stats=st, placed_log=log)
    assert [x[3] for x in log] == [M.ny_ms(D0, 575), M.ny_ms(D0, 595)]
    assert st["symbol_busy"] == 1 and st["day_cap"] == 1 and st.get("two_loss_binds", 0) == 0
    assert st["placed"] == 2 and st["outcome_stop"] == 2
    monkeypatch.setattr(M, "MAX_TRADES_DAY", 3)                  # only then can the two-loss rule bind
    st2 = {}
    assert len(M.walk(cands, ctxs, cfg, (1,), "base", stats=st2)) == 2 and st2["two_loss_binds"] == 1


def test_spot_tradable_cross_check_counts_days_without_spot_bars():
    b = random_walk("2024-01-01", 20, seed=4)
    d_ok, d_gap = M.day_num(date(2024, 1, 10)), M.day_num(date(2024, 1, 12))
    c = b.c.copy()
    t930 = M.ny_ms(d_gap, M.KZ_START)
    lab = pd.Timestamp(t930, unit="ms", tz="UTC").ceil("8h")
    gone = (b.t + M.BASE_MS > int(lab.timestamp() * 1000) - 8 * M.HOUR_MS) & (b.t + M.BASE_MS <= int(lab.timestamp() * 1000))
    c[gone] = np.nan                                             # no spot bar in that whole 8h window
    bb = M.Bars(b.t, b.o, b.h, b.l, c, 5, b.qv)
    assert M.spot_mismatch_days(b, [d_ok, d_gap]) == []
    assert M.spot_mismatch_days(bb, [d_ok, d_gap]) == [d_gap]


def test_walk_weekly_stop_and_sides():
    cfg = M.Config(5, "tp2", False)
    days = [D0, D0 + 1, D0 + 2, D0 + 7]                     # Mon, Tue, Wed, next Mon
    ctxs = {"A": {d: _ctx() for d in days}}
    cands = []
    for d in days[:3]:
        cands += [_fake("A", d, 575, -1.3, 590), _fake("A", d, 600, -1.3, 610)]
    cands.append(_fake("A", days[3], 575, 1.0, 590))
    tr = M.walk(cands, ctxs, cfg, (1,), "base")
    assert len(tr) == 5                                     # Wed's 2nd: week at -5.2U; next Monday resets
    assert tr[-1].day == days[3]
    assert M.walk(cands, ctxs, cfg, (-1,), "base") == []


# ── end to end on synthetic data: no lookahead, mirror symmetry ─────────────

@pytest.fixture(scope="module")
def synth():
    b = random_walk("2024-01-01", 200, seed=12)
    days = M.trading_days(date(2024, 2, 12), date(2024, 7, 15))      # spans the 2024-03-10 DST change
    return b, days


def _all_cands(prep):
    out = []
    for tf in (5, 15):
        for side in (1, -1):
            out += M.find_candidates(prep, tf, side)
    return out


def _key(c):
    return (c.tf, c.side, c.day, c.t_place, c.limit, c.stop, c.tp1, c.tp2, c.fvg_bot, c.score, c.grade,
            c.sweep_kinds, c.t_sweep, c.t_mss)


def test_end_to_end_runs_and_produces_setups(synth):
    b, days = synth
    prep = M.prepare("XUSDT", b, days)
    assert all(prep.ctx[d].eligible for d in days), [prep.ctx[d].why for d in days]
    assert {prep.ctx[d].bias for d in days} <= {-1, 0, 1}
    cands = _all_cands(prep)
    assert len(cands) >= 10 and {c.side for c in cands} == {1, -1} and {c.tf for c in cands} == {5, 15}
    for c in cands:
        assert c.stop < c.limit < c.tp1 < c.tp2 and c.tp2 - c.limit >= 2 * (c.limit - c.stop) - 1e-9
        assert M.ny_ms(c.day, M.KZ_START) <= c.t_place < M.ny_ms(c.day, M.KZ_END)
        assert c.t_sweep <= c.t_mss < c.t_place
    tfb = {k: v["b"] for k, v in prep.tf.items()}
    M.attach_outcomes(cands, b, tfb)
    for cfg in M.grid():
        for sides in M.SIDES.values():
            tr = M.walk(cands, {"XUSDT": prep.ctx}, cfg, sides, "base")
            assert all(np.isfinite(x.r) for x in tr)
            assert pd.Series([x.day for x in tr], dtype=float).value_counts().max() <= 2 if tr else True
            for x in tr:
                assert x.t_fill >= x.t_place and x.t_exit <= M.ny_ms(x.day, M.FLAT_AT)


def test_no_lookahead_future_bars_cannot_change_past_setups(synth):
    b, days = synth
    cut = M.ny_ms(days[len(days) // 2], 10 * 60 + 15)
    i = (cut - int(b.t[0])) // M.BASE_MS
    other = random_walk("2024-01-01", 200, seed=99)
    shift = b.c[i - 1] - other.o[i]
    alt = M.Bars(b.t, np.r_[b.o[:i], np.round(other.o[i:] + shift, 2)],
                 np.r_[b.h[:i], np.round(other.h[i:] + shift, 2)],
                 np.r_[b.l[:i], np.round(other.l[i:] + shift, 2)],
                 np.r_[b.c[:i], np.round(other.c[i:] + shift, 2)], 5, b.qv)
    p1, p2 = M.prepare("XUSDT", b, days), M.prepare("XUSDT", alt, days)
    before = lambda cs: sorted(_key(c) for c in cs if c.t_place <= cut)
    k1, k2 = before(_all_cands(p1)), before(_all_cands(p2))
    assert k1 == k2 and len(k1) > 0
    for d in days:
        if M.ny_ms(d, M.KZ_START) <= cut:
            c1, c2 = p1.ctx[d], p2.ctx[d]
            assert (c1.bias, c1.daily_dir, c1.p0, c1.liquidity) == (c2.bias, c2.daily_dir, c2.p0, c2.liquidity)
            assert min(c1.neutral_from, cut + 1) == min(c2.neutral_from, cut + 1)


def _splice(b, other, i):
    """b up to bar i-1, then `other`'s bars shifted to continue from b's last close."""
    shift = b.c[i - 1] - other.o[i]
    f = lambda a, x: np.r_[a[:i], np.round(x[i:] + shift, 2)]
    return M.Bars(b.t, f(b.o, other.o), f(b.h, other.h), f(b.l, other.l), f(b.c, other.c), 5, b.qv)


def _pipeline(bx, bref, days):
    """Two symbols, each the other's SMT reference; setups with SMT, outcomes."""
    px, pr = M.prepare("XUSDT", bx, days), M.prepare("BTCUSDT", bref, days)
    rx, rr = M.ref_taken_times(pr), M.ref_taken_times(px)
    cands = []
    for prep, ref in ((px, rx), (pr, rr)):
        cs = [c for tf in (5, 15) for sd in (1, -1) for c in M.find_candidates(prep, tf, sd, ref=ref)]
        M.attach_outcomes(cs, prep.b5, {k: v["b"] for k, v in prep.tf.items()})
        cands += cs
    return cands, {"XUSDT": px.ctx, "BTCUSDT": pr.ctx}


def _placed(cands, ctxs):
    out = {}
    for cfg in M.grid():
        for side_name, sides in M.SIDES.items():
            for scope in ("symbol", "portfolio"):
                for ck in ("base", "2x"):
                    log = []
                    M.walk(cands, ctxs, cfg, sides, ck, scope=scope, placed_log=log,
                           sym_order={"BTCUSDT": 0, "XUSDT": 1})
                    out[(cfg.name, side_name, scope, ck)] = log
    return out


def test_no_lookahead_at_several_cuts_including_smt_reference_and_walk(synth):
    b, days = synth
    bref = random_walk("2024-01-01", 200, seed=21, base=60.0)
    cands, ctxs = _pipeline(b, bref, days)
    assert any(c.smt for c in cands) and any(not c.smt for c in cands)
    base_placed = _placed(cands, ctxs)
    # Cuts: 09:30 on the reference symbol's first setup day, then cuts landing EXACTLY on
    # placement times (15m boundaries, 10:55), so that any peek past placement changes a setup.
    tp = sorted({c.t_place for c in cands})
    minute = lambda t: int(M.ny_day_minute(np.array([t]))[1][0])
    on15 = [t for t in tp if minute(t) % 15 == 0]
    at1055 = [t for t in tp if minute(t) == 10 * 60 + 55]
    cuts = ([M.ny_ms(min(c.day for c in cands if c.sym == "BTCUSDT"), 9 * 60 + 30)]
            + on15[:: max(1, len(on15) // 4)][:4] + (at1055[:1] or [M.ny_ms(cands[-1].day, 10 * 60 + 55)]))
    assert sum(c.t_place in cuts for c in cands) >= 3
    ox, orf = random_walk("2024-01-01", 200, seed=98), random_walk("2024-01-01", 200, seed=97, base=60.0)
    checked_cands = checked_orders = 0
    for cut in cuts:
        i = (cut - int(b.t[0])) // M.BASE_MS
        alt_c, alt_ctx = _pipeline(_splice(b, ox, i), _splice(bref, orf, i), days)
        k1 = sorted((c.sym, c.smt) + _key(c) for c in cands if c.t_place <= cut)
        k2 = sorted((c.sym, c.smt) + _key(c) for c in alt_c if c.t_place <= cut)
        assert k1 == k2
        checked_cands += len(k1)
        alt_placed = _placed(alt_c, alt_ctx)
        for key, log in base_placed.items():
            before = [x for x in log if x[3] < cut]
            assert before == [x for x in alt_placed[key] if x[3] < cut], key
            checked_orders += len(before)
    assert checked_cands > 0 and checked_orders > 0


def test_smt_uses_reference_information_only_up_to_the_reclaim(synth):
    b, days = synth
    prep = M.prepare("XUSDT", b, days)
    base = [c for tf in (5, 15) for sd in (1, -1) for c in M.find_candidates(prep, tf, sd)]
    tier12 = lambda c: any(k in M.TIER1 + M.TIER2 for k in c.sweep_kinds)
    assert any(tier12(c) for c in base)
    kinds = M.TIER1 + M.TIER2
    late = {d: {(sd, k): max(c.t_place for c in base if c.day == d) + M.BASE_MS for sd in (1, -1) for k in kinds}
            for d in {c.day for c in base}}
    early = {d: {(sd, k): min(c.t_sweep for c in base if c.day == d) for sd in (1, -1) for k in kinds}
             for d in {c.day for c in base}}
    for tf in (5, 15):
        for sd in (1, -1):
            for c in M.find_candidates(prep, tf, sd, ref=late):       # reference takes its level only later
                assert c.smt == tier12(c)
            assert not any(c.smt for c in M.find_candidates(prep, tf, sd, ref=early))


def test_walk_is_causal_future_outcomes_cannot_change_earlier_orders():
    rng = np.random.default_rng(5)
    syms = ["S0", "S1", "S2"]
    days = [D0 + k for k in range(5)]                           # Mon..Fri
    ctxs = {sy: {d: _ctx(bias=int(rng.integers(-1, 2))) for d in days} for sy in syms}

    def make(seed):
        g = np.random.default_rng(seed)
        out = []
        for d in days:
            for sy in syms:
                for place in sorted(g.choice(np.arange(575, 655, 5), 6, replace=False)):
                    c = _fake(sy, d, int(place), float(g.normal(-0.2, 1.5)), int(place) + int(g.integers(5, 60)),
                              filled=bool(g.random() < 0.7), grade=["A+", "B"][int(g.integers(0, 2))],
                              side=[1, -1][int(g.integers(0, 2))])
                    out.append(c)
        return out

    cands = make(1)
    cuts = [(k, mn) for k in range(len(days)) for mn in range(575, 665, 5)]   # every 5 min, 09:35-11:00
    for n_cut, (cut_day, cut_min) in enumerate(cuts):
        T = M.ny_ms(days[cut_day], cut_min)
        alt = make(1)
        g = np.random.default_rng(n_cut)
        for c in alt:
            oc = c.out["tp2"]
            if c.t_place >= T or oc.t_end > T:                  # anything not yet known at T
                filled = bool(g.random() < 0.5) if c.t_place >= T or oc.t_fill > T else oc.filled
                t_end = max(T + M.BASE_MS, c.t_place + M.BASE_MS) + int(g.integers(0, 12)) * M.BASE_MS
                new = M.Outcome(filled, "x", t_end, oc.t_fill)
                c.out = {"tp2": new, "pbe": new}
                r = float(g.choice([-5.0, 5.0]))                # extreme, so any peek moves a limit
                c.r = {(mg, k): r for mg in ("tp2", "pbe") for k in M.COSTS}
        for cfg in (M.Config(5, "tp2", True), M.Config(5, "pbe", False)):
            for sides in M.SIDES.values():
                for scope in ("symbol", "portfolio"):
                    l1, l2 = [], []
                    M.walk(cands, ctxs, cfg, sides, "base", scope=scope, placed_log=l1)
                    M.walk(alt, ctxs, cfg, sides, "base", scope=scope, placed_log=l2)
                    assert [x for x in l1 if x[3] < T] == [x for x in l2 if x[3] < T]


def test_long_and_short_detection_are_exact_mirrors(synth):
    b, days = synth
    mid = 100.0
    p1, p2 = M.prepare("XUSDT", b, days), M.prepare("XUSDT", reflect(b, mid), days)
    for d in days:
        assert p2.ctx[d].bias == -p1.ctx[d].bias and p2.ctx[d].daily_dir == -p1.ctx[d].daily_dir
    for tf in (5, 15):
        for side in (1, -1):
            a = M.find_candidates(p1, tf, side)
            m = M.find_candidates(p2, tf, -side)
            shifted = lambda c: (c.day, c.t_place, round(c.limit + side * 2 * mid, 6), round(c.stop + side * 2 * mid, 6),
                                 round(c.tp2 + side * 2 * mid, 6), c.score, c.sweep_kinds)
            assert sorted(map(shifted, m)) == sorted((c.day, c.t_place, round(c.limit, 6), round(c.stop, 6),
                                                      round(c.tp2, 6), c.score, c.sweep_kinds) for c in a)


def test_ineligible_day_after_missing_bars(synth):
    b, days = synth
    c = b.c.copy()
    d = days[3]
    c[(M.ny_ms(d, 3 * 60) - int(b.t[0])) // M.BASE_MS] = np.nan
    prep = M.prepare("XUSDT", M.Bars(b.t, b.o, b.h, b.l, c, 5, b.qv), days)
    assert not prep.ctx[d].eligible and not prep.ctx[d + 1].eligible   # needs D-1 00:00 .. D 09:30
    assert prep.ctx[days[5]].eligible


# ── real-data guard ──────────────────────────────────────────────────────────

def test_load_symbol_requires_offline_and_enforces_end(monkeypatch):
    from research import data
    monkeypatch.setattr(data, "OFFLINE", False)
    with pytest.raises(RuntimeError):
        M.load_symbol("BTCUSDT")
    monkeypatch.setattr(data, "OFFLINE", True)
    end = int(M.END.timestamp() * 1000)
    fake = {k: np.array([1.0]) for k in ("o", "h", "l", "c", "v", "qv")}
    monkeypatch.setattr(data, "load_klines", lambda *a, **k: {**fake, "t": np.array([end - M.BASE_MS], np.int64)})
    b = M.load_symbol("BTCUSDT")
    assert b.t[-1] == end - M.BASE_MS and b.c[-1] == 1.0 and np.isnan(b.c[0])
