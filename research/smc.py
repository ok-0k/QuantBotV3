"""
Smart-money (SMC/ICT) intraday playbook study: the model code fixed by
research/PREREG_SMC.md (section numbers "§" refer to it). Synthetic-data
tests: tests/test_research_smc.py. No real-data strategy result has been
computed with this code. The runner (research/smc_study.py) is written only
after the PM approves the pre-registration, and it must use these functions
and constants unchanged.

Conventions
- Bars sit on a regular UTC grid of OPEN times (ms). A missing bar is NaN.
  A bar's close time is its open + its timeframe.
- Every detector is written for the LONG side. Short setups run the same
  code on mirrored prices (Bars.flip: o, h, l, c -> -o, -l, -h, -c; levels
  and the daily range are negated too), so the two sides are exact mirror
  images by construction. "Long space" below means the possibly mirrored
  prices.
- atr_prev[k] is ATR14 through candle k-1: a candle is never measured
  against an ATR that already includes it. Order parameters use the ATR
  through the placement candle (atr[p]).
- All times of day are America/New_York wall-clock (DST-aware).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
MIN_MS = 60_000
BASE_MS = 5 * MIN_MS
HOUR_MS = 60 * MIN_MS
DAY_MS = 24 * HOUR_MS
EPOCH = date(1970, 1, 1)

# ── §2, §6 data and periods ──────────────────────────────────────────────────
DATA_START = pd.Timestamp("2022-10-01", tz="UTC")
END = pd.Timestamp("2026-10-07", tz="UTC")          # hard END: bars must open (and close) by it
IS_START, OOS_START, OOS_SPLIT = date(2023, 1, 1), date(2025, 4, 1), date(2026, 1, 1)
END_DAY = date(2026, 10, 7)                          # trading days are < END_DAY
HTF_MIN_FRAC = 0.9            # an aggregated bar / session needs >= 90% of its 5m bars
MIN_LISTED_DAYS = 30
MIN_QV_PER_DAY = 2e6          # trailing 7-day mean quote volume (USDT/day)

# NYSE full-day closures, 2023-01-01 .. 2026-10-06 (no NY open, so no killzone).
NYSE_HOLIDAYS = frozenset(date.fromisoformat(s) for s in (
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19",
    "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07",
))

# ── §4 model constants ───────────────────────────────────────────────────────
N_ENTRY, N_HTF = 2, 3          # swing lookback on the entry TF / on 1h, 4h, daily
ATR_N = 14
DISP_SINGLE = 1.5              # one body >= 1.5 x ATR
DISP_RUN = 1.0                 # or 2-3 same-direction bodies, each >= 1.0 x ATR ...
DISP_WICK = 0.25               # ... each with opposing wick <= 0.25 x its body
FVG_MIN_ATR = 0.25
EQ_TOL_ATR = 0.1               # equal highs/lows tolerance
EQ_WINDOW_MS = DAY_MS          # both swings within 24h; level marked for 24h after it forms
RECLAIM = 2                    # the sweep candle or one of the next two closes back
MSS_WINDOW = 12                # entry-TF candles from the sweep candle to the MSS candle
STOP_BUF_ATR, STOP_BUF_TICKS = 0.1, 2
TP1_R, TP2_MIN_R = 1.0, 2.0
ORDER_TTL_MIN = 60
TIME_STOP_MIN, TIME_STOP_R = 45, 0.5
PARTIAL = 0.5
KZ_START, KZ_END, FLAT_AT = 9 * 60 + 30, 11 * 60, 12 * 60     # minutes of the NY day
ASIA = (20 * 60, 24 * 60)      # previous evening
LONDON = (2 * 60, 5 * 60)
OVERNIGHT_FROM = 18 * 60       # previous day 18:00 .. 09:30
TOL = 1e-6                     # float tolerance, in ticks, for price comparisons

SCORE_BASE = 6                 # sweep 2 + MSS/displacement 2 + killzone 1 + DOL>=2R 1 (all required)
SCORE_A_PLUS, SCORE_B = 9, 7
RISK_UNIT = 0.005              # 1U = 0.5% of equity: A+ = 1U, B = 0.5U, A+ on a Neutral day = 0.5U
MAX_TRADES_DAY = 2
DAY_STOP_U, DAY_MAX_LOSSES, WEEK_STOP_U = -2.0, 2, -5.0

TIER1 = ("PD", "PW")
TIER2 = ("AS", "LO", "ON")


@dataclass(frozen=True)
class Cost:
    fee: float          # per fill, fraction of notional
    slip: float         # extra on market-type fills (stops, time / session exits)
    borrow_daily: float  # on short notional, charged per started hour


COSTS = {"base": Cost(0.0010, 0.0008, 0.0003), "2x": Cost(0.0020, 0.0016, 0.0006),
         "zero": Cost(0.0, 0.0, 0.0)}


@dataclass(frozen=True)
class Config:
    tf: int             # entry timeframe, minutes (5 or 15)
    mgmt: str           # "pbe": 50% at TP1 + stop to breakeven; "tp2": full size to TP2
    bias: bool          # bias gate as written (True) or off (False)

    @property
    def name(self) -> str:
        return f"smc_{self.tf}m_{self.mgmt}_{'bias' if self.bias else 'nobias'}"


def grid() -> list[Config]:
    """§9: the complete grid, 8 configurations."""
    return [Config(tf, mg, bi) for tf in (5, 15) for mg in ("pbe", "tp2") for bi in (True, False)]


SIDES = {"long": (1,), "short": (-1,), "both": (1, -1)}
DECISION_SIDES = ("short", "both")
PASS = {"oos_exp": 0.10, "oos_t": 2.0, "oos_n": 100, "sym_share": 0.25}


# ── bars and time ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Bars:
    t: np.ndarray       # open times, int64 ms, regular grid
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    tf: int             # minutes
    qv: np.ndarray | None = None

    @property
    def step(self) -> int:
        return self.tf * MIN_MS

    def flip(self) -> "Bars":
        return replace(self, o=-self.o, h=-self.l, l=-self.h, c=-self.c)

    def at(self, ms: int) -> int:
        """Index of the first bar opening at or after ms."""
        return int(-(-(ms - int(self.t[0])) // self.step))


def to_grid(k: dict, start_ms: int, end_ms: int) -> Bars:
    """research.data.load_klines output -> regular 5m grid [start, end), NaN where missing."""
    n = (end_ms - start_ms) // BASE_MS
    t = start_ms + BASE_MS * np.arange(n, dtype=np.int64)
    kt = np.asarray(k["t"], dtype=np.int64)
    sel = (kt >= start_ms) & (kt < end_ms) & ((kt - start_ms) % BASE_MS == 0)
    pos = (kt[sel] - start_ms) // BASE_MS
    out = {}
    for f in ("o", "h", "l", "c", "qv"):
        a = np.full(n, np.nan)
        a[pos] = np.asarray(k[f], dtype=float)[sel]
        out[f] = a
    return Bars(t, out["o"], out["h"], out["l"], out["c"], 5, out["qv"])


def resample(b: Bars, tf: int) -> Bars:
    """Entry-TF bars from 5m bars: a bar exists only if all its 5m bars do."""
    k = tf // b.tf
    assert (int(b.t[0]) // MIN_MS) % tf == 0, "grid must start on a TF boundary"
    n = len(b.t) // k
    blk = lambda a: a[: n * k].reshape(n, k)
    ok = ~np.isnan(blk(b.c)).any(axis=1)
    w = lambda a: np.where(ok, a, np.nan)
    qv = w(blk(b.qv).sum(axis=1)) if b.qv is not None else None
    return Bars(blk(b.t)[:, 0].copy(), w(blk(b.o)[:, 0]), w(blk(b.h).max(axis=1)),
                w(blk(b.l).min(axis=1)), w(blk(b.c)[:, -1]), tf, qv)


def ny_day_minute(t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NY local calendar day (days since 1970-01-01) and minute of day of each time."""
    loc = pd.to_datetime(np.asarray(t_ms, dtype=np.int64), unit="ms", utc=True).tz_convert(NY).tz_localize(None)
    m = loc.values.astype("datetime64[m]").astype(np.int64)
    return m // 1440, m % 1440


def ny_ms(day: int, minute: int) -> int:
    """UTC ms of NY wall-clock time `minute` on NY day `day` (never a DST-ambiguous hour here)."""
    d = EPOCH + timedelta(days=int(day))
    return int(datetime(d.year, d.month, d.day, minute // 60, minute % 60, tzinfo=NY).timestamp() * 1000)


def day_num(d: date) -> int:
    return (d - EPOCH).days


def weekday(day):
    return (day + 3) % 7          # Monday = 0 (1970-01-01 was a Thursday)


def trading_days(start: date = IS_START, end: date = END_DAY) -> list[int]:
    """NY weekdays in [start, end) that are not NYSE full-day holidays."""
    out, d = [], start
    while d < end:
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            out.append(day_num(d))
        d += timedelta(days=1)
    return out


# ── ticks (§3.4) ─────────────────────────────────────────────────────────────

def infer_ticks(b5: Bars) -> dict[tuple[int, int], float]:
    """Price tick per UTC calendar month: the largest power of ten that every
    open/high/low/close printed that month is a whole multiple of."""
    ts = pd.to_datetime(b5.t, unit="ms", utc=True)
    key = ts.year.values * 100 + ts.month.values
    ok = ~np.isnan(b5.c)
    out = {}
    for k in np.unique(key[ok]):
        sel = ok & (key == k)
        p = np.concatenate([b5.o[sel], b5.h[sel], b5.l[sel], b5.c[sel]])
        tick = None
        for d in range(-4, 13):
            x = p * 10.0 ** d
            if np.all(np.abs(x - np.round(x)) < 1e-6):
                tick = 10.0 ** -d
                break
        out[(int(k) // 100, int(k) % 100)] = tick if tick is not None else 1e-12
    return out


def _dec(tick: float) -> int:
    return int(round(-math.log10(tick)))


def snap(x: float, tick: float) -> float:
    return round(float(x), _dec(tick))


def floor_tick(x: float, tick: float) -> float:
    return snap(math.floor(x / tick + 1e-9) * tick, tick)


def ceil_tick(x: float, tick: float) -> float:
    return snap(math.ceil(x / tick - 1e-9) * tick, tick)


# ── indicators ───────────────────────────────────────────────────────────────

def atr(b: Bars, n: int = ATR_N) -> tuple[np.ndarray, np.ndarray]:
    """Wilder ATR over present bars (seed: mean of the first n true ranges).
    Returns (atr through k, atr_prev[k] = atr through k-1); missing bars carry
    the last value."""
    T = len(b.c)
    out = np.full(T, np.nan)
    a, acc, cnt, pc = np.nan, 0.0, 0, np.nan
    h, l, c = b.h, b.l, b.c
    for x in range(T):
        cx = c[x]
        if cx != cx:
            out[x] = a
            continue
        tr = h[x] - l[x] if pc != pc else max(h[x] - l[x], abs(h[x] - pc), abs(l[x] - pc))
        pc = cx
        if cnt < n:
            acc += tr
            cnt += 1
            if cnt == n:
                a = acc / n
        else:
            a += (tr - a) / n
        out[x] = a
    prev = np.r_[np.nan, out[:-1]]
    return out, prev


def swings(h: np.ndarray, l: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Swing high at i: high strictly above the n highs before and the n after
    (all present). Confirmed only at the CLOSE of candle i+n. Lows mirror."""
    T = len(h)
    sh = np.zeros(T, bool)
    sl = np.zeros(T, bool)
    if T < 2 * n + 1:
        return sh, sl
    core = slice(n, T - n)
    sh[core] = True
    sl[core] = True
    for k in range(1, n + 1):
        sh[core] &= (h[n:T - n] > h[n - k:T - n - k]) & (h[n:T - n] > h[n + k:T - n + k])
        sl[core] &= (l[n:T - n] < l[n - k:T - n - k]) & (l[n:T - n] < l[n + k:T - n + k])
    return sh, sl


@dataclass(frozen=True)
class Structure:
    brk: np.ndarray        # +1 / -1: candle CLOSED beyond the reference swing high / low
    prev: np.ndarray       # structure state before the candle (+1 up, -1 down, 0 none)
    ref: np.ndarray        # the swing price that candle broke
    state: np.ndarray      # state after the candle
    last_sh: np.ndarray    # most recent confirmed swing high price as of the candle's close
    last_sl: np.ndarray


def structure(b: Bars, n: int) -> Structure:
    """§4.2 body-close structure. The reference high is the most recent
    confirmed swing high, until a candle CLOSES above it (a break; a wick does
    not count). After a break there is no reference until a newer swing high
    is confirmed. A break against the prior state is an MSS (CHoCH), with the
    state it started from in `prev`. Swings confirmed at a candle's close are
    usable from the next candle. If one close breaks both references, both are
    consumed and the state is unchanged (no break)."""
    sh, sl = swings(b.h, b.l, n)
    T = len(b.c)
    brk = np.zeros(T, np.int8)
    prev = np.zeros(T, np.int8)
    ref = np.full(T, np.nan)
    state = np.zeros(T, np.int8)
    lsh = np.full(T, np.nan)
    lsl = np.full(T, np.nan)
    rh = rl = np.nan
    ah = al = np.nan
    st = 0
    h, l, c = b.h, b.l, b.c
    for x in range(T):
        prev[x] = st
        cx = c[x]
        if cx == cx:
            up = rh == rh and cx > rh
            dn = rl == rl and cx < rl
            if up and dn:
                rh = rl = np.nan
            elif up:
                brk[x], ref[x], st, rh = 1, rh, 1, np.nan
            elif dn:
                brk[x], ref[x], st, rl = -1, rl, -1, np.nan
        i = x - n
        if i >= 0:
            if sh[i]:
                rh = ah = h[i]
            if sl[i]:
                rl = al = l[i]
        state[x], lsh[x], lsl[x] = st, ah, al
    return Structure(brk, prev, ref, state, lsh, lsl)


def disp_leg(b: Bars, atr_prev: np.ndarray, m: int, start_min: int) -> list[int] | None:
    """§4.3 bullish displacement leg ending at candle m, starting at or after
    start_min: the longest of (m-2..m), (m-1..m) whose candles are all bullish
    with body >= DISP_RUN x ATR and upper wick <= DISP_WICK x body; else [m]
    if its body >= DISP_SINGLE x ATR; else None. (The FVG requirement is
    checked separately.)"""
    o, h, c = b.o, b.h, b.c

    def run_ok(k):
        body = c[k] - o[k]
        return body > 0 and body >= DISP_RUN * atr_prev[k] and (h[k] - c[k]) <= DISP_WICK * body

    for length in (3, 2):
        s = m - length + 1
        if s >= max(start_min, 0) and all(run_ok(k) for k in range(s, m + 1)):
            return list(range(s, m + 1))
    body = c[m] - o[m]
    if m >= start_min and body > 0 and body >= DISP_SINGLE * atr_prev[m]:
        return [m]
    return None


def bear_disp_member(b: Bars, atr_prev: np.ndarray, k: int, last: int) -> bool:
    """Candle k is part of a BEARISH displacement (single body >= 1.5 ATR, or a
    2-3 candle bearish run, each body >= 1.0 ATR with lower wick <= 0.25 body)
    using candles <= last only. Used by the breakout test (§4.5)."""
    o, l, c = b.o, b.l, b.c

    def body(x):
        return o[x] - c[x]

    def run_ok(x):
        bd = body(x)
        return bd > 0 and bd >= DISP_RUN * atr_prev[x] and (c[x] - l[x]) <= DISP_WICK * bd

    if body(k) > 0 and body(k) >= DISP_SINGLE * atr_prev[k]:
        return True
    for length in (2, 3):
        for s in range(k - length + 1, k + 1):
            e = s + length - 1
            if s >= 0 and e <= last and all(run_ok(x) for x in range(s, e + 1)):
                return True
    return False


def leg_fvgs(b: Bars, atr_prev: np.ndarray, leg: list[int], last: int) -> list[tuple[int, float, float]]:
    """§4.4 bullish FVGs whose middle candle is in the leg and whose third
    candle is <= last: (c2, bottom = high[c2-1], top = low[c2+1]), height
    >= FVG_MIN_ATR x ATR through c2-1."""
    out = []
    for c2 in leg:
        c1, c3 = c2 - 1, c2 + 1
        if c1 < 0 or c3 > last:
            continue
        bot, top = b.h[c1], b.l[c3]
        if bot < top and (top - bot) >= FVG_MIN_ATR * atr_prev[c2]:
            out.append((c2, float(bot), float(top)))
    return out


def first_through_low(l: np.ndarray, a: int, z: int, level: float, tick: float) -> int:
    """First index in [a, z) whose low trades >= 1 tick below level, or -1."""
    if z <= a:
        return -1
    hit = np.flatnonzero(l[a:z] <= level - tick + TOL * tick)
    return a + int(hit[0]) if hit.size else -1


def first_through_high(h: np.ndarray, a: int, z: int, level: float, tick: float) -> int:
    if z <= a:
        return -1
    hit = np.flatnonzero(h[a:z] >= level + tick - TOL * tick)
    return a + int(hit[0]) if hit.size else -1


def classify_sweep(b: Bars, atr_prev: np.ndarray, j: int, level: float) -> int | None:
    """§4.5 candle j traded >= 1 tick below `level` (a sell-side level).
    Breakout (None) if j and j+1 both close below it and one of them is part
    of a bearish displacement. Otherwise a sweep, reclaimed at the first of
    j, j+1, j+2 that closes above the level (returns that index); no reclaim
    within them -> None."""
    c = b.c
    n = len(c)
    c0 = c[j]
    if c0 > level:
        return j
    c1 = c[j + 1] if j + 1 < n else np.nan
    if c0 < level and c1 < level and (bear_disp_member(b, atr_prev, j, j + 1)
                                       or bear_disp_member(b, atr_prev, j + 1, j + 1)):
        return None
    for k in range(1, RECLAIM + 1):
        ck = c[j + k] if j + k < n else np.nan
        if ck > level:
            return j + k
    return None


def mss_after_sweep(b: Bars, atr_prev: np.ndarray, st: Structure, j: int, r: int) -> int | None:
    """§4.6 the FIRST bullish structure break in candles j..j+MSS_WINDOW
    decides: it is the MSS only if it flips a DOWN state, comes at or after
    the reclaim candle r, ends a displacement leg starting at or after j, and
    that leg leaves an FVG (third candle <= m+1). Otherwise the sweep is void."""
    hi = min(j + MSS_WINDOW, len(b.c) - 2)
    for m in range(j, hi + 1):
        if st.brk[m] == 1:
            if st.prev[m] != -1 or m < r:
                return None
            leg = disp_leg(b, atr_prev, m, j)
            if leg is None or not leg_fvgs(b, atr_prev, leg, m + 1):
                return None
            return m
    return None


# ── levels (§4.7) ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HTF:
    t_close: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    key: np.ndarray

    def bars(self) -> Bars:
        return Bars(self.t_close, self.o, self.h, self.l, self.c, 0)


def htf_bars(b5: Bars, day: np.ndarray, minute: np.ndarray, kind: str) -> HTF:
    """1h (clock hour), 4h (NY 00/04/08/12/16/20), 1d (NY calendar day) or 1w
    (NY Monday-Sunday) bars from 5m bars. Valid only with >= 90% of the 5m
    slots of that block present; the first and last block of the grid are
    always invalid (they may be cut by the grid edge)."""
    if kind == "1h":
        key = b5.t // HOUR_MS
    elif kind == "4h":
        key = day * 6 + minute // 240
    elif kind == "1d":
        key = day.copy()
    elif kind == "1w":
        key = day - weekday(day)
    else:
        raise ValueError(kind)
    T = len(key)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], T]
    ok = ~np.isnan(b5.c)
    present = np.add.reduceat(ok.astype(np.int64), starts)
    valid = present >= np.ceil(HTF_MIN_FRAC * (ends - starts) - 1e-9)
    valid[0] = valid[-1] = False
    pos = np.arange(T)
    first = np.minimum.reduceat(np.where(ok, pos, T - 1), starts)
    last = np.maximum.reduceat(np.where(ok, pos, 0), starts)
    w = lambda a: np.where(valid, a, np.nan)
    return HTF(b5.t[ends - 1] + BASE_MS, w(b5.o[first]),
               w(np.maximum.reduceat(np.where(ok, b5.h, -np.inf), starts)),
               w(np.minimum.reduceat(np.where(ok, b5.l, np.inf), starts)),
               w(b5.c[last]), key[starts])


def _session_hl(b5: Bars, target: np.ndarray, sel: np.ndarray) -> dict[int, tuple[float, float]]:
    df = pd.DataFrame({"tg": target[sel], "h": b5.h[sel], "l": b5.l[sel], "ok": ~np.isnan(b5.c[sel])})
    g = df.groupby("tg").agg(h=("h", "max"), l=("l", "min"), n=("ok", "size"), k=("ok", "sum"))
    g = g[g["k"] >= np.ceil(HTF_MIN_FRAC * g["n"] - 1e-9)]
    return {int(d): (float(r.h), float(r.l)) for d, r in zip(g.index, g.itertuples())}


@dataclass(frozen=True)
class Level:
    kind: str      # PD, PW, AS, LO, ON (Tier 1/2) or EQ (Tier 3)
    price: float   # long-space price
    avail: int     # ms from which it is marked
    expire: int = 2 ** 62


def day_levels(b5: Bars, day: np.ndarray, minute: np.ndarray, d1: HTF, w1: HTF) -> dict[int, dict]:
    """Tier 1/2 levels per NY trading day (actual prices): {kind: (high, low,
    avail_ms)}. PD = previous NY calendar day, PW = previous NY week (Mon-Sun),
    AS = Asia 20:00-24:00 the evening before, LO = London 02:00-05:00,
    ON = overnight 18:00 (previous day) - 09:30."""
    asia = _session_hl(b5, day + 1, minute >= ASIA[0])
    lon = _session_hl(b5, day, (minute >= LONDON[0]) & (minute < LONDON[1]))
    on_t = np.where(minute >= OVERNIGHT_FROM, day + 1, day)
    on = _session_hl(b5, on_t, (minute >= OVERNIGHT_FROM) | (minute < KZ_START))
    dmap = {int(k): (float(h), float(lo)) for k, h, lo in zip(d1.key, d1.h, d1.l) if h == h}
    wmap = {int(k): (float(h), float(lo)) for k, h, lo in zip(w1.key, w1.h, w1.l) if h == h}
    out = {}
    for D in np.unique(day):
        D = int(D)
        mon = D - int(weekday(D))
        lv = {}
        for kind, src, key, minute_avail, avail_day in (
                ("PD", dmap, D - 1, 0, D), ("PW", wmap, mon - 7, 0, mon), ("AS", asia, D, 0, D),
                ("LO", lon, D, LONDON[1], D), ("ON", on, D, KZ_START, D)):
            if key in src:
                hi, lo = src[key]
                lv[kind] = (hi, lo, ny_ms(avail_day, minute_avail))
        out[D] = lv
    return out


def eq_levels(b: Bars, atr_now: np.ndarray, n: int, tick) -> list[Level]:
    """Tier 3 equal lows (long space) over the whole series: swing lows i1 < i2
    whose candles open within 24h of each other, |low1 - low2| <= 0.1 x ATR
    (through i2's confirmation candle), with no candle strictly between them
    trading >= 1 tick below the lower one. Level = the lower low, marked from
    i2's confirmation close for 24h. `tick` may be a function of the candle."""
    _, sl = swings(b.h, b.l, n)
    idx = np.flatnonzero(sl)
    if idx.size < 2:
        return []
    px = b.l[idx]
    tt = b.t[idx]
    out = []
    for q in range(1, idx.size):
        i2 = int(idx[q])
        if i2 + n >= len(b.c):
            break
        lo = np.searchsorted(tt, tt[q] - EQ_WINDOW_MS, "left")
        if lo >= q:
            continue
        tol = EQ_TOL_ATR * atr_now[i2 + n]
        cand = np.flatnonzero(np.abs(px[lo:q] - px[q]) <= tol) + lo
        if cand.size == 0:
            continue
        conf = int(b.t[i2 + n]) + b.step
        tk = tick(i2) if callable(tick) else tick
        for p in cand:
            i1 = int(idx[p])
            lvl = min(float(px[p]), float(px[q]))
            between = b.l[i1 + 1:i2]
            if between.size == 0 or not np.any(between <= lvl - tk + TOL * tk):
                out.append(Level("EQ", lvl, conf, conf + EQ_WINDOW_MS))
    return out


# ── per-day context: eligibility and bias (§3.6, §5) ────────────────────────

@dataclass
class DayCtx:
    day: int
    eligible: bool
    why: str = ""
    levels: dict = field(default_factory=dict)  # kind -> (high, low, avail_ms), actual prices
    bias: int = 0                 # at 09:30: +1 Bullish, -1 Bearish, 0 Neutral
    neutral_from: int = 2 ** 62   # ms from which a 15m MSS against the bias made the day Neutral
    daily_dir: int = 0            # daily structure state at the previous NY day's close
    eq: float = np.nan            # daily dealing-range midpoint
    p0: float = np.nan            # close of the 09:25-09:30 5m bar
    structure_vote: str = ""
    liquidity: int = 0
    tick: float = np.nan

    def bias_at(self, t_ms: int) -> int:
        return 0 if t_ms >= self.neutral_from else self.bias


def _asof(t_close: np.ndarray, t_ms: int) -> int:
    return int(np.searchsorted(t_close, t_ms, "right")) - 1


def bias_rule(votes: tuple[int, int, int], liquidity: int, p0: float, eq: float) -> tuple[int, str]:
    """§5.4 exact bias table. votes = (daily, 4h, 1h) structure states."""
    up = sum(v == 1 for v in votes)
    dn = sum(v == -1 for v in votes)
    if up and dn:
        return 0, "conflict"
    s = 1 if up >= 2 else (-1 if dn >= 2 else 0)
    disc = p0 < eq if eq == eq else False
    prem = p0 > eq if eq == eq else False
    if s == 1:
        return (1 if liquidity == 1 or (liquidity == 0 and disc) else 0), "up"
    if s == -1:
        return (-1 if liquidity == -1 or (liquidity == 0 and prem) else 0), "down"
    if liquidity == 1 and disc:
        return 1, "none"
    if liquidity == -1 and prem:
        return -1, "none"
    return 0, "none"


def _taken_window(b5: Bars, a: int, z: int, level: float, hi: bool, tick: float) -> bool:
    if hi:
        return first_through_high(b5.h, a, z, level, tick) >= 0
    return first_through_low(b5.l, a, z, level, tick) >= 0


def neutral_time(b15: Bars, atr15_prev: np.ndarray, st15: Structure, D: int) -> int:
    """§5.5 earliest confirmation time (close of the FVG's third candle) of a
    15m displacement MSS whose MSS candle opens in [09:30, 11:00) on day D,
    in LONG space (pass mirrored bars and structure for a bullish bias)."""
    a, z = b15.at(ny_ms(D, KZ_START)), b15.at(ny_ms(D, KZ_END))
    for m in range(a, min(z, len(b15.c) - 1)):
        if st15.brk[m] == 1 and st15.prev[m] == -1:
            leg = disp_leg(b15, atr15_prev, m, 0)
            if leg is not None and leg_fvgs(b15, atr15_prev, leg, m + 1):
                return int(b15.t[m + 1]) + b15.step
    return 2 ** 62


# ── symbol container and setup detection ────────────────────────────────────

@dataclass
class Prepared:
    """Everything derived from one symbol's 5m bars, independent of config."""
    sym: str
    b5: Bars
    day5: np.ndarray
    min5: np.ndarray
    ticks: dict
    ctx: dict                       # day -> DayCtx
    tf: dict                        # (tf, side) -> dict(b=Bars long space, atr, atr_prev, st, eq)
    first_ms: int


def prepare(sym: str, b5: Bars, days: list[int]) -> Prepared:
    day5, min5 = ny_day_minute(b5.t)
    ticks = infer_ticks(b5)
    d1, h4, h1, w1 = (htf_bars(b5, day5, min5, k) for k in ("1d", "4h", "1h", "1w"))
    sd, s4, s1 = (structure(x.bars(), N_HTF) for x in (d1, h4, h1))
    levels = day_levels(b5, day5, min5, d1, w1)
    b15 = resample(b5, 15)
    tfd = {}
    for tf, b in ((5, b5), (15, b15)):
        a_now, a_prev = atr(b)
        for side in (1, -1):
            bb = b if side == 1 else b.flip()
            tick_of = (lambda i, _b=bb: _tick_at(ticks, int(_b.t[i])))
            tfd[(tf, side)] = dict(b=bb, atr=a_now, atr_prev=a_prev, st=structure(bb, N_ENTRY),
                                   eq=eq_levels(bb, a_now, N_ENTRY, tick_of))
    ok = ~np.isnan(b5.c)
    first_ms = int(b5.t[np.argmax(ok)]) if ok.any() else 2 ** 62
    qv = np.where(ok, b5.qv if b5.qv is not None else 0.0, 0.0)
    cqv = np.r_[0.0, np.cumsum(qv)]
    cok = np.r_[0, np.cumsum(ok)]
    ctx = {}
    for D in days:
        ctx[D] = _day_ctx(D, b5, ok, cqv, cok, first_ms, ticks, levels, d1, h4, h1, sd, s4, s1, tfd)
    return Prepared(sym, b5, day5, min5, ticks, ctx, tfd, first_ms)


def _tick_at(ticks: dict, t_ms: int) -> float:
    ts = pd.Timestamp(t_ms, unit="ms", tz="UTC")
    return ticks.get((ts.year, ts.month), 1e-12)


def _day_ctx(D, b5, ok, cqv, cok, first_ms, ticks, levels, d1, h4, h1, sd, s4, s1, tfd) -> DayCtx:
    d0 = ny_ms(D, 0)
    i_prev0, i_kz = b5.at(ny_ms(D - 1, 0)), b5.at(ny_ms(D, KZ_START))
    lv = levels.get(D, {})
    tick = ticks.get(((EPOCH + timedelta(days=D)).year, (EPOCH + timedelta(days=D)).month))
    ctx = DayCtx(D, False, levels=lv, tick=tick if tick else np.nan)
    if i_prev0 < 0 or i_kz > len(b5.t):
        ctx.why = "outside data"
        return ctx
    if cok[i_kz] - cok[i_prev0] != i_kz - i_prev0:
        ctx.why = "missing 5m bars before the session"
        return ctx
    if "PW" not in lv or any(k not in lv for k in ("PD", "AS", "LO", "ON")):
        ctx.why = "missing levels"
        return ctx
    if first_ms > d0 - MIN_LISTED_DAYS * DAY_MS:
        ctx.why = "listed < 30 days"
        return ctx
    i7 = b5.at(ny_ms(D - 7, 0))
    if i7 < 0 or (cqv[b5.at(d0)] - cqv[i7]) / 7.0 < MIN_QV_PER_DAY:
        ctx.why = "illiquid"
        return ctx
    ctx.eligible = True
    # §5 bias
    ctx.daily_dir = int(sd.state[_asof(d1.t_close, d0)])
    t930 = ny_ms(D, KZ_START)
    votes = (ctx.daily_dir, int(s4.state[_asof(h4.t_close, t930)]), int(s1.state[_asof(h1.t_close, t930)]))
    xd = _asof(d1.t_close, d0)
    ctx.eq = 0.5 * (sd.last_sh[xd] + sd.last_sl[xd])
    ctx.p0 = float(b5.c[i_kz - 1])
    lows, highs = [], []
    for kind in ("PD", "PW", "AS"):
        hi, lo, av = lv[kind]
        a = b5.at(max(av, ny_ms(D - 1, OVERNIGHT_FROM)))
        if _taken_window(b5, a, i_kz, lo, False, tick):
            lows.append(lo)
        if _taken_window(b5, a, i_kz, hi, True, tick):
            highs.append(hi)
    bull = any(ctx.p0 > lo for lo in lows)
    bear = any(ctx.p0 < hi for hi in highs)
    ctx.liquidity = 1 if (bull and not highs) else (-1 if (bear and not lows) else 0)
    ctx.bias, ctx.structure_vote = bias_rule(votes, ctx.liquidity, ctx.p0, ctx.eq)
    if ctx.bias != 0:
        side = -ctx.bias                       # the direction AGAINST the bias, as long space
        e = tfd[(15, side)]
        ctx.neutral_from = neutral_time(e["b"], e["atr_prev"], e["st"], D)
    return ctx


@dataclass
class Candidate:
    sym: str
    tf: int
    side: int            # +1 long, -1 short
    day: int
    t_place: int         # ms: close of the placement candle p = m+1
    p: int               # placement candle index (entry TF)
    limit: float         # LONG-space prices (actual = side x value)
    stop: float
    tp1: float
    tp2: float
    fvg_bot: float
    tick: float
    score: int
    grade: str | None    # "A+", "B" or None (score < 7: no trade)
    aligned: bool
    discount: bool
    smt: bool
    sweep_kinds: tuple
    t_sweep: int         # open of the earliest sweep candle
    t_mss: int           # open of the MSS candle
    out: dict = field(default_factory=dict)     # mgmt -> Outcome
    r: dict = field(default_factory=dict)       # (mgmt, cost key) -> net R

    def actual(self, x: float) -> float:
        return self.side * x


def ref_taken_times(prep: Prepared) -> dict[int, dict]:
    """For SMT (§4.10): per trading day, the close time of the first 5m bar
    that traded >= 1 tick through each Tier 1/2 level of this (reference)
    symbol, from the level's availability to 11:00; inf if not taken."""
    out = {}
    b5 = prep.b5
    for D, ctx in prep.ctx.items():
        if not ctx.levels or ctx.tick != ctx.tick:
            continue
        z = b5.at(ny_ms(D, KZ_END))
        rec = {}
        for kind, (hi, lo, av) in ctx.levels.items():
            a = b5.at(av)
            jl = first_through_low(b5.l, a, z, lo, ctx.tick)
            jh = first_through_high(b5.h, a, z, hi, ctx.tick)
            rec[(1, kind)] = int(b5.t[jl]) + BASE_MS if jl >= 0 else 2 ** 62
            rec[(-1, kind)] = int(b5.t[jh]) + BASE_MS if jh >= 0 else 2 ** 62
        out[D] = rec
    return out


def find_candidates(prep: Prepared, tf: int, side: int, ref: dict | None = None,
                    counts: dict | None = None) -> list[Candidate]:
    """§4.5-§4.10 setups for one symbol, entry TF and side, for every eligible
    trading day. Independent of management, bias gate and costs."""
    e = prep.tf[(tf, side)]
    b, a_now, a_prev, st = e["b"], e["atr"], e["atr_prev"], e["st"]
    eqs = e["eq"]
    eq_avail = np.array([x.avail for x in eqs], dtype=np.int64)
    cnt = counts if counts is not None else {}
    out = []
    for D, ctx in prep.ctx.items():
        if not ctx.eligible:
            continue
        tick = ctx.tick
        d0, kz0, kz1 = ny_ms(D, 0), ny_ms(D, KZ_START), ny_ms(D, KZ_END)
        z, day_a = b.at(kz1), b.at(d0)
        sweep_lv, dol_lv = [], []
        for kind, (hi, lo, av) in ctx.levels.items():
            sw, dl = (lo, hi) if side == 1 else (-hi, -lo)
            sweep_lv.append(Level(kind, sw, av))
            dol_lv.append(Level(kind, dl, av))
        sel = (eq_avail < kz1) & (eq_avail + EQ_WINDOW_MS > d0)
        sweep_lv += [eqs[i] for i in np.flatnonzero(sel)]
        # sweeps: each level's first trade-through candle, classified
        events = []
        for lv in sweep_lv:
            a = b.at(lv.avail)
            j = first_through_low(b.l, a, min(z, b.at(lv.expire)), lv.price, tick)
            if j < day_a:               # untaken, or taken before day D (already used up)
                continue
            r = classify_sweep(b, a_prev, j, lv.price)
            cnt["levels_taken"] = cnt.get("levels_taken", 0) + 1
            if r is None:
                cnt["breakouts_or_no_reclaim"] = cnt.get("breakouts_or_no_reclaim", 0) + 1
                continue
            m = mss_after_sweep(b, a_prev, st, j, r)
            if m is None:
                continue
            events.append((m, j, r, lv.kind))
        by_m: dict[int, list] = {}
        for m, j, r, kind in events:
            by_m.setdefault(m, []).append((j, r, kind))
        for m in sorted(by_m):
            evs = by_m[m]
            cnt["mss"] = cnt.get("mss", 0) + 1
            p = m + 1
            t_place = int(b.t[p]) + b.step
            if not (kz0 <= t_place < kz1) or b.c[p] != b.c[p]:
                continue
            cnt["mss_in_killzone"] = cnt.get("mss_in_killzone", 0) + 1
            j0 = min(x[0] for x in evs)
            leg = disp_leg(b, a_prev, m, j0)
            h_ref = st.ref[m]
            best = None
            for c2, bot, top in leg_fvgs(b, a_prev, leg, p):
                ce = 0.5 * (bot + top)
                if c2 + 1 < p:
                    seg = slice(c2 + 2, p + 1)
                    if np.any(b.l[seg] <= ce) or np.any(b.c[seg] < bot):
                        continue
                dist = 0.0 if bot <= h_ref <= top else min(abs(h_ref - bot), abs(h_ref - top))
                if best is None or dist < best[0] or (dist == best[0] and c2 > best[1]):
                    best = (dist, c2, bot, top)
            if best is None:
                cnt["no_valid_fvg"] = cnt.get("no_valid_fvg", 0) + 1
                continue
            _, c2, bot, top = best
            sweep_low = float(np.nanmin(b.l[j0:p + 1]))
            limit = floor_tick(0.5 * (bot + top), tick)
            stop = floor_tick(sweep_low - max(STOP_BUF_TICKS * tick, STOP_BUF_ATR * a_now[p]), tick)
            risk = limit - stop
            if not risk > 0:
                continue
            tp1 = ceil_tick(limit + TP1_R * risk, tick)
            dols = []
            for lv in dol_lv:
                if lv.avail > t_place or not lv.price > limit:
                    continue
                if first_through_high(b.h, b.at(lv.avail), p + 1, lv.price, tick) < 0:
                    dols.append(lv.price)
            if not dols or min(dols) - limit < TP2_MIN_R * risk - TOL * tick:
                cnt["dol_lt_2r"] = cnt.get("dol_lt_2r", 0) + 1
                continue
            tp2 = min(dols)
            aligned = ctx.daily_dir * side == 1
            eq_side = ctx.eq * side
            discount = bool(limit < eq_side) if eq_side == eq_side else False
            smt = False
            if ref is not None and D in ref:
                for j, r, kind in evs:
                    if kind in TIER1 + TIER2:
                        rt = ref[D].get((side, kind))
                        if rt is not None and rt > int(b.t[r]) + b.step:
                            smt = True
            score = SCORE_BASE + 2 * aligned + discount + smt
            grade = "A+" if score >= SCORE_A_PLUS else ("B" if score >= SCORE_B else None)
            cnt["setups"] = cnt.get("setups", 0) + 1
            out.append(Candidate(prep.sym, tf, side, D, t_place, p, limit, stop, tp1, tp2, bot, tick,
                                 score, grade, aligned, discount, smt,
                                 tuple(sorted({x[2] for x in evs})), int(b.t[j0]), int(b.t[m])))
    return out


# ── order and position simulation (§6) ──────────────────────────────────────

@dataclass(frozen=True)
class Outcome:
    filled: bool
    reason: str          # cancel reason, or the final exit reason
    t_end: int           # ms from which the symbol is free again
    t_fill: int = 0      # open of the 5m fill bar
    exits: tuple = ()    # ((fraction, long-space price, market?), ...)
    hours: int = 0       # started hours in the position (borrow)


def entry_closes(b5: Bars, btf: Bars) -> np.ndarray:
    """Per 5m slot: the close of the entry-TF candle that ends with that slot
    (NaN elsewhere or if that candle is missing). Long space."""
    if btf.tf == b5.tf:
        return btf.c
    k = btf.tf // b5.tf
    out = np.full(len(b5.c), np.nan)
    out[k - 1::k][: len(btf.c)] = btf.c[: len(out[k - 1::k])]
    return out


def simulate(b5: Bars, tfc: np.ndarray, cand: Candidate, mgmt: str) -> Outcome:
    """§6 one order's life on 5m bars (long space). Pending: TP1 touched ->
    cancel (checked first in each bar); a low >= 1 tick through the limit ->
    filled at the limit; an entry-TF close below the FVG bottom -> cancel;
    live for bars opening before min(placement + 60 min, 11:00). In the
    position: time/session exits at the open of the first bar at or after
    their time; a stop touch fills at the stop or the bar's open if it gapped
    through; stop before any target in the same bar; targets need a trade
    1 tick through; the fill bar can only stop out."""
    o, h, l, c, t = b5.o, b5.h, b5.l, b5.c, b5.t
    L, S, T1, T2, tk = cand.limit, cand.stop, cand.tp1, cand.tp2, cand.tick
    eps = TOL * tk
    D = cand.day
    live_end = min(cand.t_place + ORDER_TTL_MIN * MIN_MS, ny_ms(D, KZ_END))
    flat = ny_ms(D, FLAT_AT)
    n = len(t)
    q = b5.at(cand.t_place)
    f = -1
    while q < n and t[q] < live_end:
        if c[q] == c[q]:
            if h[q] >= T1 - eps:
                return Outcome(False, "tp1_first", int(t[q]) + BASE_MS)
            if l[q] <= L - tk + eps:
                f = q
                break
            if tfc[q] == tfc[q] and tfc[q] < cand.fvg_bot - eps:
                return Outcome(False, "fvg_invalid", int(t[q]) + BASE_MS)
        q += 1
    if f < 0:
        return Outcome(False, "expired" if live_end < ny_ms(D, KZ_END) else "killzone_end", int(live_end))
    t_fill = int(t[f])
    hours = lambda end: max(1, -(-(end - t_fill) // HOUR_MS))
    if l[f] <= S + eps:
        px = o[f] if o[f] <= S else S
        return Outcome(True, "stop", t_fill + BASE_MS, t_fill, ((1.0, float(px), True),), hours(t_fill + BASE_MS))
    risk = L - S
    rem, stop, tp1_done, armed = 1.0, S, False, True
    deadline = t_fill + BASE_MS + TIME_STOP_MIN * MIN_MS
    exits: list = []
    q = f + 1
    last_ok = f
    while q < n:
        timed = "session_end" if t[q] >= flat else ("time_stop" if armed and t[q] >= deadline else None)
        if timed:
            while q < n and c[q] != c[q]:
                q += 1
            if q >= n:
                break
            exits.append((rem, float(o[q]), True))
            return Outcome(True, timed, int(t[q]), t_fill, tuple(exits), hours(int(t[q])))
        if c[q] != c[q]:
            q += 1
            continue
        last_ok = q
        end = int(t[q]) + BASE_MS
        if l[q] <= stop + eps:
            px = o[q] if o[q] <= stop else stop
            exits.append((rem, float(px), True))
            return Outcome(True, "breakeven" if tp1_done else "stop", end, t_fill, tuple(exits), hours(end))
        if mgmt == "pbe" and not tp1_done and h[q] >= T1 + tk - eps:
            exits.append((PARTIAL, T1, False))
            rem, stop, tp1_done, armed = 1.0 - PARTIAL, L, True, False
            if l[q] <= L + eps:
                exits.append((rem, L, True))
                return Outcome(True, "breakeven", end, t_fill, tuple(exits), hours(end))
        if h[q] >= T2 + tk - eps:
            exits.append((rem, T2, False))
            return Outcome(True, "tp2", end, t_fill, tuple(exits), hours(end))
        if h[q] >= L + TIME_STOP_R * risk - eps:
            armed = False
        q += 1
    exits.append((rem, float(c[last_ok]), True))           # data ended (delisting)
    end = int(t[last_ok]) + BASE_MS
    return Outcome(True, "data_end", end, t_fill, tuple(exits), hours(end))


def net_r(cand: Candidate, oc: Outcome, cost: Cost) -> float:
    """§7 net P&L in R (1R = |limit - stop| x size). Actual prices; fee on
    every fill; slippage on market-type fills; borrow on shorts per started
    hour on the entry notional."""
    s = cand.side
    L, S = s * cand.limit, s * cand.stop
    q = 1.0 / abs(L - S)
    pnl = 0.0
    fees = cost.fee * q * L
    for frac, x_long, market in oc.exits:
        x = s * x_long
        if market:
            x = x * (1.0 - s * cost.slip)
        pnl += frac * s * (x - L) * q
        fees += cost.fee * q * frac * x
    borrow = cost.borrow_daily / 24.0 * oc.hours * q * L if s == -1 else 0.0
    return pnl - fees - borrow


# ── portfolio walk: gates and limits (§8) ───────────────────────────────────

@dataclass
class Trade:
    sym: str
    day: int
    side: int
    grade: str
    w: float             # risk weight in U
    r: float             # net R under the walk's cost case
    r_gross: float       # same trade at zero cost (diagnostic)
    reason: str
    t_place: int
    t_fill: int
    t_exit: int


def gate(cand: Candidate, cfg: Config, ctx: DayCtx) -> float | None:
    """§8.1 risk weight in U, or None if not allowed."""
    if cand.grade is None:
        return None
    full = 1.0 if cand.grade == "A+" else 0.5
    if not cfg.bias:
        return full
    b = ctx.bias_at(cand.t_place)
    if b == cand.side:
        return full
    if b == 0 and cand.grade == "A+":
        return 0.5
    return None


def walk(cands: list[Candidate], ctxs: dict, cfg: Config, sides: tuple, cost_key: str,
         scope: str = "symbol", sym_order: dict | None = None) -> list[Trade]:
    """§8 apply the bias gate and the risk limits to the candidate stream in
    placement order (ties: SYMBOLS order, then long before short).
    scope="symbol": every symbol is its own playbook account (the decision
    run); scope="portfolio": one shared account (reported only).
    Limits per account: at most MAX_TRADES_DAY filled-or-pending orders per NY
    day; no new order once the day's realized U <= -2 or its last two closed
    trades both lost; none once the week's realized U <= -5; one order or
    position per symbol at a time."""
    so = sym_order or {}
    order = sorted((c for c in cands if c.side in sides and c.tf == cfg.tf),
                   key=lambda c: (c.t_place, so.get(c.sym, 0), -c.side))
    busy: dict[str, int] = {}
    placed: dict[tuple, list] = {}    # (account, day) -> [(filled, t_end, trade or None)]
    trades: list[Trade] = []
    for c in order:
        ctx = ctxs[c.sym][c.day]
        w = gate(c, cfg, ctx)
        if w is None or busy.get(c.sym, -1) > c.t_place:
            continue
        acct = c.sym if scope == "symbol" else "ALL"
        today = placed.setdefault((acct, c.day), [])
        if sum(1 for filled, t_end, tr in today if filled or t_end > c.t_place) >= MAX_TRADES_DAY:
            continue
        closed = sorted((tr for _, _, tr in today if tr is not None and tr.t_exit <= c.t_place),
                        key=lambda x: x.t_exit)
        if sum(x.w * x.r for x in closed) <= DAY_STOP_U:
            continue
        if len(closed) >= DAY_MAX_LOSSES and all(x.r < 0 for x in closed[-DAY_MAX_LOSSES:]):
            continue
        mon = c.day - int(weekday(c.day))
        week_u = sum(tr.w * tr.r for d in range(mon, c.day + 1) for _, _, tr in placed.get((acct, d), ())
                     if tr is not None and tr.t_exit <= c.t_place)
        if week_u <= WEEK_STOP_U:
            continue
        oc = c.out[cfg.mgmt]
        busy[c.sym] = oc.t_end
        tr = None
        if oc.filled:
            tr = Trade(c.sym, c.day, c.side, c.grade, w, c.r[(cfg.mgmt, cost_key)], c.r[(cfg.mgmt, "zero")],
                       oc.reason, c.t_place, oc.t_fill, oc.t_end)
            trades.append(tr)
        today.append((oc.filled, oc.t_end, tr))
    return trades


def attach_outcomes(cands: list[Candidate], b5: Bars, tf_bars: dict) -> None:
    """Simulate every candidate under both managements and price it under
    every cost case. tf_bars: (tf, side) -> long-space entry-TF Bars."""
    cache = {}
    for c in cands:
        key = (c.tf, c.side)
        if key not in cache:
            b5s = b5 if c.side == 1 else b5.flip()
            cache[key] = (b5s, entry_closes(b5s, tf_bars[key]))
        b5s, tfc = cache[key]
        for mg in ("pbe", "tp2"):
            oc = simulate(b5s, tfc, c, mg)
            c.out[mg] = oc
            for ck, cost in COSTS.items():
                c.r[(mg, ck)] = net_r(c, oc, cost) if oc.filled else np.nan


# ── statistics and decision rules (§10, §11) ────────────────────────────────

def cluster_t(r: np.ndarray, day: np.ndarray) -> float:
    """t-stat of the mean with standard errors clustered by NY trading day:
    SE = sqrt(G/(G-1) * sum_g (sum_{i in g} (r_i - mean))^2) / N."""
    r = np.asarray(r, float)
    n = r.size
    if n < 2:
        return np.nan
    e = r - r.mean()
    g = pd.Series(e).groupby(np.asarray(day)).sum().to_numpy()
    G = g.size
    if G < 2:
        return np.nan
    se = math.sqrt(G / (G - 1) * float((g ** 2).sum())) / n
    return float(r.mean() / se) if se > 0 else np.nan


def max_dd(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    eq = np.r_[0.0, np.cumsum(x)]
    return float((eq - np.maximum.accumulate(eq)).min())


def period_of(day: int) -> str:
    d = EPOCH + timedelta(days=int(day))
    if d < IS_START:
        return "warmup"
    if d < OOS_START:
        return "IS"
    return "OOS_H1" if d < OOS_SPLIT else "OOS_H2"


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "exp": np.nan, "t": np.nan, "total_r": 0.0, "max_sym_share": np.nan}
    tr = sorted(trades, key=lambda x: (x.t_exit, x.sym))
    r = np.array([x.r for x in tr])
    w = np.array([x.w for x in tr])
    day = np.array([x.day for x in tr])
    wins, losses = r[r > 0], r[r < 0]
    by_sym = pd.Series(r).groupby([x.sym for x in tr]).sum()
    tot = float(r.sum())
    per_day_u = pd.Series(w * r).groupby(day).sum()
    return {
        "n": int(r.size), "exp": float(r.mean()), "median": float(np.median(r)), "t": cluster_t(r, day),
        "win_rate": float((r > 0).mean()), "avg_win": float(wins.mean()) if wins.size else np.nan,
        "avg_loss": float(losses.mean()) if losses.size else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.size and losses.sum() < 0 else np.nan,
        "total_r": tot, "total_u": float((w * r).sum()), "exp_w": float((w * r).sum() / w.sum()),
        "mdd_r": max_dd(r), "mdd_u": max_dd(w * r),
        "exp_gross": float(np.mean([x.r_gross for x in tr])),
        "cost_per_trade_r": float(np.mean([x.r_gross - x.r for x in tr])),
        "days": int(per_day_u.size), "worst_day_u": float(per_day_u.min()),
        "max_trades_day": int(pd.Series(day).value_counts().max()),
        "max_sym_share": float(by_sym.max() / tot) if tot > 0 else np.nan,
        "top_sym": str(by_sym.idxmax()),
        "n_a_plus": int(sum(x.grade == "A+" for x in tr)),
        "reasons": pd.Series([x.reason for x in tr]).value_counts().to_dict(),
    }


def split(trades: list[Trade]) -> dict[str, list[Trade]]:
    out = {"IS": [], "OOS": [], "OOS_H1": [], "OOS_H2": []}
    for x in trades:
        p = period_of(x.day)
        if p == "IS":
            out["IS"].append(x)
        elif p.startswith("OOS"):
            out["OOS"].append(x)
            out[p].append(x)
    return out


def verdict(base: dict[str, dict], stress: dict) -> tuple[str, list[str]]:
    """§11 base: summarize() per period at base cost; stress: OOS at 2x cost."""
    o = base["OOS"]
    checks = {
        "is_exp_pos": base["IS"].get("exp", np.nan) > 0,
        "oos_exp_min": o.get("exp", np.nan) >= PASS["oos_exp"],
        "oos_t": o.get("t", np.nan) >= PASS["oos_t"],
        "oos_n": o.get("n", 0) >= PASS["oos_n"],
        "oos_h1_pos": base["OOS_H1"].get("exp", np.nan) > 0,
        "oos_h2_pos": base["OOS_H2"].get("exp", np.nan) > 0,
        "oos_2x_nonneg": stress.get("exp", np.nan) >= 0,
        "sym_share": o.get("max_sym_share", np.nan) <= PASS["sym_share"],
    }
    failed = [k for k, v in checks.items() if not v]
    return ("CANDIDATE (needs independent re-run + forward test)" if not failed else "rejected"), failed


# ── real-data loading (runner only) ─────────────────────────────────────────

def load_symbol(sym: str) -> Bars:
    """Cached 5m klines on the study grid [DATA_START, END). Refuses to run
    unless RESEARCH_OFFLINE=1 (no requests, no cache writes)."""
    from research import data
    if not data.OFFLINE:
        raise RuntimeError("set RESEARCH_OFFLINE=1: the study reads the cache only")
    s, e = int(DATA_START.timestamp() * 1000), int(END.timestamp() * 1000)
    k = data.load_klines(sym, "5m", s, e)
    assert k["t"].size == 0 or int(k["t"].max()) + BASE_MS <= e, "data past END"
    return to_grid(k, s, e)


def fingerprint(bars: dict[str, Bars]) -> str:
    hsh = hashlib.sha256()
    for sym in sorted(bars):
        b = bars[sym]
        hsh.update(sym.encode())
        for a in (b.t, b.o, b.h, b.l, b.c):
            hsh.update(np.ascontiguousarray(np.nan_to_num(a, nan=-1.0)).tobytes())
    return hsh.hexdigest()
