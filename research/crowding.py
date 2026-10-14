"""
Crowding study: signal construction, eligibility, portfolio weights, the
rebalance schedule and the decision rules, exactly as pre-registered in
research/PREREG_FUNDING.md. The pre-registration was written before any
result for these signals existed.

Nothing here runs a backtest. The runner is written only after the
pre-registration has been approved, and it must use these functions unchanged.

Timing (PREREG_FUNDING.md, section 3):
  8h grid   labels t at 00:00 / 08:00 / 16:00 UTC; label t = window (t-8h, t]
  funding   F[t] = sum of settlements in (t-8h, t]  (research.study.funding_buckets)
  premium   P[t] = close of the 8h premium-index bar (t-8h, t]  (research.data.load_premium)
  signal    S[t] uses F and P at labels <= t only
  fill      the weights decided at t sit on the 1h backtest row t + 1h: filled
            at close(t + 1h), first earning the 1h bar (t + 1h, t + 2h].
            Nothing trades between rebalances (research.backtest.run trade_at).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from research import backtest as bt
from research import data as D
from research import strategies as S
from research.data import utc
from research.study import CANDIDATE, CONFIRM, DATA_START, funding_buckets, spot_tradable

END = utc(2026, 10, 7)                  # hard cap: the cache keeps growing; nothing after END is used
OOS_SPLIT = utc(2026, 1, 1)             # OOS halves for the both-halves alpha rule
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
STEP = pd.Timedelta("8h")
LAG = pd.Timedelta("1h")                # decision at t -> fill at the close of the next 1h bar
MIN_ELIGIBLE = 10
BAND = 0.5                              # no-trade band (research.backtest.run), the only value used
SCORE_DECIMALS = 12                     # scores rounded before ranking so float noise cannot break ties
BORROW_STRESS_DAILY = 0.0010            # 0.10%/day on short notional (fees and slippage at base)
COSTS = {                               # every config is run under all three
    "base": bt.Costs(),                 # 0.10% fee + 0.08% slippage per side, borrow 0.03%/day
    "2x": bt.Costs().scaled(2.0),       # 0.20% + 0.16%, borrow 0.06%/day
    "borrow": bt.Costs(borrow_daily=BORROW_STRESS_DAILY),
}
BARS_PER_DAY, BARS_PER_YEAR = 24, 8760

SIGNALS = ("fund_lvl", "fund_chg", "prem_lvl", "prem_chg")
BOOKS = ("so", "ls")                    # short-only; long/short
KS = (3, 5)
PERIODS = ("8h", "24h", "72h")

# New rules on top of research.study's CANDIDATE / CONFIRM (kept unchanged).
PASS = {
    "oos_alpha_yr": 0.10,               # OOS alpha vs the EW universe >= 10%/yr, net of base costs
    "half_alpha": 0.0,                  # alpha > 0 in each OOS half
    "borrow_sharpe": 0.5,               # at 0.10%/day borrow: OOS Sharpe >= 0.5 ...
    "borrow_alpha": 0.0,                # ... and OOS alpha > 0
    "neighbour_oos_alpha": 0.0,         # the stable neighbour also has OOS alpha > 0
    "max_skipped_oos": 0.05,            # at most 5% of OOS rebalances skipped
    "drop_one_alpha": 0.0,              # OOS alpha > 0 with any single symbol removed
}


# ── the grid ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    signal: str
    book: str
    k: int
    period: str

    @property
    def name(self) -> str:
        return f"{self.signal}_{self.book}_k{self.k}_{self.period}"


def grid() -> list[Config]:
    """The complete pre-registered grid: 4 signals x 2 books x 2 k x 3 periods = 48."""
    return [Config(s, b, k, p) for s in SIGNALS for b in BOOKS for k in KS for p in PERIODS]


def neighbours(c: Config) -> list[Config]:
    """Same signal and book, one step away in k or in the rebalance period."""
    out = []
    for vals, field in ((KS, "k"), (PERIODS, "period")):
        i = vals.index(getattr(c, field))
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append(replace(c, **{field: vals[j]}))
    return out


# ── signals ──────────────────────────────────────────────────────────────────

def grid_8h(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every 8h label from start to end (epoch-anchored: 00:00, 08:00, 16:00 UTC)."""
    return pd.date_range(start.ceil(STEP), end.floor(STEP), freq=STEP)


def on_grid(x: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Put an 8h-labelled frame on the full grid. Labels without data become
    NaN; a gap is never filled. Off-grid labels are a bug, so they raise."""
    if not x.index.isin(index).all():
        raise ValueError(f"labels off the 8h grid: {x.index[~x.index.isin(index)][:3].tolist()}")
    return x.reindex(index)


def trailing_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """M_n[t] = mean of x over the n labels t-(n-1)*8h .. t (t included), NaN
    unless all n are present. x must be on the full 8h grid (see on_grid)."""
    return x.rolling(n, min_periods=n).mean()


def signals(F: pd.DataFrame, P: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The four crowding scores; higher = more crowded long = short candidate."""
    return {
        "fund_lvl": trailing_mean(F, 9),                          # 3-day mean funding
        "fund_chg": trailing_mean(F, 3) - trailing_mean(F, 21),   # last day vs last week
        "prem_lvl": trailing_mean(P, 9),                          # 3-day mean premium
        "prem_chg": trailing_mean(P, 3) - trailing_mean(P, 21),
    }


def eligible(score: pd.DataFrame, spot_ok: pd.DataFrame, liquid: pd.DataFrame) -> pd.DataFrame:
    """In the ranking at t: a complete score, the spot pair traded in (t-8h, t]
    (research.study.spot_tradable) and the 1h universe filter of research.study
    (strategies.tradable: listed >= 30 days, 7-day ADV >= $2M, close at t)."""
    idx, cols = score.index, score.columns
    return (score.notna()
            & spot_ok.reindex(index=idx, columns=cols).eq(True)
            & liquid.reindex(index=idx, columns=cols).eq(True))


# ── portfolio construction ───────────────────────────────────────────────────

def _slots(x: np.ndarray, k: int) -> np.ndarray:
    """Share of a slot per name when taking the k LARGEST values of x: names
    strictly above the k-th largest value get a full slot, names tied with it
    share the slots left over equally. Sums to k (needs len(x) >= k)."""
    cut = np.sort(x)[::-1][k - 1]
    above, tied = x > cut, x == cut
    out = above.astype(float)
    out[tied] = (k - above.sum()) / tied.sum()
    return out


def book_weights(scores: pd.Series, k: int, book: str) -> pd.Series:
    """Target weights for one rebalance from the eligible names' scores.

    so: short the top k at -1/k each (gross 1, net -1).
    ls: short the top k at -0.5/k, long the bottom k at +0.5/k (gross 1, net 0);
        a name in both tie blocks nets out.
    """
    x = np.round(scores.to_numpy(float), SCORE_DECIMALS)
    if book == "so":
        w = -_slots(x, k) / k
    elif book == "ls":
        w = 0.5 / k * (_slots(-x, k) - _slots(x, k))
    else:
        raise ValueError(f"unknown book {book!r}")
    return pd.Series(w, index=scores.index)


def decision_times(index: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    """Scheduled decision labels on the 8h grid: 8h = every label; 24h = 00:00
    UTC; 72h = 00:00 UTC on days whose number since 1970-01-01 is divisible by 3."""
    if period == "8h":
        keep = np.ones(len(index), bool)
    elif period == "24h":
        keep = index.hour == 0
    elif period == "72h":
        keep = (index.hour == 0) & ((index - EPOCH).days % 3 == 0)
    else:
        raise ValueError(f"unknown period {period!r}")
    return index[keep]


def targets(score: pd.DataFrame, elig: pd.DataFrame, k: int, book: str, period: str,
            index_1h: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Target weights on the 1h backtest grid, for research.backtest.run(..., trade_at=).

    Returns (W, trade_at, log). W holds each executed rebalance's targets on row
    t + LAG (zero elsewhere; the backtester ignores rows where trade_at is False);
    trade_at is True on exactly those rows. log has one row per scheduled
    decision time whose fill row exists: n_eligible and skipped. A rebalance with
    fewer than MIN_ELIGIBLE names is skipped: nothing trades, the book keeps its
    drifted positions (flat if it has none yet).
    """
    cols = score.columns
    Wn = np.zeros((len(index_1h), len(cols)))
    ta = np.zeros(len(index_1h), bool)
    rows = []
    for t in decision_times(score.index, period):
        pos = index_1h.get_indexer([t + LAG])[0]
        if pos < 0:
            continue                                # fill bar outside the data
        ok = elig.loc[t].to_numpy(bool)
        n = int(ok.sum())
        rows.append((t, n, n < MIN_ELIGIBLE))
        if n < MIN_ELIGIBLE:
            continue
        Wn[pos, ok] = book_weights(score.loc[t, ok], k, book).to_numpy()
        ta[pos] = True
    log = pd.DataFrame(rows, columns=["t", "n_eligible", "skipped"]).set_index("t")
    return (pd.DataFrame(Wn, index=index_1h, columns=cols), pd.Series(ta, index=index_1h), log)


# ── inputs ───────────────────────────────────────────────────────────────────

def load_inputs(symbols) -> dict:
    """Every input on its grid, truncated to END, from the research cache only.

    Refuses to run without RESEARCH_OFFLINE=1: the study must see exactly the
    cached data, with no requests and no cache writes.
    """
    if not D.OFFLINE:
        raise RuntimeError("set RESEARCH_OFFLINE=1: the crowding study reads the cache only")
    p1h = D.build_panel(symbols, "1h", DATA_START)
    p1h = {k: v[v.index <= END] for k, v in p1h.items()}
    g = grid_8h(DATA_START, END)
    fund, prem = {}, {}
    for s in symbols:
        f = D.load_funding(s, DATA_START)
        if len(f):
            fund[s] = f
        p = D.load_premium(s, DATA_START)
        if len(p):
            prem[s] = p
    cols = list(p1h["close"].columns)
    F = funding_buckets(pd.DataFrame(fund).sort_index())
    F = on_grid(F[F.index <= END], g).reindex(columns=cols)
    P = pd.DataFrame(prem).sort_index()
    P = on_grid(P[P.index <= END], g).reindex(columns=cols)
    liquid_1h = S.tradable(p1h, BARS_PER_DAY)
    bench = (p1h["close"].pct_change(fill_method=None)
             .where(liquid_1h.shift(1, fill_value=False)).mean(axis=1).fillna(0.0))
    return {"p1h": p1h, "F": F, "P": P,
            "spot_ok": spot_tradable(p1h["close"]).reindex(index=g, columns=cols),
            "liquid": liquid_1h.reindex(index=g, columns=cols),
            "bench": bench}


def fingerprint(inputs: dict) -> str:
    """sha256 over the 1h closes, funding and premium frames; printed with the
    results so an independent re-run can show it used identical data."""
    h = hashlib.sha256()
    for df in (inputs["p1h"]["close"], inputs["F"], inputs["P"]):
        h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
        h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()


# ── decision rules ───────────────────────────────────────────────────────────

def _ge(m: dict, key: str, th: float) -> bool:
    v = m.get(key)
    return v is not None and bool(np.isfinite(v)) and v >= th


def _gt(m: dict, key: str, th: float) -> bool:
    v = m.get(key)
    return v is not None and bool(np.isfinite(v)) and v > th


def verdict(rows: dict[str, dict]) -> dict[str, dict]:
    """Apply the pre-registered rules to every config in grid().

    rows[name] holds research.backtest.metrics dicts "is", "oos", "oos_2x",
    "oos_borrow", "oos_h1", "oos_h2", the float "skipped_oos" (share of OOS
    rebalances skipped) and, only once every other check has passed,
    "drop_one_min_alpha" (lowest OOS alpha_yr over the re-runs with one symbol
    removed). Returns {name: {"verdict": str, "failed": [check names]}}.
    """
    out = {}
    for c in grid():
        r = rows[c.name]
        i, o = r["is"], r["oos"]
        cand = {"is_sharpe": _ge(i, "sharpe", CANDIDATE["is_sharpe"]),
                "is_alpha_t": _ge(i, "alpha_t", CANDIDATE["is_alpha_t"])}
        nb = [rows[n.name] for n in neighbours(c)]
        checks = {
            "oos_sharpe": _ge(o, "sharpe", CONFIRM["oos_sharpe"]),
            "oos_alpha_pos": _gt(o, "alpha_yr", CONFIRM["oos_alpha"]),
            "oos_sharpe_2x": _ge(r["oos_2x"], "sharpe", CONFIRM["oos_sharpe_2x"]),
            "neighbour": any(_ge(x["is"], "sharpe", CONFIRM["neighbour_is_sharpe"])
                             and _gt(x["oos"], "alpha_yr", PASS["neighbour_oos_alpha"]) for x in nb),
            "oos_alpha_min": _ge(o, "alpha_yr", PASS["oos_alpha_yr"]),
            "oos_h1_alpha": _gt(r["oos_h1"], "alpha_yr", PASS["half_alpha"]),
            "oos_h2_alpha": _gt(r["oos_h2"], "alpha_yr", PASS["half_alpha"]),
            "borrow_sharpe": _ge(r["oos_borrow"], "sharpe", PASS["borrow_sharpe"]),
            "borrow_alpha": _gt(r["oos_borrow"], "alpha_yr", PASS["borrow_alpha"]),
            "coverage": r["skipped_oos"] <= PASS["max_skipped_oos"],
        }
        if all(checks.values()):                    # drop-one re-runs only for otherwise-passing configs
            checks["drop_one"] = _gt(r, "drop_one_min_alpha", PASS["drop_one_alpha"])
        failed = [k for k, ok in {**cand, **checks}.items() if not ok]
        if not all(cand.values()):
            v = "rejected"
        elif not all(checks.values()):
            v = "candidate (failed OOS)"
        else:
            v = "PASSED (candidate only; needs independent re-run)"
        out[c.name] = {"verdict": v, "failed": failed}
    return out
