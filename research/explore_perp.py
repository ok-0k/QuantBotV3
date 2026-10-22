"""
EXPLORATORY — NOT EVIDENCE. Perp-cost sensitivity re-score of past studies.

Every out-of-sample period used here has already been seen (RESULTS.md,
RESULTS_FUNDING.md, RESULTS_SMC.md). Nothing here confirms or rejects
anything. It only asks whether any idea deserves a NEW pre-registered forward
test if it were traded on Binance USDT-M perpetuals instead of spot.

Cost levels (fixed before any re-scored number was computed):
  zero     no fee, no slippage, no borrow, no funding: gross price P&L
  perp     taker 0.05% per fill; maker 0.02% per fill only where the
           original model used a limit fill (SMC entries and TP1/TP2 exits);
           slippage 0.02% per market-type fill (an assumption for liquid
           perps); no borrow; funding paid or received at the actual
           settlement rates for every settlement a position spans
  perp_s5  the same with 0.05% slippage
Carry is a long-spot / short-perp hedge, so its spot leg keeps the spot cost
(0.10% + 0.08%); only its perp leg moves to 0.05% taker + 0.02% (0.05%)
slippage: a switch costs 0.25% (0.28%) of notional instead of 0.31%. Its
"zero" level has no switch cost; funding is its income at every level.

Universe: each study's own universe, restricted to symbol-times with a listed
perp, [first funding settlement - 8h, last settlement] in the cache. All 50
symbols have a perp at some point; the gaps are FET before 2023-01-16, STX
before 2023-02-21, ILV before 2023-11-10 and MKR after 2025-09-08. Prices stay
Binance SPOT (perp basis ignored). The crowding books need funding/premium to
rank a name at all, so their eligibility is used unchanged.

What is reused unchanged: every signal, weight, schedule, eligibility rule,
order simulation and walk of the committed studies. What is new: (1) run_perp,
a line-for-line copy of research.backtest.run plus a funding term and
sign-change counters (checked equal to backtest.run with funding off);
(2) perp_net_r, research.smc.net_r with maker/taker fees and funding instead
of a single fee and borrow. Each study is first reproduced at its original
base cost on its original universe and checked against the stored run.

    TRADING_ENV_FILE=/nonexistent DISCORD_WEBHOOK_URL= RESEARCH_OFFLINE=1 \\
    TRADING_DATA_DIR=<scratch>/td TRADING_LOG_DIR=<scratch>/tl \\
    nice -n 10 .venv/bin/python -m research.explore_perp {crowding,price,smc,diag,report} --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research import backtest as bt
from research import data as D

LABEL = "EXPLORATORY — NOT EVIDENCE"
REPO = Path(__file__).resolve().parent.parent
TAKER, MAKER, SLIP, SLIP5 = 0.0005, 0.0002, 0.0002, 0.0005
LEVELS = {"zero": (0.0, False), "perp": (TAKER + SLIP, True), "perp_s5": (TAKER + SLIP5, True)}
CARRY_SWITCH = {"zero": 0.0, "perp": 0.0018 + TAKER + SLIP, "perp_s5": 0.0018 + TAKER + SLIP5}
WEEKS8 = 56.0                                    # days in the screen's 8-week window
MIN_EVENTS_8W = 30                               # PM screen (b)
CROWD_RUN = REPO / "research" / "runs" / "crowding_run1_20261014T035310Z"
SMC_RUN = REPO / "research" / "runs" / "smc_run1_20261020T060848Z"
PRICE_RUNS = {0.0: D.ROOT / "results" / "20261011T001031", 0.5: D.ROOT / "results" / "20261011T001040"}
PRICE_END = pd.Timestamp("2026-10-07 05:00", tz="UTC")       # last 1h bar of the stored runs
CROWD_FP = "4a7bd3e083fc984323a8ad0c444b73f280ca472135ead750cf9b5f9dc84042e0"
TOL = 1e-9


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def _close(a, b, what: str) -> None:
    a = float("nan") if a is None else float(a)
    b = float("nan") if b is None else float(b)
    if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= TOL * max(1.0, abs(b))):
        raise AssertionError(f"reproduction failed: {what}: {a!r} != stored {b!r}")


# ── perp inputs ─────────────────────────────────────────────────────────────

def load_fund(symbols, end: pd.Timestamp) -> dict[str, pd.Series]:
    """Raw funding settlements per symbol (rate per settlement), cache only, up to end."""
    out = {}
    for s in symbols:
        f = D.load_funding(s, pd.Timestamp("2022-10-01", tz="UTC"))
        f = f[f.index <= end]
        if len(f):
            out[s] = f
    return out


def perp_listed(fund: dict, index: pd.DatetimeIndex, cols) -> pd.DataFrame:
    """True where a perp is listed: [first settlement - 8h, last settlement]."""
    out = pd.DataFrame(False, index=index, columns=cols)
    for s in cols:
        f = fund.get(s)
        if f is not None and len(f):
            out[s] = (index >= f.index.min() - pd.Timedelta("8h")) & (index <= f.index.max())
    return out


def funding_on_grid(fund: dict, index: pd.DatetimeIndex, cols) -> np.ndarray:
    """Funding settled in each bar (t - step, t] on a close-time grid, by
    symbol (0 where none). A position held over bar t pays w x rate."""
    step = index[1] - index[0]
    out = pd.DataFrame(0.0, index=index, columns=cols)
    for s in cols:
        f = fund.get(s)
        if f is None or not len(f):
            continue
        lab = f.index.round("1h").ceil(step)
        out[s] = f.groupby(lab).sum().reindex(index).fillna(0.0).to_numpy()
    return out.to_numpy(float)


# ── backtester: research.backtest.run + funding ─────────────────────────────

def run_perp(weights: pd.DataFrame, close: pd.DataFrame, bars_per_day: float, side: float,
             borrow_daily: float = 0.0, band: float = 0.0, trade_at: pd.Series | None = None,
             fund: np.ndarray | None = None) -> tuple[bt.Result, dict]:
    """research.backtest.run, copied line for line, plus (1) funding: the
    position held over bar t+1 earns -target @ fund[t+1] (a long pays a
    positive rate, a short receives it); (2) per-bar counters: `events`
    (some position opens, closes or flips: a signal-driven rebalance; drift
    trades are not counted) and `entries` (positions opened or flipped).
    With fund=None and the same costs it reproduces backtest.run exactly."""
    close = close.sort_index()
    W = weights.reindex(index=close.index, columns=close.columns).fillna(0.0).to_numpy(float)
    TA = None if trade_at is None else trade_at.reindex(close.index, fill_value=False).to_numpy(bool)
    R = close.pct_change(fill_method=None).to_numpy(float)
    R = np.where(np.isfinite(R), R, 0.0)
    F = np.zeros_like(R) if fund is None else fund
    T, N = W.shape
    net = np.zeros(T); gross = np.zeros(T); cost = np.zeros(T); borrow = np.zeros(T)
    turn = np.zeros(T); gexp = np.zeros(T); nexp = np.zeros(T)
    fu = np.zeros(T); ev = np.zeros(T); en = np.zeros(T)
    held = np.zeros(N)
    borrow_bar = borrow_daily / bars_per_day
    for t in range(T - 1):
        target = W[t]
        if TA is not None and not TA[t]:
            target = held
        elif band > 0:
            keep = (np.sign(held) == np.sign(target)) & (target != 0) & \
                   (np.abs(held - target) <= band * np.abs(target))
            target = np.where(keep, held, target)
        traded = np.abs(target - held).sum()
        c = traded * side
        r = R[t + 1]
        g = float(target @ r)
        b = float(np.clip(-target, 0.0, None).sum()) * borrow_bar
        f = -float(target @ F[t + 1])
        p = g - c - b + f
        net[t + 1], gross[t + 1], cost[t + 1], borrow[t + 1] = p, g, c, b
        turn[t + 1] = traded
        gexp[t + 1] = np.abs(target).sum()
        nexp[t + 1] = target.sum()
        fu[t + 1] = f
        chg = np.sign(target) != np.sign(held)
        ev[t + 1] = float(chg.any())
        en[t + 1] = float((chg & (target != 0)).sum())
        if 1.0 + p <= 0.0:
            net[t + 1] = -1.0
            net[t + 2:] = 0.0
            break
        held = target * (1.0 + r) / (1.0 + p)
    idx = close.index
    mk = lambda a: pd.Series(a, index=idx)
    return (bt.Result(mk(net), mk(gross), mk(cost), mk(borrow), mk(turn), mk(gexp), mk(nexp)),
            {"funding": mk(fu), "events": mk(ev), "entries": mk(en)})


def summ(res: bt.Result, ex: dict, bench: pd.Series, a, b, bpy: float) -> dict:
    """Window stats: alpha vs the study's own benchmark (backtest.metrics),
    plus funding and trading cost a year and events / entries per 8 weeks."""
    w = res.window(a, b)
    m = bt.metrics(w, bpy, bench[a:b])
    yrs = len(w.net) / bpy
    per8 = WEEKS8 / (yrs * 365.0)
    return {"alpha_yr": m.get("alpha_yr"), "alpha_t": m.get("alpha_t"), "sharpe": m.get("sharpe"),
            "cagr": m.get("cagr"), "gross_yr": m.get("gross_return_yr"),
            "cost_yr": float((w.cost.sum() + w.borrow.sum()) / yrs),
            "funding_yr": float(ex["funding"][a:b].sum() / yrs) if ex else 0.0,
            "turnover_yr": m.get("turnover_yr"),
            "events_8w": float(ex["events"][a:b].sum() * per8) if ex else None,
            "entries_8w": float(ex["entries"][a:b].sum() * per8) if ex else None}


# ── crowding (48 configs) ───────────────────────────────────────────────────

def crowding(out: Path) -> None:
    from config import SYMBOLS
    from research import crowding as C
    from research import crowding_study as CS
    t0 = time.time()
    inputs = C.load_inputs(SYMBOLS)
    fp = C.fingerprint(inputs)
    assert fp == CROWD_FP, f"data fingerprint {fp} != run of record {CROWD_FP}"
    close, bench = inputs["p1h"]["close"], inputs["bench"]
    sig, elig = CS.books(inputs)
    fund = load_fund(close.columns, C.END)
    F = funding_on_grid(fund, close.index, close.columns)
    stored = json.loads((CROWD_RUN / "summary.json").read_text())["configs"]
    win = {k: v for k, v in CS.windows().items() if k in ("is", "oos")}
    rows = {}
    for i, c in enumerate(C.grid()):
        W, ta, log = C.targets(sig[c.signal], elig[c.signal], c.k, c.book, c.period, close.index)
        if i == 0:                                   # the copy == backtest.run at base cost, no funding
            base = bt.run(W, close, C.BARS_PER_DAY, C.COSTS["base"], band=C.BAND, trade_at=ta)
            cp, _ = run_perp(W, close, C.BARS_PER_DAY, C.COSTS["base"].side, C.COSTS["base"].borrow_daily,
                             C.BAND, ta)
            assert np.array_equal(base.net.to_numpy(), cp.net.to_numpy()), "run_perp != backtest.run"
            for k in ("is", "oos"):
                _close(CS._metrics(base, bench, k)["alpha_yr"], stored[c.name][k]["alpha_yr"], f"{c.name} {k}")
        lv = {}
        for name, (side, use_f) in LEVELS.items():
            res, ex = run_perp(W, close, C.BARS_PER_DAY, side, 0.0, C.BAND, ta, F if use_f else None)
            lv[name] = {k: summ(res, ex, bench, a, b, C.BARS_PER_YEAR) for k, (a, b) in win.items()}
        _close(lv["zero"]["oos"]["alpha_yr"], stored[c.name]["oos_zero"]["alpha_yr"], f"{c.name} oos zero")
        cov = CS.coverage(log)
        yrs_oos = len(close.loc[win["oos"][0]:win["oos"][1]]) / C.BARS_PER_YEAR
        rows[c.name] = {"study": "crowding", "levels": lv,
                        "decisions_8w": cov["oos_executed"] * WEEKS8 / (yrs_oos * 365.0),
                        "spot_base_oos_alpha": stored[c.name]["oos"]["alpha_yr"]}
        _log(f"  {c.name:22} zero IS/OOS a {lv['zero']['is']['alpha_yr']:+.3f}/{lv['zero']['oos']['alpha_yr']:+.3f}"
             f"  perp OOS a {lv['perp']['oos']['alpha_yr']:+.3f}  ({time.time() - t0:.0f}s)")
    meta = {"label": LABEL, "fingerprint": fp, "end": str(C.END), "runtime_s": time.time() - t0,
            "checks": "run_perp == backtest.run (base, no funding); base IS/OOS alpha and zero-cost OOS alpha "
                      "of all 48 equal the run of record"}
    (out / "crowding.json").write_text(json.dumps(_clean({"meta": meta, "rows": rows}), indent=1))


# ── price strategies (17) + carry (2) from the 2026-10-07 study ─────────────

def price(out: Path) -> None:
    from config import SYMBOLS
    from research import strategies as S
    from research import study as ST
    t0 = time.time()
    p1h = D.build_panel(SYMBOLS, "1h", ST.DATA_START)
    p1h = {k: v[v.index <= PRICE_END] for k, v in p1h.items()}
    panels = {"1h": p1h, "4h": D.resample(p1h, "4h"), "1D": D.resample(p1h, "1D")}
    masks = {f: S.tradable(panels[f], ST.FREQ[f][0]) for f in panels}
    bench = {}
    for f, p in panels.items():
        prev = masks[f].shift(1, fill_value=False)
        bench[f] = p["close"].pct_change(fill_method=None).where(prev).mean(axis=1).fillna(0.0)
    cols = list(p1h["close"].columns)
    fund = load_fund(cols, PRICE_END)
    perp_ok = {f: perp_listed(fund, panels[f]["close"].index, cols) for f in panels}
    Fg = {f: funding_on_grid(fund, panels[f]["close"].index, cols) for f in panels}
    stored = {band: json.loads((d / "summary.json").read_text())["metrics"] for band, d in PRICE_RUNS.items()}
    NS = pd.Timedelta("1ns")
    win = {"is": (ST.IS_START, ST.OOS_START - NS), "oos": (ST.OOS_START, None)}
    rows = {}
    for sp in ST.specs():
        if sp.family == "carry":
            continue
        bpd, bpy = ST.FREQ[sp.freq]
        p, m = panels[sp.freq], masks[sp.freq]
        w_spot, w_perp = sp.make(p, m), sp.make(p, m & perp_ok[sp.freq])
        for band in (0.0, 0.5):
            base = ST._split(bt.run(w_spot, p["close"], bpd, band=band), bpy, bench[sp.freq])
            for k in ("is", "oos"):
                for fld in ("sharpe", "alpha_yr"):
                    _close(base[k].get(fld), stored[band][sp.name][k].get(fld), f"{sp.name} band {band} {k} {fld}")
            lv = {}
            for name, (side, use_f) in LEVELS.items():
                res, ex = run_perp(w_perp, p["close"], bpd, side, 0.0, band, None, Fg[sp.freq] if use_f else None)
                lv[name] = {k: summ(res, ex, bench[sp.freq], a, b, bpy) for k, (a, b) in win.items()}
            key = f"{sp.name}@band{band:g}"
            rows[key] = {"study": "price", "levels": lv, "spot_base_oos_alpha": base["oos"].get("alpha_yr")}
            _log(f"  {key:22} zero IS/OOS a {lv['zero']['is']['alpha_yr']:+.3f}/{lv['zero']['oos']['alpha_yr']:+.3f}"
                 f"  perp OOS a {lv['perp']['oos']['alpha_yr']:+.3f}  ({time.time() - t0:.0f}s)")

    # carry: same sleeve model; only the switch cost changes (spot leg at spot cost, perp leg at perp cost)
    # the stored carry runs saw settlements up to the last spot 8h window (2026-10-07 08:00), so the
    # cap here is the study's own: buckets <= spot_ok.index.max(), as in research.study.main
    raw = pd.DataFrame(load_fund(cols, PRICE_END + pd.Timedelta("1D"))).sort_index()
    funding = ST.funding_buckets(raw)
    spot_ok = ST.spot_tradable(p1h["close"])
    funding = funding[funding.index <= spot_ok.index.max()]
    bench["8h"] = bench["1h"].pipe(lambda r: (1 + r).resample("8h", closed="right", label="right").prod() - 1)
    bpy = ST.FREQ["8h"][1]
    for k, th in ((1, 0.0001), (2, 0.0002)):
        name = f"carry_{k}"
        base = ST._split(ST._carry_result(funding, th, tradable=spot_ok), bpy, bench["8h"])
        for per in ("is", "oos"):
            for fld in ("sharpe", "alpha_yr"):
                _close(base[per].get(fld), stored[0.0][name][per].get(fld), f"{name} {per} {fld}")
        lv = {}
        for lvl, sc in CARRY_SWITCH.items():
            sleeve, on = S.funding_carry(funding, th, switch_cost=sc, tradable=spot_ok, return_state=True)
            port = sleeve.mean(axis=1).fillna(0.0).shift(1).fillna(0.0)      # as study._carry_result
            z = pd.Series(0.0, index=port.index)
            res = bt.Result(net=port, gross=port, cost=z, borrow=z, turnover=z, gross_exposure=z, net_exposure=z)
            d = on.diff()
            d.iloc[0] = on.iloc[0]
            ex = {"funding": z, "events": (d != 0).any(axis=1).astype(float).shift(1).fillna(0.0),
                  "entries": (d > 0).sum(axis=1).astype(float).shift(1).fillna(0.0)}
            lv[lvl] = {per: summ(res, ex, bench["8h"], a, b, bpy) for per, (a, b) in win.items()}
            for per, (a, b) in win.items():
                lv[lvl][per].update(cost_yr=None, funding_yr=None,       # both are inside the sleeve return
                                    hedged_periods=int(on.loc[a:b].to_numpy().sum()))
        rows[name] = {"study": "carry", "levels": lv, "spot_base_oos_alpha": base["oos"].get("alpha_yr")}
        _log(f"  {name:22} zero IS/OOS a {lv['zero']['is']['alpha_yr']:+.4f}/{lv['zero']['oos']['alpha_yr']:+.4f}")
    meta = {"label": LABEL, "end": str(PRICE_END), "runtime_s": time.time() - t0,
            "checks": "base-cost IS/OOS Sharpe and alpha of all 17 strategies x 2 bands and both carry "
                      "configs equal the stored 2026-10-11 runs"}
    (out / "price.json").write_text(json.dumps(_clean({"meta": meta, "rows": rows}), indent=1))


# ── SMC (8 configs x long/short/both) ───────────────────────────────────────

@dataclass(frozen=True)
class PerpCost:
    taker: float
    maker: float
    slip: float
    funding: bool = True


_FUND_MS: dict[str, tuple[np.ndarray, np.ndarray]] = {}          # sym -> (settlement ms, cumulative rate)


def _funding_between(sym: str, t0: int, t1: int) -> float:
    """Sum of funding rates settled strictly inside (t0, t1) ms."""
    ts, cum = _FUND_MS.get(sym, (np.zeros(0, np.int64), np.zeros(1)))
    i, j = np.searchsorted(ts, t0, side="right"), np.searchsorted(ts, t1, side="left")
    return float(cum[j] - cum[i]) if j > i else 0.0


def perp_net_r(cand, oc, cost: PerpCost) -> float:
    """research.smc.net_r with perp costs: maker on the limit entry and on
    TP1/TP2 limit exits, taker + slippage on market-type exits (stops,
    breakeven, time and session exits), no borrow, and funding on the entry
    notional for every settlement strictly between the fill-bar open and the
    exit time (full size, also after a pbe partial: a small overstatement)."""
    s = cand.side
    L, S_ = s * cand.limit, s * cand.stop
    q = 1.0 / abs(L - S_)
    pnl = 0.0
    fees = cost.maker * q * L
    for frac, x_long, market in oc.exits:
        x = s * x_long
        if market:
            x = x * (1.0 - s * cost.slip)
        pnl += frac * s * (x - L) * q
        fees += (cost.taker if market else cost.maker) * q * frac * x
    fund = -s * _funding_between(cand.sym, oc.t_fill, oc.t_end) * q * L if cost.funding else 0.0
    return pnl - fees + fund


SMC_PERP = {"perp": PerpCost(TAKER, MAKER, SLIP), "perp_s5": PerpCost(TAKER, MAKER, SLIP5),
            "perp_nofund": PerpCost(TAKER, MAKER, SLIP, funding=False)}


def smc_run(out: Path) -> None:
    from research import smc
    from research import smc_study as SS
    t0 = time.time()
    orig = smc.net_r
    smc.net_r = lambda c, oc, cost: perp_net_r(c, oc, cost) if isinstance(cost, PerpCost) else orig(c, oc, cost)
    smc.COSTS = {**smc.COSTS, **SMC_PERP}
    syms = SS.universe()
    fund = load_fund(syms, smc.END)
    for s, f in fund.items():
        ms = ((f.index.round("1min") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms")).to_numpy(np.int64)
        _FUND_MS[s] = (ms, np.r_[0.0, np.cumsum(f.to_numpy(float))])
    span = {s: (int((f.index.min() - pd.Timedelta("8h")).value // 1_000_000), int(f.index.max().value // 1_000_000))
            for s, f in fund.items()}
    days = smc.trading_days()
    preps, refs = {}, {}
    for sym in SS.REFS:
        if sym in syms:
            preps[sym] = smc.prepare(sym, smc.load_symbol(sym), days)
            refs[sym] = smc.ref_taken_times(preps[sym])
    cands, ctxs = [], {}
    for i, sym in enumerate(syms, 1):
        prep = preps.pop(sym, None) or smc.prepare(sym, smc.load_symbol(sym), days)
        ref = refs.get("ETHUSDT") if sym == "BTCUSDT" else refs.get("BTCUSDT")
        r = SS.symbol_pass(prep, ref)
        cands += r["cands"]
        ctxs[sym] = r["ctx"]
        del prep
        _log(f"{sym} done ({i}/{len(syms)}, {time.time() - t0:.0f}s)")
    in_perp = lambda c: c.sym in span and span[c.sym][0] <= c.t_place <= span[c.sym][1]
    cands_perp = [c for c in cands if in_perp(c)]
    order = {s: i for i, s in enumerate(syms)}
    stored = json.loads((SMC_RUN / "summary.json").read_text())["runs"]
    days_of = {"IS": (smc.OOS_START - smc.IS_START).days, "OOS": (smc.END_DAY - smc.OOS_START).days}
    rows = {}
    for cfg in smc.grid():
        for side_name, sides in smc.SIDES.items():
            base = smc.split(smc.walk(cands, ctxs, cfg, sides, "base", scope="symbol", sym_order=order))
            for per in ("IS", "OOS"):
                st = stored[cfg.name][side_name]["symbol"]["base"]["periods"][per]
                s = smc.summarize(base[per])
                _close(s["n"], st["n"], f"{cfg.name} {side_name} {per} n")
                _close(s["exp"], st["exp"], f"{cfg.name} {side_name} {per} exp")
            lv = {}
            for ck in ("zero", "perp", "perp_s5", "perp_nofund"):
                sp = smc.split(smc.walk(cands_perp, ctxs, cfg, sides, ck, scope="symbol", sym_order=order))
                lv[ck] = {}
                for per in ("IS", "OOS"):
                    s = smc.summarize(sp[per])
                    lv[ck][per.lower()] = {"n": s["n"], "exp": s["exp"], "t": s["t"], "exp_gross": s.get("exp_gross"),
                                           "cost_per_trade_r": s.get("cost_per_trade_r"),
                                           "trades_8w": s["n"] * WEEKS8 / days_of[per]}
            rows[f"{cfg.name}:{side_name}"] = {"study": "smc", "decisional": side_name in smc.DECISION_SIDES,
                                               "levels": lv, "dropped_no_perp": len(cands) - len(cands_perp)}
        _log(f"{cfg.name} walked ({time.time() - t0:.0f}s)")
    meta = {"label": LABEL, "end": str(smc.END), "runtime_s": time.time() - t0,
            "candidates": len(cands), "candidates_perp": len(cands_perp),
            "checks": "base-cost IS/OOS n and expectancy of all 24 config x side units (symbol scope) equal "
                      "the run of record"}
    (out / "smc.json").write_text(json.dumps(_clean({"meta": meta, "rows": rows}), indent=1))


# ── diagnostics for the top-ranked survivors (descriptive only) ─────────────

DIAG_CROWD = ("fund_lvl_ls_k3_24h", "fund_lvl_ls_k5_24h")
DIAG_DROP = ((), ("AXSUSDT",), ("ENJUSDT",), ("AXSUSDT", "ENJUSDT"))
DIAG_XS = (14, 28)


def diag(out: Path) -> None:
    """Where the perp P&L of the top-ranked survivors comes from: funding by
    symbol, the perp basis that the re-score ignores (premium change on the
    held weight, 8h resolution), and perp-cost re-runs without the symbols
    that carry most of it. Descriptive; nothing here is a screen."""
    from config import SYMBOLS
    from research import crowding as C
    from research import crowding_study as CS
    from research import strategies as S
    from research import study as ST
    t0 = time.time()
    side = TAKER + SLIP
    res_out: dict = {}
    inputs = C.load_inputs(SYMBOLS)
    close, bench = inputs["p1h"]["close"], inputs["bench"]
    fund = load_fund(close.columns, C.END)
    win = {k: v for k, v in CS.windows().items() if k in ("is", "oos")}
    for name in DIAG_CROWD:
        sg, book, k, per = name[:8], name.split("_")[2], int(name.split("_")[3][1:]), name.split("_")[4]
        d = {}
        for drop in DIAG_DROP:
            sub = {"p1h": {kk: v.drop(columns=list(drop)) for kk, v in inputs["p1h"].items()},
                   **{kk: inputs[kk].drop(columns=list(drop)) for kk in ("F", "P", "spot_ok", "liquid")}}
            sig = C.signals(sub["F"], sub["P"])
            el = C.eligible(sig[sg], sub["spot_ok"], sub["liquid"])
            cl = sub["p1h"]["close"]
            W, ta, _ = C.targets(sig[sg], el, k, book, per, cl.index)
            res, ex = run_perp(W, cl, C.BARS_PER_DAY, side, 0.0, C.BAND, ta, funding_on_grid(fund, cl.index, cl.columns))
            d["drop " + ("+".join(drop) or "none")] = {kk: summ(res, ex, bench, a, b, C.BARS_PER_YEAR)
                                                       for kk, (a, b) in win.items()}
            if not drop:                             # attribution on the full universe
                hp = CS.held_targets(W, ta).shift(1).fillna(0.0)
                fpnl = -(hp * funding_on_grid(fund, cl.index, cl.columns))
                h8 = hp.reindex(inputs["P"].index)
                bpnl = (h8 * inputs["P"].diff()).fillna(0.0)
                for kk, (a, b) in win.items():
                    yrs = len(cl.loc[a:b]) / C.BARS_PER_YEAR
                    fs = fpnl.loc[a:b].sum() / yrs
                    top = fs.sort_values(ascending=False).head(3)
                    d[f"{kk}_funding_top"] = {s: float(v) for s, v in top.items()}
                    d[f"{kk}_funding_total"] = float(fs.sum())
                    d[f"{kk}_basis_yr"] = float(bpnl.loc[a:b].sum().sum() / yrs)
                d["oos_funding_by_month"] = {str(i.date()): float(v) for i, v in
                                             fpnl.loc[win["oos"][0]:win["oos"][1]].sum(axis=1).resample("ME").sum().items()}
        res_out[name] = d
        _log(f"  diag {name} ({time.time() - t0:.0f}s)")
    p1h = D.build_panel(SYMBOLS, "1h", ST.DATA_START)
    p1h = {k: v[v.index <= PRICE_END] for k, v in p1h.items()}
    p = D.resample(p1h, "1D")
    m = S.tradable(p, 1)
    b1 = p["close"].pct_change(fill_method=None).where(m.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    fd = load_fund(list(p["close"].columns), PRICE_END)
    ok = perp_listed(fd, p["close"].index, p["close"].columns)
    NS = pd.Timedelta("1ns")
    w8 = {"is": (ST.IS_START, ST.OOS_START - NS), "oos": (ST.OOS_START, None)}
    R = p["close"].pct_change(fill_method=None).fillna(0.0)
    for L in DIAG_XS:
        w = S.xs_rank(p, m & ok, lookback=L, skip=1)
        contrib = (w.shift(1).fillna(0.0) * R).loc[ST.OOS_START:]
        by_sym = contrib.sum().sort_values(ascending=False)
        d = {"oos_price_top": {s: float(v) for s, v in by_sym.head(3).items()},
             "oos_price_by_quarter": {str(i.date()): float(v) for i, v in contrib.sum(axis=1).resample("QE").sum().items()}}
        for drop in ((), (by_sym.index[0],)):
            cols = [c for c in p["close"].columns if c not in drop]
            pp = {k: v[cols] for k, v in p.items()}
            ww = S.xs_rank(pp, (m & ok)[cols], lookback=L, skip=1)
            res, ex = run_perp(ww, pp["close"], 1, side, 0.0, 0.5, None, funding_on_grid(fd, pp["close"].index, cols))
            d["drop " + ("+".join(drop) or "none")] = {k: summ(res, ex, b1, a, z, 365) for k, (a, z) in w8.items()}
        res_out[f"xsmom_{L}@band0.5"] = d
    (out / "diag.json").write_text(json.dumps(_clean({"meta": {"label": LABEL, "runtime_s": time.time() - t0},
                                                      "rows": res_out}), indent=1))


# ── screen + report ─────────────────────────────────────────────────────────

def screen(row: dict) -> dict:
    """PM forward-candidate screen. (a) gross (zero-cost) alpha, or
    expectancy for SMC, > 0 in IS and in OOS; (b) at least 30 events per 8
    weeks at the OOS rate: SMC fills, or signal-driven rebalances (bars where
    a position opens, closes or flips) for the portfolio books. Perp OOS >= 0
    is reported, never decisive (seen data)."""
    lv = row["levels"]
    if row["study"] == "smc":
        g_is, g_oos = lv["zero"]["is"]["exp"], lv["zero"]["oos"]["exp"]
        p_oos, p5_oos = lv["perp"]["oos"]["exp"], lv["perp_s5"]["oos"]["exp"]
        n8 = lv["perp"]["oos"]["trades_8w"]
        loose = n8
    else:
        g_is, g_oos = lv["zero"]["is"]["alpha_yr"], lv["zero"]["oos"]["alpha_yr"]
        p_oos, p5_oos = lv["perp"]["oos"]["alpha_yr"], lv["perp_s5"]["oos"]["alpha_yr"]
        n8 = lv["perp"]["oos"]["events_8w"]
        loose = lv["perp"]["oos"]["entries_8w"]
    ok = lambda x: x is not None and math.isfinite(x)
    a = ok(g_is) and ok(g_oos) and g_is > 0 and g_oos > 0
    b = ok(n8) and n8 >= MIN_EVENTS_8W
    return {"g_is": g_is, "g_oos": g_oos, "p_oos": p_oos, "p5_oos": p5_oos, "n8": n8, "entries8": loose,
            "a": a, "b": b, "pass": a and b, "perp_nonneg": ok(p_oos) and p_oos >= 0}


def report(out: Path) -> None:
    rows = {}
    for f in ("crowding.json", "price.json", "smc.json"):
        rows.update(json.loads((out / f).read_text())["rows"])
    sc = {k: screen(v) for k, v in rows.items()}
    pct = lambda x: "—" if x is None else f"{x * 100:+.1f}%"
    rr = lambda x: "—" if x is None else f"{x:+.3f}"
    num = lambda x: "—" if x is None else f"{x:.1f}"
    L = [f"# {LABEL}", ""]
    groups = (("crowding", "Crowding (48 configs)"), ("price", "Price strategies (17 x band 0 / 0.5)"),
              ("carry", "Carry (2)"), ("smc", "SMC (8 configs x short / long / both; per-symbol scope)"))
    for g, title in groups:
        keys = [k for k in rows if rows[k]["study"] == g]
        L += [f"## {title}", ""]
        if g == "smc":
            L += ["| unit | IS n | OOS n | IS gross | OOS gross | IS perp | OOS perp | OOS perp s5 | funding R/trade OOS"
                  " | trades/8wk | (a) | (b) |", "|" + "---|" * 12]
            for k in keys:
                lv, s = rows[k]["levels"], sc[k]
                fr = (None if lv["perp"]["oos"]["exp"] is None else
                      lv["perp"]["oos"]["exp"] - lv["perp_nofund"]["oos"]["exp"])
                L.append(f"| `{k}` | {lv['zero']['is']['n']} | {lv['zero']['oos']['n']} | {rr(s['g_is'])} | "
                         f"{rr(s['g_oos'])} | {rr(lv['perp']['is']['exp'])} | {rr(s['p_oos'])} | {rr(s['p5_oos'])} | "
                         f"{rr(fr)} | {num(s['n8'])} | {'✓' if s['a'] else '✗'} | {'✓' if s['b'] else '✗'} |")
        else:
            L += ["| config | IS gross α | OOS gross α | IS perp α | OOS perp α | OOS perp α t | OOS perp s5 α | "
                  "OOS funding/yr | OOS cost/yr | events/8wk | entries/8wk | (a) | (b) |", "|" + "---|" * 13]
            for k in keys:
                lv, s = rows[k]["levels"], sc[k]
                po = lv["perp"]["oos"]
                L.append(f"| `{k}` | {pct(s['g_is'])} | {pct(s['g_oos'])} | {pct(lv['perp']['is']['alpha_yr'])} | "
                         f"{pct(s['p_oos'])} | {rr(po['alpha_t'])} | {pct(s['p5_oos'])} | {pct(po['funding_yr'])} | "
                         f"{pct(po['cost_yr'])} | {num(s['n8'])} | {num(s['entries8'])} | "
                         f"{'✓' if s['a'] else '✗'} | {'✓' if s['b'] else '✗'} |")
        L.append("")
    passed = sorted((k for k in sc if sc[k]["pass"]),
                    key=lambda k: -(rows[k]["levels"]["perp"]["oos"].get("alpha_t") or rows[k]["levels"]["perp"]["oos"].get("t") or -99))
    a_only = [k for k in sc if sc[k]["a"] and not sc[k]["b"]]
    L += ["## Screen", "", f"- screened: {len(sc)} rows ({sum(r['study'] == 'crowding' for r in rows.values())} crowding, "
          f"{sum(r['study'] == 'price' for r in rows.values())} price x band, "
          f"{sum(r['study'] == 'carry' for r in rows.values())} carry, "
          f"{sum(r['study'] == 'smc' for r in rows.values())} SMC units of which "
          f"{sum(r['study'] == 'smc' and r['decisional'] for r in rows.values())} short)",
          f"- pass (a) gross > 0 IS and OOS: {sum(s['a'] for s in sc.values())}",
          f"- pass (a) and (b) >= {MIN_EVENTS_8W} events / 8 weeks: {len(passed)}",
          f"- pass (a) but fail (b): {', '.join(a_only) or 'none'}", ""]
    for i, k in enumerate(passed, 1):
        s = sc[k]
        L.append(f"{i}. `{k}`: gross IS {s['g_is']:+.4f}, gross OOS {s['g_oos']:+.4f}, perp OOS {s['p_oos']:+.4f}, "
                 f"perp s5 OOS {s['p5_oos']:+.4f}, events/8wk {s['n8']:.1f}, perp OOS >= 0: {s['perp_nonneg']}")
    text = "\n".join(L)
    (out / "report.md").write_text(text + "\n")
    flat = []
    for k, v in rows.items():
        s = sc[k]
        flat.append({"config": k, "study": v["study"], **{f: s[f] for f in ("g_is", "g_oos", "p_oos", "p5_oos", "n8",
                                                                          "entries8", "a", "b", "pass", "perp_nonneg")}})
    pd.DataFrame(flat).to_csv(out / "screen.csv", index=False, float_format="%.6g")
    print(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="research.explore_perp")
    ap.add_argument("part", choices=("crowding", "price", "smc", "diag", "report"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if a.part != "report" and not D.OFFLINE:
        raise SystemExit("set RESEARCH_OFFLINE=1: this re-score reads the cache only")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"crowding": crowding, "price": price, "smc": smc_run, "diag": diag, "report": report}[a.part](a.out)
    _log(f"{a.part} done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
