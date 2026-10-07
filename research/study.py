"""
Pre-registered signal study (written and committed BEFORE any result was seen).

    .venv/bin/python -m research.study

Question: does any simple, well-known crypto signal beat realistic costs on
the live bot's universe, out of sample?

Universe     config.SYMBOLS (the live bot's 50, incl. 4 since-delisted pairs —
             their history up to delisting is kept, which slightly reduces
             survivorship bias; the list was still chosen in 2026, so some
             survivorship bias remains).
Data         Binance spot 1h klines from 2022-10-01 (warmup), resampled to
             4h / 1D; USDT-M perp funding rates for the carry candidates.
Periods      IS  (design)   2023-01-01 .. 2025-04-01
             OOS (held out) 2025-04-01 .. latest
Costs        per fill 0.10% fee + 0.08% slippage (= the live bot's paper
             model), short borrow 0.03%/day; stress test at 2x.
Tradable     listed >= 30 days, 7-day avg quote volume >= $2M/day.

Candidates (the complete grid — nothing is added after seeing results):
  tsmom_ls_{7,14,28}     daily time-series momentum, long/short sleeves
  tsmom_lo_{7,14,28}     same, long-only
  xsmom_{7,14,28}        daily cross-sectional momentum (skip 1 day), top/bottom 20%, dollar-neutral
  xsrev_{1,3}            daily cross-sectional reversal, dollar-neutral
  xsrev_4h               4h cross-sectional reversal
  donchian_ls_{20,55}    daily breakout, long/short
  donchian_lo_{20,55}    daily breakout, long-only
  mr_1h                  1h z-score mean reversion (24h window, in 2.5 / out 0.5)
  carry_{1,2}            perp cash-and-carry while 3-day mean funding > 0.01% / 0.02% per 8h
Benchmarks: equal-weight market (daily rebalanced), BTC buy-and-hold.

Decision rules:
  CANDIDATE (IS only)   IS Sharpe >= 1.0 and IS alpha t-stat >= 2.0 vs the
                        equal-weight market (so pure beta cannot qualify)
  CONFIRMED (OOS)       a CANDIDATE that also has OOS Sharpe >= 0.75,
                        OOS alpha > 0, OOS Sharpe >= 0.5 at 2x costs, and a
                        same-family neighbour with IS Sharpe >= 0.5
Only a CONFIRMED signal is worth a paper experiment in the live bot.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from config import SYMBOLS
from research import backtest as bt
from research import strategies as S
from research.data import ROOT, build_panel, load_funding, resample, utc

DATA_START = utc(2022, 10, 1)
IS_START, OOS_START = utc(2023, 1, 1), utc(2025, 4, 1)
FREQ = {"1h": (24, 8760), "4h": (6, 2190), "1D": (1, 365), "8h": (3, 1095)}
CANDIDATE = {"is_sharpe": 1.0, "is_alpha_t": 2.0}
CONFIRM = {"oos_sharpe": 0.75, "oos_alpha": 0.0, "oos_sharpe_2x": 0.5, "neighbour_is_sharpe": 0.5}


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    freq: str
    param: float
    make: object = field(compare=False)


def specs() -> list[Spec]:
    out = []
    for L in (7, 14, 28):
        out.append(Spec(f"tsmom_ls_{L}", "tsmom_ls", "1D", L, partial(S.tsmom, lookback=L)))
        out.append(Spec(f"tsmom_lo_{L}", "tsmom_lo", "1D", L, partial(S.tsmom, lookback=L, long_only=True)))
        out.append(Spec(f"xsmom_{L}", "xsmom", "1D", L, partial(S.xs_rank, lookback=L, skip=1)))
    for L in (1, 3):
        out.append(Spec(f"xsrev_{L}", "xsrev", "1D", L, partial(S.xs_rank, lookback=L, reverse=True)))
    out.append(Spec("xsrev_4h", "xsrev_4h", "4h", 1, partial(S.xs_rank, lookback=1, reverse=True)))
    for n in (20, 55):
        out.append(Spec(f"donchian_ls_{n}", "donchian_ls", "1D", n, partial(S.donchian, n=n)))
        out.append(Spec(f"donchian_lo_{n}", "donchian_lo", "1D", n, partial(S.donchian, n=n, long_only=True)))
    out.append(Spec("mr_1h", "mr", "1h", 24, partial(S.zscore_mr, window=24)))
    for k, th in ((1, 0.0001), (2, 0.0002)):
        out.append(Spec(f"carry_{k}", "carry", "8h", th, None))
    return out


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _split(res: bt.Result, bpy: float, bench: pd.Series) -> dict:
    return {
        "is": bt.metrics(res.window(IS_START, OOS_START - pd.Timedelta("1ns")), bpy,
                         bench[IS_START:OOS_START - pd.Timedelta("1ns")]),
        "oos": bt.metrics(res.window(OOS_START, None), bpy, bench[OOS_START:]),
    }


def _carry_result(funding: pd.DataFrame, theta: float, cost_mult: float = 1.0) -> bt.Result:
    sleeve = S.funding_carry(funding, theta, switch_cost=0.0031 * cost_mult)
    port = sleeve.mean(axis=1).fillna(0.0).shift(1).fillna(0.0)      # label by period end
    on = sleeve.notna() & (sleeve != 0)
    z = pd.Series(0.0, index=port.index)
    return bt.Result(net=port, gross=port, cost=z, borrow=z,
                     turnover=z, gross_exposure=on.mean(axis=1).shift(1).fillna(0.0), net_exposure=z)


def main() -> None:
    t0 = time.time()
    _log(f"building 1h panel for {len(SYMBOLS)} symbols from {DATA_START.date()} (cached after first run)")
    p1h = build_panel(SYMBOLS, "1h", DATA_START)
    panels = {"1h": p1h, "4h": resample(p1h, "4h"), "1D": resample(p1h, "1D")}
    masks = {f: S.tradable(panels[f], FREQ[f][0]) for f in panels}
    bench = {}
    for f, p in panels.items():
        prev = masks[f].shift(1, fill_value=False)
        bench[f] = p["close"].pct_change(fill_method=None).where(prev).mean(axis=1).fillna(0.0)
    end = p1h["close"].index.max()
    _log(f"panel ready: {p1h['close'].shape[0]} hourly bars, last {end}")

    _log("loading perp funding history")
    fund = {}
    for s in SYMBOLS:
        try:
            f = load_funding(s, DATA_START)
            if len(f):
                fund[s] = f
        except Exception as exc:          # no perp for this symbol
            _log(f"  funding {s}: {exc}")
    funding = pd.DataFrame(fund).sort_index()
    funding.index = funding.index.floor("8h")
    funding = funding.groupby(level=0).last()
    bench["8h"] = bench["1h"].pipe(lambda r: (1 + r).resample("8h", closed="right", label="right").prod() - 1)

    rows = {}
    for sp in specs():
        bpd, bpy = FREQ[sp.freq]
        if sp.family == "carry":
            res = _carry_result(funding, sp.param)
            res2 = _carry_result(funding, sp.param, 2.0)
        else:
            p, m = panels[sp.freq], masks[sp.freq]
            w = sp.make(p, m)
            res = bt.run(w, p["close"], bpd)
            res2 = bt.run(w, p["close"], bpd, bt.Costs().scaled(2.0))
        r = _split(res, bpy, bench[sp.freq])
        r["oos_2x"] = bt.metrics(res2.window(OOS_START, None), bpy, bench[sp.freq][OOS_START:])
        r["daily_equity"] = (1 + res.net).resample("1D", closed="right", label="right").prod().cumprod()
        rows[sp.name] = r
        _log(f"  {sp.name:15} IS sharpe {r['is'].get('sharpe', float('nan')):+.2f}  "
             f"OOS sharpe {r['oos'].get('sharpe', float('nan')):+.2f}")

    for name, fn in (("bench_ew", S.equal_weight), ("bench_btc", S.hold)):
        res = bt.run(fn(panels["1D"], masks["1D"]), panels["1D"]["close"], 1)
        r = _split(res, 365, bench["1D"])
        r["daily_equity"] = (1 + res.net).cumprod()
        rows[name] = r

    # ── decision rules ──────────────────────────────────────────────────────
    sp_by = {s.name: s for s in specs()}
    verdict = {}
    for s in specs():
        r = rows[s.name]
        isr, oos, oos2 = r["is"], r["oos"], r["oos_2x"]
        cand = isr.get("sharpe", -9) >= CANDIDATE["is_sharpe"] and isr.get("alpha_t", -9) >= CANDIDATE["is_alpha_t"]
        fam = sorted((x for x in specs() if x.family == s.family), key=lambda x: x.param)
        i = [x.name for x in fam].index(s.name)
        neigh = [fam[j].name for j in (i - 1, i + 1) if 0 <= j < len(fam)]
        neigh_ok = any(rows[n]["is"].get("sharpe", -9) >= CONFIRM["neighbour_is_sharpe"] for n in neigh) if neigh else True
        conf = (cand and oos.get("sharpe", -9) >= CONFIRM["oos_sharpe"] and oos.get("alpha_yr", -9) > CONFIRM["oos_alpha"]
                and oos2.get("sharpe", -9) >= CONFIRM["oos_sharpe_2x"] and neigh_ok)
        verdict[s.name] = "CONFIRMED" if conf else ("candidate (failed OOS)" if cand else "rejected")

    # ── report ──────────────────────────────────────────────────────────────
    def f(x, pct=False):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "   —  "
        return f"{x * 100:+6.1f}%" if pct else f"{x:+6.2f}"

    lines = [f"Signal study — IS {IS_START.date()}..{OOS_START.date()}  OOS {OOS_START.date()}..{end.date()}",
             "", f"{'strategy':16}{'IS Sh':>8}{'IS CAGR':>9}{'IS a_t':>8}{'β':>6} | {'OOS Sh':>7}{'OOS CAGR':>9}"
                 f"{'OOS MDD':>9}{'a/yr':>8}{'a_t':>7}{'Sh 2x':>7}{'cost/yr':>9}  verdict"]
    for name, r in rows.items():
        i, o, o2 = r["is"], r["oos"], r.get("oos_2x", {})
        lines.append(f"{name:16}{f(i.get('sharpe')):>8}{f(i.get('cagr'), True):>9}{f(i.get('alpha_t')):>8}"
                     f"{f(i.get('beta')):>6} | {f(o.get('sharpe')):>7}{f(o.get('cagr'), True):>9}"
                     f"{f(o.get('max_dd'), True):>9}{f(o.get('alpha_yr'), True):>8}{f(o.get('alpha_t')):>7}"
                     f"{f(o2.get('sharpe')):>7}{f(o.get('cost_yr'), True):>9}  {verdict.get(name, 'benchmark')}")
    report = "\n".join(lines)
    print(report)

    out = ROOT / "results" / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report + "\n")
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "daily_equity"} for k, v in rows.items()}
    (out / "summary.json").write_text(json.dumps({"verdict": verdict, "metrics": summary,
                                                  "is": str(IS_START), "oos": str(OOS_START), "end": str(end)},
                                                 indent=1, default=float))
    pd.DataFrame({k: v["daily_equity"] for k, v in rows.items()}).to_csv(out / "daily_equity.csv")
    _log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
