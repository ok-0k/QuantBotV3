"""
Crowding study runner. Pre-registered in research/PREREG_FUNDING.md (98e8cec),
which the PM approved as written. This file implements that document's
sections 10 (reporting) and 11 (decision rules) using the functions and
constants of research.crowding, unchanged.

    TRADING_ENV_FILE=/nonexistent TRADING_DATA_DIR=<scratch> TRADING_LOG_DIR=<scratch> \\
    RESEARCH_OFFLINE=1 .venv/bin/python -m research.crowding_study --out <dir>

(The TRADING_* variables keep `import config` away from the live data dir
and the secrets file; RESEARCH_OFFLINE=1 is enforced by crowding.load_inputs.)

Per configuration, the book is simulated four times on the 1h grid: base
costs, 2x costs, borrow stress (0.10%/day) and zero cost (a diagnostic). The
drop-one re-runs and the schedule-phase diagnostics are base costs only, and
run only where section 11 says so.

Outputs in --out: report.txt, summary.json, daily_equity.csv (base costs).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import backtest as bt
from research import crowding as C
from research.data import ROOT
from research.study import IS_START, OOS_START
from research.universe import RESEARCH_SYMBOLS as SYMBOLS

BONFERRONI_T = 3.08                     # one-sided 5% / 48 tests (PREREG section 12), informational
CHECKS = ("is_sharpe", "is_alpha_t", "oos_sharpe", "oos_alpha_pos", "oos_sharpe_2x", "neighbour",
          "oos_alpha_min", "oos_h1_alpha", "oos_h2_alpha", "borrow_sharpe", "borrow_alpha",
          "coverage", "drop_one")
ZERO = bt.Costs(0.0, 0.0, 0.0)
NS = pd.Timedelta("1ns")
REPO = Path(__file__).resolve().parent.parent


def windows() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Metric windows on 1h bar labels, both ends inclusive."""
    return {"is": (IS_START, OOS_START - NS), "oos": (OOS_START, C.END),
            "oos_h1": (OOS_START, C.OOS_SPLIT - NS), "oos_h2": (C.OOS_SPLIT, C.END)}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _metrics(res: bt.Result, bench: pd.Series, key: str) -> dict:
    a, b = windows()[key]
    return bt.metrics(res.window(a, b), C.BARS_PER_YEAR, bench[a:b])


def books(inputs: dict) -> tuple[dict, dict]:
    sig = C.signals(inputs["F"], inputs["P"])
    return sig, {s: C.eligible(sig[s], inputs["spot_ok"], inputs["liquid"]) for s in C.SIGNALS}


def coverage(log: pd.DataFrame) -> dict:
    """Executed / skipped rebalances per window. OOS decision labels are in
    [OOS_START, END) (PREREG rule `coverage`), IS labels in [IS_START, OOS_START)."""
    out = {}
    for key, a, b in (("is", IS_START, OOS_START), ("oos", OOS_START, C.END)):
        lg = log[(log.index >= a) & (log.index < b)]
        n, sk = len(lg), int(lg["skipped"].sum()) if len(lg) else 0
        out.update({f"{key}_decisions": n, f"{key}_skipped": sk, f"{key}_executed": n - sk,
                    f"{key}_eligible_mean": float(lg["n_eligible"].mean()) if n else float("nan"),
                    f"{key}_eligible_min": int(lg["n_eligible"].min()) if n else 0})
    out["skipped_oos"] = out["oos_skipped"] / out["oos_decisions"] if out["oos_decisions"] else 1.0
    return out


def held_targets(W: pd.DataFrame, ta: pd.Series) -> pd.DataFrame:
    """The last executed target on every 1h row. Between rebalances a position
    keeps its target's sign (the band only ever keeps a same-sign position)."""
    return W.loc[ta.to_numpy()].reindex(W.index).ffill().fillna(0.0)


def short_time_share(W: pd.DataFrame, ta: pd.Series, top: int = 5) -> dict:
    """Each symbol's share of OOS short-holding time (symbol-hours short)."""
    h = held_targets(W, ta).loc[OOS_START:C.END]
    n = (h < 0).sum()
    tot = n.sum()
    return {s: float(v / tot) for s, v in n.sort_values(ascending=False).head(top).items() if v > 0} if tot else {}


def oos_symbols(W: pd.DataFrame, ta: pd.Series) -> list[str]:
    """Symbols with a non-zero target on any OOS fill row (PREREG rule `drop_one`)."""
    rows = ta.to_numpy() & (W.index >= OOS_START) & (W.index <= C.END)
    return [s for s in W.columns if (W.loc[rows, s] != 0).any()]


def run_config(cfg: C.Config, sig: dict, elig: dict, close: pd.DataFrame, bench: pd.Series,
               cases=("base", "2x", "borrow", "zero"), times: pd.DatetimeIndex | None = None) -> dict:
    """Simulate one configuration. times (diagnostics only) replaces the
    schedule with explicit decision labels; the weights still come from
    crowding.targets, unchanged."""
    score = sig[cfg.signal] if times is None else sig[cfg.signal].loc[times]
    W, ta, log = C.targets(score, elig[cfg.signal], cfg.k, cfg.book,
                           cfg.period if times is None else "8h", close.index)
    res = {c: bt.run(W, close, C.BARS_PER_DAY, ZERO if c == "zero" else C.COSTS[c],
                     band=C.BAND, trade_at=ta) for c in cases}
    base = res["base"]
    row = {"is": _metrics(base, bench, "is"), "oos": _metrics(base, bench, "oos"),
           "oos_h1": _metrics(base, bench, "oos_h1"), "oos_h2": _metrics(base, bench, "oos_h2")}
    for c, key in (("2x", "oos_2x"), ("borrow", "oos_borrow"), ("zero", "oos_zero")):
        if c in res:
            row[key] = _metrics(res[c], bench, "oos")
    row["coverage"] = coverage(log)
    row["skipped_oos"] = row["coverage"]["skipped_oos"]
    return {"row": row, "base": base, "W": W, "ta": ta}


def drop_one(cfg: C.Config, inputs: dict, symbols: list[str]) -> dict[str, float]:
    """OOS alpha (base costs) of cfg re-run once per symbol, with that
    symbol's column removed from every input except the benchmark."""
    out = {}
    for s in symbols:
        sub = {"p1h": {k: v.drop(columns=s) for k, v in inputs["p1h"].items()},
               **{k: inputs[k].drop(columns=s) for k in ("F", "P", "spot_ok", "liquid")}}
        sig = C.signals(sub["F"], sub["P"])           # recomputed without s; only cfg's signal is needed
        elig = {cfg.signal: C.eligible(sig[cfg.signal], sub["spot_ok"], sub["liquid"])}
        r = run_config(cfg, sig, elig, sub["p1h"]["close"], inputs["bench"], cases=("base",))
        out[s] = r["row"]["oos"].get("alpha_yr", float("nan"))
    return out


def phase_times(index: pd.DatetimeIndex, period: str) -> dict[str, pd.DatetimeIndex]:
    """The schedule's other phases (PREREG section 10 diagnostic)."""
    if period == "24h":
        return {"24h@08:00": index[index.hour == 8], "24h@16:00": index[index.hour == 16]}
    if period == "72h":
        d = np.asarray((index - C.EPOCH).days) % 3
        return {f"72h@day%3=={j}": index[(index.hour == 0) & (d == j)] for j in (1, 2)}
    return {}


def references(sig, elig, close, bench) -> dict:
    """Benchmarks and the short-the-universe books (never candidates)."""
    out = {}
    z = pd.Series(0.0, index=close.index)
    for name, r in (("bench_ew", bench),
                    ("bench_btc", close["BTCUSDT"].pct_change(fill_method=None).fillna(0.0)
                     if "BTCUSDT" in close else None)):
        if r is not None:
            res = bt.Result(net=r, gross=r, cost=z, borrow=z, turnover=z, gross_exposure=z + 1, net_exposure=z + 1)
            out[name] = {"res": res}
    fl = sig["fund_lvl"]
    flat = fl.notna().astype(float).where(fl.notna())     # every eligible name ties -> -1/N each
    for p in C.PERIODS:
        W, ta, log = C.targets(flat, elig["fund_lvl"], 3, "so", p, close.index)
        out[f"ref_short_all_{p}"] = {"res": bt.run(W, close, C.BARS_PER_DAY, C.COSTS["base"],
                                                   band=C.BAND, trade_at=ta), "log": log}
    for v in out.values():
        res = v.pop("res")
        v["row"] = {k: _metrics(res, bench, k) for k in windows()}
        if "log" in v:
            v["row"]["coverage"] = coverage(v.pop("log"))
        v["equity"] = (1 + res.net).resample("1D", closed="right", label="right").prod().cumprod()
    return out


def status(name: str, v: dict) -> dict[str, str]:
    """Each check as pass / FAIL / n/a (drop_one runs only once all else passed)."""
    st = {c: ("FAIL" if c in v[name]["failed"] else "pass") for c in CHECKS}
    if st["drop_one"] == "pass" and not v[name]["verdict"].startswith("PASSED"):
        st["drop_one"] = "n/a"
    return st


def _git() -> str:
    try:
        h = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception as exc:               # not a checkout: record why, never fail the run
        return f"unknown ({exc.__class__.__name__})"


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _f(x, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:+.1f}%" if pct else f"{x:+.2f}"


def report(rows, v, refs, meta) -> str:
    g = lambda r, k, f: r.get(k, {}).get(f)
    L = [f"Crowding study (PREREG_FUNDING.md) — IS {IS_START.date()}..{OOS_START.date()}  "
         f"OOS {OOS_START.date()}..{C.END}  split {C.OOS_SPLIT.date()}  band {C.BAND:g}",
         f"commit {meta['commit']}  data {meta['fingerprint'][:16]}  runtime {meta['runtime_s']:.0f}s", "",
         "1. All 48 configurations (base costs unless noted; alpha vs the EW universe, annualised)", ""]
    hdr = (f"{'config':22}{'IS Sh':>7}{'IS a':>8}{'IS t':>7} |{'OOS Sh':>7}{'OOS a':>8}{'t':>6}"
           f"{'H1 a':>8}{'H2 a':>8}{'2x Sh':>7}{'brw Sh':>7}{'brw a':>8}{'0c a':>8}{'skip':>6}  verdict")
    L += [hdr, "-" * len(hdr)]
    for c in C.grid():
        r = rows[c.name]
        L.append(f"{c.name:22}{_f(g(r,'is','sharpe')):>7}{_f(g(r,'is','alpha_yr'),1):>8}{_f(g(r,'is','alpha_t')):>7} |"
                 f"{_f(g(r,'oos','sharpe')):>7}{_f(g(r,'oos','alpha_yr'),1):>8}{_f(g(r,'oos','alpha_t')):>6}"
                 f"{_f(g(r,'oos_h1','alpha_yr'),1):>8}{_f(g(r,'oos_h2','alpha_yr'),1):>8}"
                 f"{_f(g(r,'oos_2x','sharpe')):>7}{_f(g(r,'oos_borrow','sharpe')):>7}"
                 f"{_f(g(r,'oos_borrow','alpha_yr'),1):>8}{_f(g(r,'oos_zero','alpha_yr'),1):>8}"
                 f"{r['skipped_oos'] * 100:>5.1f}%  {v[c.name]['verdict']}")
    L += ["", "2. Decision rules per configuration (pass / FAIL / n/a = not run)", ""]
    ab = ("ISsh", "ISt", "OSsh", "OSa>0", "2xSh", "neigh", "a>=10", "H1", "H2", "bSh", "ba", "cov", "drop1")
    L.append(f"{'config':22}" + "".join(f"{a:>7}" for a in ab))
    for c in C.grid():
        st = status(c.name, v)
        L.append(f"{c.name:22}" + "".join(f"{st[k]:>7}" for k in CHECKS))
    L += ["", "3. References (never candidates)", ""]
    for name, x in refs.items():
        r = x["row"]
        L.append(f"{name:22} IS Sh {_f(g(r,'is','sharpe'))}  IS CAGR {_f(g(r,'is','cagr'),1)}  "
                 f"OOS Sh {_f(g(r,'oos','sharpe'))}  OOS CAGR {_f(g(r,'oos','cagr'),1)}  "
                 f"OOS a {_f(g(r,'oos','alpha_yr'),1)}  beta {_f(g(r,'oos','beta'))}")
    cands = [c.name for c in C.grid() if not v[c.name]["verdict"].startswith("rejected")]
    passed = [n for n in cands if v[n]["verdict"].startswith("PASSED")]
    L += ["", f"4. Candidates (IS gate): {len(cands)}   PASSED: {len(passed)}", ""]
    for n in cands:
        r = rows[n]
        L.append(f"{n:22} {v[n]['verdict']}; failed: {', '.join(v[n]['failed']) or 'none'}; "
                 f"IS alpha t {_f(g(r,'is','alpha_t'))} (Bonferroni 3.08: "
                 f"{'clears' if r['bonferroni'] else 'does not clear'})")
        if "drop_one" in r:
            worst = min(r["drop_one"], key=lambda s: r["drop_one"][s])
            L.append(f"{'':22} drop-one min OOS alpha {_f(r['drop_one_min_alpha'],1)} (without {worst})")
        for ph, m in r.get("phases", {}).items():
            L.append(f"{'':22} phase {ph}: OOS Sh {_f(m.get('sharpe'))}  OOS a {_f(m.get('alpha_yr'),1)}")
        L.append(f"{'':22} top OOS short time: " + ", ".join(f"{s} {x:.0%}" for s, x in r["concentration"].items()))
    L += ["", "PASSED = candidate only: an independent clean re-run, then a new rebalancing executor with its",
          "own review, then a forward paper test (PREREG section 14). 48 configs were tested (section 12)."]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="research.crowding_study")
    ap.add_argument("--out", type=Path, default=None)
    out = ap.parse_args(argv if argv is not None else sys.argv[1:]).out
    out = out or ROOT / "results" / (time.strftime("%Y%m%dT%H%M%S") + "-crowding")
    t0 = time.time()
    commit = _git()
    _log(f"commit {commit}; loading inputs (offline cache, END {C.END})")
    inputs = C.load_inputs(SYMBOLS)
    fp = C.fingerprint(inputs)
    close, bench = inputs["p1h"]["close"], inputs["bench"]
    _log(f"inputs ready: {close.shape[0]} 1h bars x {close.shape[1]} symbols, last {close.index.max()}; "
         f"fingerprint {fp[:16]}")
    sig, elig = books(inputs)

    rows, eq, held = {}, {}, {}
    for c in C.grid():
        r = run_config(c, sig, elig, close, bench)
        row = r["row"]
        row["bonferroni"] = bool((row["is"].get("alpha_t") or -math.inf) >= BONFERRONI_T)
        row["concentration"] = short_time_share(r["W"], r["ta"])
        rows[c.name], held[c.name] = row, oos_symbols(r["W"], r["ta"])
        eq[c.name] = (1 + r["base"].net).resample("1D", closed="right", label="right").prod().cumprod()
        _log(f"  {c.name:22} IS Sh {_f(row['is'].get('sharpe'))}  OOS Sh {_f(row['oos'].get('sharpe'))}  "
             f"OOS a {_f(row['oos'].get('alpha_yr'), 1)}")

    v = C.verdict(rows)
    for c in C.grid():                                 # drop-one only where everything else passed
        if v[c.name]["failed"] == ["drop_one"]:
            _log(f"  drop-one for {c.name} over {len(held[c.name])} symbols")
            d = drop_one(c, inputs, held[c.name])
            rows[c.name]["drop_one"] = d
            rows[c.name]["drop_one_min_alpha"] = min(d.values()) if d else float("nan")
    v = C.verdict(rows)
    for c in C.grid():                                 # phase diagnostics, PASSED only, never decisional
        if v[c.name]["verdict"].startswith("PASSED"):
            rows[c.name]["phases"] = {
                ph: run_config(c, sig, elig, close, bench, cases=("base",), times=t)["row"]["oos"]
                for ph, t in phase_times(sig[c.signal].index, c.period).items()}

    refs = references(sig, elig, close, bench)
    meta = {"commit": commit, "fingerprint": fp, "end": str(C.END), "is": str(IS_START), "oos": str(OOS_START),
            "oos_split": str(C.OOS_SPLIT), "band": C.BAND, "lag": str(C.LAG), "symbols": list(close.columns),
            "last_bar": str(close.index.max()), "runtime_s": time.time() - t0}
    text = report(rows, v, refs, meta)
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text + "\n")
    summary = {"meta": meta,
               "verdict": {n: v[n]["verdict"] for n in v},
               "checks": {n: status(n, v) for n in v},
               "failed": {n: v[n]["failed"] for n in v},
               "configs": rows,
               "references": {n: x["row"] for n, x in refs.items()}}
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    pd.DataFrame({**eq, **{n: x["equity"] for n, x in refs.items()}}).to_csv(out / "daily_equity.csv",
                                                                             float_format="%.8g")
    _log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
