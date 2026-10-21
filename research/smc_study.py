"""
SMC playbook study runner. Pre-registered in research/PREREG_SMC.md
(revision 1, approved by the PM at 1fc108a). This file implements sections
10 (reporting) and 11 (decision rules) with the functions and constants of
research.smc, unchanged.

    cd /home/admin/wt-research-smc && \\
    TRADING_ENV_FILE=/nonexistent DISCORD_WEBHOOK_URL= \\
    TRADING_DATA_DIR=<scratch>/td TRADING_LOG_DIR=<scratch>/tl RESEARCH_OFFLINE=1 \\
    /home/admin/trading_bot/.venv/bin/python -m research.smc_study

Refuses to run unless RESEARCH_OFFLINE=1, TRADING_ENV_FILE names a file that
does not exist (so `import config` cannot load secrets), and the git tree is
clean. Everything is computed in ONE invocation:
- 8 configs x sides (long, short, both) x scopes (per-symbol, portfolio) x
  costs (base, 2x), each summarised for IS, OOS, OOS-H1 and OOS-H2;
- the verdict for every config x side. Only `short` decides
  (smc.DECISION_SIDES). The portfolio input to port_exp_pos is the same
  side's OOS at base cost;
- the three sanity counters (fvg_invalid cancels, two-loss binds,
  spot_tradable mismatches) for every config x side x scope x cost. If any is
  not 0, every verdict is FROZEN (PREREG §10).

Output: research/runs/smc_run1_<UTC>/ (or --out), containing command.txt,
stdout.txt, stderr.txt, exit_code.txt, summary.json, report.txt,
trades.csv.gz (every trade of every simulation) and setups.csv.gz (every
setup with its outcomes). Until every result file is written, only progress
is logged (symbol names, elapsed time); the report is printed last.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research import data
from research import smc
from research.universe import RESEARCH_SYMBOLS

REPO = Path(__file__).resolve().parent.parent
SCOPES = ("symbol", "portfolio")
COST_CASES = ("base", "2x")
PERIODS = ("IS", "OOS", "OOS_H1", "OOS_H2")
COUNTERS = ("fvg_invalid", "two_loss_binds", "spot_mismatch")
REFS = ("BTCUSDT", "ETHUSDT")
BIAS_NAME = {1: "bullish", -1: "bearish", 0: "neutral"}


def universe() -> list[str]:
    return list(RESEARCH_SYMBOLS)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def preflight() -> str:
    """Refuse to run unless offline, secrets-proof and on a clean tree. Returns the commit."""
    problems = []
    env_off = os.getenv("RESEARCH_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    if not (env_off and data.OFFLINE):
        problems.append("RESEARCH_OFFLINE=1 is required")
    env_file = os.getenv("TRADING_ENV_FILE", "").strip()
    if not env_file or Path(env_file).exists():
        problems.append("TRADING_ENV_FILE must name a file that does not exist")
    dirty = _git("status", "--porcelain").strip()
    if dirty:
        problems.append("git tree is dirty:\n" + dirty)
    if problems:
        raise SystemExit("refusing to run: " + "; ".join(problems))
    return _git("rev-parse", "HEAD").strip()


# ── computation ─────────────────────────────────────────────────────────────

def _date(day: int) -> str:
    return str(smc.EPOCH + timedelta(days=int(day)))


def _iso(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()


def data_info(b: smc.Bars) -> dict:
    ok = ~np.isnan(b.c)
    if not ok.any():
        return {"bars": 0}
    idx = np.flatnonzero(ok)
    return {"bars": int(ok.sum()), "first": _iso(b.t[idx[0]]), "last": _iso(b.t[idx[-1]])}


def symbol_pass(prep: smc.Prepared, ref: dict | None) -> dict:
    """Setups (both TFs and sides) with outcomes, day contexts, funnel counts
    and the spot_tradable mismatch days, for one prepared symbol."""
    cands, funnel = [], {}
    for tf in (5, 15):
        for side in (1, -1):
            cnt: dict = {}
            cands += smc.find_candidates(prep, tf, side, ref=ref, counts=cnt)
            funnel[f"{tf}m_{'long' if side == 1 else 'short'}"] = cnt
    smc.attach_outcomes(cands, prep.b5, {k: v["b"] for k, v in prep.tf.items()})
    eligible = [d for d, c in prep.ctx.items() if c.eligible]
    mismatch = set(smc.spot_mismatch_days(prep.b5, eligible))
    ticks = [c.tick / c.p0 for c in prep.ctx.values() if c.eligible and c.p0 == c.p0 and c.p0 > 0]
    return {"cands": cands, "ctx": prep.ctx, "funnel": funnel, "mismatch": mismatch,
            "tick_over_price": float(np.median(ticks)) if ticks else None}


def run(commit: str) -> dict:
    t0 = time.time()
    syms = universe()
    days = smc.trading_days()
    _log(f"commit {commit}; {len(syms)} symbols; offline cache, END {smc.END}")
    hsh, info = hashlib.sha256(), {}
    for sym in sorted(syms):                         # pass 1: fingerprint, one symbol in memory at a time
        b = smc.load_symbol(sym)
        smc.fingerprint_update(hsh, sym, b)
        info[sym] = data_info(b)
    fp = hsh.hexdigest()
    _log("data fingerprinted")
    preps, refs = {}, {}
    for sym in REFS:                                 # SMT references: BTC for alts, ETH for BTC
        if sym in syms:
            preps[sym] = smc.prepare(sym, smc.load_symbol(sym), days)
            refs[sym] = smc.ref_taken_times(preps[sym])
    per = {}
    for i, sym in enumerate(syms, 1):
        prep = preps.pop(sym, None) or smc.prepare(sym, smc.load_symbol(sym), days)
        ref = refs.get("ETHUSDT") if sym == "BTCUSDT" else refs.get("BTCUSDT")
        per[sym] = symbol_pass(prep, ref)
        del prep
        _log(f"{sym} done ({i}/{len(syms)}, {time.time() - t0:.0f}s)")
    cands = [c for s in syms for c in per[s]["cands"]]
    day_of = {(c.sym, c.tf, c.side, c.t_place): c.day for c in cands}
    ctxs = {s: per[s]["ctx"] for s in syms}
    order = {s: i for i, s in enumerate(syms)}
    runs: dict = {}
    trades: list[dict] = []
    for cfg in smc.grid():
        for side_name, sides in smc.SIDES.items():
            for scope in SCOPES:
                for ck in COST_CASES:
                    st: dict = {}
                    log: list = []
                    tr = smc.walk(cands, ctxs, cfg, sides, ck, scope=scope, sym_order=order,
                                  stats=st, placed_log=log)
                    sp = smc.split(tr)
                    placed_days = {(k[0], day_of[k]) for k in log}
                    counters = {"fvg_invalid": int(st.get("outcome_fvg_invalid", 0)),
                                "two_loss_binds": int(st.get("two_loss_binds", 0)),
                                "spot_mismatch": int(sum(d in per[s]["mismatch"] for s, d in placed_days))}
                    runs.setdefault(cfg.name, {}).setdefault(side_name, {}).setdefault(scope, {})[ck] = {
                        "periods": {p: smc.summarize(sp[p]) for p in PERIODS},
                        "walk": dict(sorted(st.items())), "counters": counters}
                    trades += [{"config": cfg.name, "sides": side_name, "scope": scope, "cost": ck,
                                "sym": x.sym, "day": _date(x.day), "period": smc.period_of(x.day),
                                "side": x.side, "grade": x.grade, "w": x.w, "r": x.r, "r_gross": x.r_gross,
                                "reason": x.reason, "t_place": _iso(x.t_place), "t_fill": _iso(x.t_fill),
                                "t_exit": _iso(x.t_exit)} for x in tr]
        _log(f"{cfg.name} simulated ({time.time() - t0:.0f}s)")
    totals = {k: sum(runs[c][sd][sc][ck]["counters"][k] for c in runs for sd in runs[c]
                     for sc in runs[c][sd] for ck in runs[c][sd][sc]) for k in COUNTERS}
    frozen = any(v != 0 for v in totals.values())
    verdicts = {}
    for cfg in smc.grid():
        for side_name in smc.SIDES:
            r = runs[cfg.name][side_name]
            v, failed = smc.verdict(r["symbol"]["base"]["periods"], r["symbol"]["2x"]["periods"]["OOS"],
                                    r["portfolio"]["base"]["periods"]["OOS"])
            decisional = side_name in smc.DECISION_SIDES
            shown = v if decisional else f"informational ({v})"
            if frozen:
                shown = "FROZEN (a sanity counter is not 0; PREREG §10)"
            verdicts.setdefault(cfg.name, {})[side_name] = {
                "verdict": shown, "computed": v, "failed": failed, "decisional": decisional}
    why, bias, neutral = {}, {}, 0
    for s in syms:
        for D, c in per[s]["ctx"].items():
            if not c.eligible:
                why[c.why] = why.get(c.why, 0) + 1
                continue
            key = f"{smc.period_of(D)}:{BIAS_NAME[c.bias]}"
            bias[key] = bias.get(key, 0) + 1
            neutral += int(c.bias != 0 and c.neutral_from < smc.ny_ms(D, smc.KZ_END))
    funnel: dict = {}
    for s in syms:
        for k, cnt in per[s]["funnel"].items():
            for kk, vv in cnt.items():
                funnel.setdefault(k, {})[kk] = funnel.get(k, {}).get(kk, 0) + vv
    for c in cands:
        k = f"{c.tf}m_{'long' if c.side == 1 else 'short'}"
        g = f"grade_{c.grade or 'none'}"
        funnel.setdefault(k, {})[g] = funnel[k].get(g, 0) + 1
    setups = [{"sym": c.sym, "tf": c.tf, "side": c.side, "day": _date(c.day), "period": smc.period_of(c.day),
               "t_place": _iso(c.t_place), "limit": c.actual(c.limit), "stop": c.actual(c.stop),
               "tp1": c.actual(c.tp1), "tp2": c.actual(c.tp2), "tick": c.tick, "score": c.score,
               "grade": c.grade, "aligned": c.aligned, "discount": c.discount, "smt": c.smt,
               "sweep_kinds": "+".join(c.sweep_kinds), "t_sweep": _iso(c.t_sweep), "t_mss": _iso(c.t_mss),
               **{f"{mg}_{f}": v for mg in ("pbe", "tp2") for f, v in (
                   ("filled", c.out[mg].filled), ("reason", c.out[mg].reason),
                   ("r_base", c.r[(mg, "base")]), ("r_2x", c.r[(mg, "2x")]), ("r_zero", c.r[(mg, "zero")]))}}
              for c in cands]
    meta = {"commit": commit, "fingerprint": fp, "end": str(smc.END), "data_start": str(smc.DATA_START),
            "is_start": str(smc.IS_START), "oos_start": str(smc.OOS_START), "oos_split": str(smc.OOS_SPLIT),
            "symbols": syms, "trading_days": len(days), "decision_sides": list(smc.DECISION_SIDES),
            "pass": smc.PASS, "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "runtime_s": round(time.time() - t0, 1)}
    context = {"ineligible_symbol_days": why, "eligible_symbol_days": int(sum(bias.values())),
               "bias_days": dict(sorted(bias.items())), "neutralised_days": neutral,
               "tick_over_price_median": {s: per[s]["tick_over_price"] for s in syms}, "data": info}
    return {"meta": meta, "runs": runs, "verdicts": verdicts, "sanity": {"totals": totals, "frozen": frozen},
            "funnel": funnel, "context": context, "trades": trades, "setups": setups}


# ── output ──────────────────────────────────────────────────────────────────

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _f(x, nd: int = 3, w: int = 7) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—".rjust(w)
    return f"{x:+.{nd}f}".rjust(w) if isinstance(x, float) else str(x).rjust(w)


def report(res: dict) -> str:
    m, san = res["meta"], res["sanity"]
    L = ["SMC playbook study (research/PREREG_SMC.md, revision 1)",
         f"commit {m['commit']}  fingerprint {m['fingerprint']}",
         f"END {m['end']}  IS {m['is_start']}..{m['oos_start']}  OOS {m['oos_start']}..END (split {m['oos_split']})",
         f"{len(m['symbols'])} symbols, {m['trading_days']} trading days, runtime {m['runtime_s']}s",
         "",
         "SANITY COUNTERS (must all be 0): " + ", ".join(f"{k}={v}" for k, v in san["totals"].items())
         + ("   ->  VERDICTS FROZEN (PREREG §10)" if san["frozen"] else "   ->  ok"),
         ""]
    hdr = (f"{'config / side':26}{'IS exp':>8}{'OOS n':>7}{'OOS exp':>8}{'OOS t':>7}{'H1 exp':>8}{'H2 exp':>8}"
           f"{'2x exp':>8}{'exp_w':>8}{'port':>8}{'sym%':>7}{'day%':>7}{'mdd R':>8}  verdict / failed checks")
    for title, sides in (("DECISION UNITS (config x short)", smc.DECISION_SIDES),
                         ("INFORMATIONAL (cannot produce a candidate)",
                          tuple(s for s in smc.SIDES if s not in smc.DECISION_SIDES))):
        L += [title, hdr]
        for side in sides:
            for cfg in smc.grid():
                r = res["runs"][cfg.name][side]
                p, o = r["symbol"]["base"]["periods"], r["symbol"]["base"]["periods"]["OOS"]
                v = res["verdicts"][cfg.name][side]
                L.append(f"{cfg.name + ' ' + side:26}{_f(p['IS'].get('exp'))}{o.get('n', 0):>7}"
                         f"{_f(o.get('exp'))}{_f(o.get('t'), 2)}{_f(p['OOS_H1'].get('exp'))}"
                         f"{_f(p['OOS_H2'].get('exp'))}{_f(r['symbol']['2x']['periods']['OOS'].get('exp'))}"
                         f"{_f(o.get('exp_w'))}{_f(r['portfolio']['base']['periods']['OOS'].get('exp'))}"
                         f"{_f(o.get('max_sym_share'), 2)}{_f(o.get('max_day_share'), 2)}{_f(o.get('mdd_r'), 1, 8)}"
                         f"  {v['verdict']}" + (f"  [{', '.join(v['failed'])}]" if v["failed"] else ""))
        L.append("")
    L += ["ALL SIMULATIONS (per period)",
          f"{'config':20}{'sides':>6}{'scope':>10}{'cost':>5}{'period':>8}{'n':>7}{'exp':>8}{'t':>7}{'win':>7}"
          f"{'PF':>7}{'tot R':>9}{'mdd R':>8}{'mdd U':>8}{'gross':>8}{'cost/tr':>8}{'A+':>6}"]
    for cfg in smc.grid():
        for side in smc.SIDES:
            for scope in SCOPES:
                for ck in COST_CASES:
                    for per in PERIODS:
                        s = res["runs"][cfg.name][side][scope][ck]["periods"][per]
                        L.append(f"{cfg.name:20}{side:>6}{scope:>10}{ck:>5}{per:>8}{s.get('n', 0):>7}"
                                 f"{_f(s.get('exp'))}{_f(s.get('t'), 2)}{_f(s.get('win_rate'), 2)}"
                                 f"{_f(s.get('profit_factor'), 2)}{_f(s.get('total_r'), 1, 9)}"
                                 f"{_f(s.get('mdd_r'), 1, 8)}{_f(s.get('mdd_u'), 1, 8)}{_f(s.get('exp_gross'))}"
                                 f"{_f(s.get('cost_per_trade_r'))}{s.get('n_a_plus', 0):>6}")
    L += ["", "ORDERS PER SIMULATION (walk stats; base cost, per-symbol scope)"]
    for cfg in smc.grid():
        for side in smc.SIDES:
            L.append(f"  {cfg.name:20} {side:6} " + ", ".join(
                f"{k}={v}" for k, v in res["runs"][cfg.name][side]["symbol"]["base"]["walk"].items()))
    L += ["", "SANITY COUNTERS PER SIMULATION (non-zero only)"]
    nz = [(c, sd, sc, ck, r["counters"]) for c, a in res["runs"].items() for sd, b in a.items()
          for sc, d in b.items() for ck, r in d.items() if any(r["counters"].values())]
    L += [f"  {c} {sd} {sc} {ck}: {cn}" for c, sd, sc, ck, cn in nz] or ["  none"]
    L += ["", "FUNNEL (all symbols)"]
    L += [f"  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in sorted(v.items())) for k, v in sorted(res["funnel"].items())]
    ctx = res["context"]
    L += ["", "CONTEXT",
          f"  eligible symbol-days {ctx['eligible_symbol_days']}; ineligible: "
          + ", ".join(f"{k}={v}" for k, v in sorted(ctx["ineligible_symbol_days"].items())),
          "  bias days: " + ", ".join(f"{k}={v}" for k, v in ctx["bias_days"].items()),
          f"  days neutralised by a 15m MSS (§5.5): {ctx['neutralised_days']}",
          "  median tick/price: " + ", ".join(f"{s}={v:.1e}" for s, v in ctx["tick_over_price_median"].items()
                                              if v is not None)]
    return "\n".join(L)


def write_outputs(out: Path, res: dict) -> None:
    summary = {k: res[k] for k in ("meta", "sanity", "verdicts", "runs", "funnel", "context")}
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    (out / "report.txt").write_text(report(res) + "\n")
    cols = ["config", "sides", "scope", "cost", "sym", "day", "period", "side", "grade", "w", "r", "r_gross",
            "reason", "t_place", "t_fill", "t_exit"]
    pd.DataFrame(res["trades"], columns=cols).to_csv(out / "trades.csv.gz", index=False,
                                                     float_format="%.10g", compression="gzip")
    pd.DataFrame(res["setups"]).to_csv(out / "setups.csv.gz", index=False, float_format="%.10g",
                                       compression="gzip")


class _Tee(io.TextIOBase):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, s):
        self.a.write(s)
        self.b.write(s)
        return len(s)

    def flush(self):
        self.a.flush()
        self.b.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.smc_study")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    commit = preflight()
    out = a.out or REPO / "research" / "runs" / f"smc_run1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    if out.exists():
        raise SystemExit(f"refusing to run: {out} already exists")
    out.mkdir(parents=True)
    env = {k: os.getenv(k) for k in ("RESEARCH_OFFLINE", "RESEARCH_DATA_DIR", "TRADING_ENV_FILE",
                                     "TRADING_DATA_DIR", "TRADING_LOG_DIR")}
    (out / "command.txt").write_text(
        f"commit {commit}\ncwd {os.getcwd()}\n{sys.executable} -m research.smc_study {' '.join(a_ for a_ in (argv or sys.argv[1:]))}\n"
        + "".join(f"{k}={v}\n" for k, v in env.items() if v is not None))
    fo = open(out / "stdout.txt", "w")
    fe = open(out / "stderr.txt", "w")
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old[0], fo), _Tee(old[1], fe)
    code = 1
    try:
        res = run(commit)
        write_outputs(out, res)                      # every result file first ...
        _log(f"outputs written to {out}")
        print(report(res))                           # ... then the first statistic is printed
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old
        fo.close()
        fe.close()
        (out / "exit_code.txt").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
