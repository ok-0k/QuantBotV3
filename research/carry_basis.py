"""
EXPLORATORY (post-hoc, after research.study results were seen): funding-carry
economics with the hedge's basis risk included.

    .venv/bin/python -m research.carry_basis

The pre-registered study modelled carry as funding income minus switching
costs, ignoring basis (perp vs spot) drift — which inflated its Sharpe. This
re-runs ONLY the two pre-registered thresholds (no new tuning) with the perp
premium index added, and reports the economics that matter: how often capital
is deployed and what it earns while deployed.
"""

import math

import numpy as np
import pandas as pd

from config import SYMBOLS
from research import strategies as S
from research.data import load_funding, load_premium
from research.study import DATA_START, IS_START, OOS_START, funding_buckets

PERIODS_PER_YEAR = 1095


def main() -> None:
    fund = funding_buckets(pd.DataFrame({s: load_funding(s, DATA_START) for s in SYMBOLS}).sort_index())
    prem = pd.DataFrame({s: load_premium(s, DATA_START) for s in SYMBOLS}).sort_index()
    prem = prem.reindex(fund.index)
    print(f"premium coverage on the funding grid: {prem.notna().values.sum() / fund.notna().values.sum():.1%}")
    print("\nper-period returns are per sleeve (one symbol's capital); 'deployed' = while the hedge is on\n")
    print(f"{'theta/8h':>9} {'basis':>6} {'period':>6} {'active':>7} {'avg on':>7} {'deployed APR':>13} "
          f"{'basis APR':>10} {'deployed Sh':>12} {'portfolio CAGR':>15}")
    for th in (0.0001, 0.0002):
        base = S.funding_carry(fund, th)
        with_b = S.funding_carry(fund, th, premium=prem)
        on = base.notna() & (base != 0)
        for label, sl in (("no", base), ("yes", with_b)):
            for nm, a, b in (("IS", IS_START, OOS_START), ("OOS", OOS_START, None)):
                w, o = sl[a:b], on[a:b]
                act = w[o].stack()
                deployed_apr = act.mean() * PERIODS_PER_YEAR
                sh = act.mean() / act.std() * math.sqrt(PERIODS_PER_YEAR) if act.std() > 0 else float("nan")
                basis_apr = ((with_b - base)[a:b][o].stack().mean() * PERIODS_PER_YEAR) if label == "yes" else 0.0
                port = w.mean(axis=1).fillna(0.0)
                years = len(port) / PERIODS_PER_YEAR
                cagr = (1 + port).prod() ** (1 / years) - 1
                print(f"{th*100:8.2f}% {label:>6} {nm:>6} {o.values.sum() / w.notna().values.sum():7.1%} "
                      f"{o.sum(axis=1).mean():7.1f} {deployed_apr:+12.1%} {basis_apr:+10.1%} {sh:12.2f} {cagr:+14.2%}")


if __name__ == "__main__":
    main()
