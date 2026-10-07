"""
Funding-carry FORWARD test — frozen 2026-10-07, before any forward data existed.

    .venv/bin/python -m research.carry_forward

The one strategy that survived research.study (carry_2) is re-evaluated only
on data AFTER FORWARD_START, with the parameters below fixed now. Same code
path as the backtest (strategies.funding_carry with basis P&L), fed by fresh
Binance data on each run — nothing to keep running in the background.

Frozen rules:
  universe  config.SYMBOLS perps        hedge   long spot + short perp
  enter     3-day mean funding > 0.02% per 8h (~22% APR)
  exit      3-day mean funding < 0.01% per 8h
  costs     0.31% of notional per open or close (both legs, fee + slippage)
  capital   notional = sleeve capital / 1.2 (20% perp margin)
  basis     perp premium-index changes booked while hedged

Pass / fail (decided now): evaluate once >= 20 sleeve trades have CLOSED.
  PASS if the mean net return per closed trade > 0 AND return on deployed
  capital >= 5% APR (beats parking USDT). Until then: "insufficient data".
"""

import pandas as pd

from config import SYMBOLS
from research import strategies as S
from research.data import load_funding, load_premium, utc
from research.study import DATA_START, funding_buckets

FORWARD_START = utc(2026, 10, 7)
THETA_ON = 0.0002
MIN_CLOSED_TRADES = 20
MIN_DEPLOYED_APR = 0.05
PERIODS_PER_YEAR = 1095


def main() -> None:
    fund = funding_buckets(pd.DataFrame({s: load_funding(s, DATA_START) for s in SYMBOLS}).sort_index())
    prem = pd.DataFrame({s: load_premium(s, DATA_START) for s in SYMBOLS}).sort_index().reindex(fund.index)
    ret, on = S.funding_carry(fund, THETA_ON, premium=prem, return_state=True)
    before = on[:FORWARD_START - pd.Timedelta("1ns")]
    carried = before.iloc[-1].astype(bool) if len(before) else pd.Series(False, index=on.columns)
    ret, on = ret[FORWARD_START:], on[FORWARD_START:]
    if ret.empty:
        print("No forward data yet.")
        return
    on_b = on.astype(bool)
    # Positions already open at FORWARD_START belong to the backtest period:
    # drop their remaining run so only hedges OPENED in the forward window count.
    for sym in on_b.columns[carried.reindex(on_b.columns, fill_value=False).to_numpy()]:
        first_off = on_b.index[~on_b[sym].to_numpy()]
        cut = first_off[0] if len(first_off) else on_b.index[-1] + pd.Timedelta("1ns")
        on_b.loc[:cut - pd.Timedelta("1ns"), sym] = False
        ret.loc[:cut - pd.Timedelta("1ns"), sym] = 0.0
    trades = []                                   # (symbol, open, close or None, net return)
    for sym in on_b.columns:
        s, r = on_b[sym], ret[sym].fillna(0.0)
        opened = None
        for t in s.index:
            if s[t] and opened is None:
                opened = t
            elif not s[t] and opened is not None:
                trades.append((sym, opened, t, float(r[opened:t].sum())))
                opened = None
        if opened is not None:
            trades.append((sym, opened, None, float(r[opened:].sum())))
    closed = [x for x in trades if x[2] is not None]
    active = ret[on_b].stack().dropna()
    deployed_apr = active.mean() * PERIODS_PER_YEAR if len(active) else float("nan")
    days = (ret.index[-1] - ret.index[0]).total_seconds() / 86400
    print(f"Carry forward test since {FORWARD_START.date()} ({days:.1f} days, last settlement {ret.index[-1]})")
    print(f"  sleeve-periods hedged: {int(on_b.values.sum())} of {int(ret.notna().values.sum())} "
          f"({on_b.values.sum() / max(ret.notna().values.sum(), 1):.1%})")
    print(f"  trades: {len(closed)} closed, {len(trades) - len(closed)} open")
    if len(active):
        print(f"  return on deployed capital: {deployed_apr:+.1%} APR")
    if closed:
        mean_trade = sum(x[3] for x in closed) / len(closed)
        print(f"  mean net return per closed trade: {mean_trade:+.3%}")
    open_now = [x for x in trades if x[2] is None]
    latest = fund.rolling(9, min_periods=9).mean().iloc[-1]
    print("  hedged now: " + (", ".join(f"{x[0]} (since {x[1]:%m-%d %H:%M})" for x in open_now) or "none"))
    print("  top 3-day funding now: " + ", ".join(f"{k} {v * 100:.4f}%/8h" for k, v in latest.nlargest(5).items()))
    if len(closed) < MIN_CLOSED_TRADES:
        print(f"  verdict: insufficient data ({len(closed)}/{MIN_CLOSED_TRADES} closed trades)")
    else:
        ok = (sum(x[3] for x in closed) / len(closed) > 0) and deployed_apr >= MIN_DEPLOYED_APR
        print(f"  verdict: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
