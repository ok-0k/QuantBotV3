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

Corrections (2026-10-11, audit; at that point 0 trades had closed, so no
verdict was affected and the frozen rules above are unchanged):
  * Return on deployed capital is computed from the CLOSED trades, as
    sum(closed-trade net returns) / sum(periods hedged) x 1095. It used to be
    the mean return over hedged periods of all trades, which left out every
    exit cost (booked on the first un-hedged period) and mixed in open trades.
  * A sleeve is only live while its spot pair trades
    (research.study.spot_tradable): a hedge cannot outlive its spot leg, as
    the phantom MATICUSDT hedge did in the backtest (see RESULTS.md).

Universe (2026-10-21, no rule change): "config.SYMBOLS" above is now the
frozen copy research.universe.RESEARCH_SYMBOLS (the same 50, same order), so
the live bot dropping its delisted pairs leaves this test's universe unchanged.
"""

import pandas as pd

from research import strategies as S
from research.data import load_funding, load_premium, utc
from research.study import DATA_START, funding_buckets, load_spot_tradable
from research.universe import RESEARCH_SYMBOLS as SYMBOLS

FORWARD_START = utc(2026, 10, 7)
THETA_ON = 0.0002
MIN_CLOSED_TRADES = 20
MIN_DEPLOYED_APR = 0.05
PERIODS_PER_YEAR = 1095


def trades(ret: pd.DataFrame, on: pd.DataFrame) -> list[tuple]:
    """Sleeve trades as (symbol, open, close or None, net return, periods hedged).

    A trade's net return runs from its opening period (income less entry
    cost) through its closing period, the first un-hedged one, where the exit
    cost is booked; an open trade has paid no exit cost yet.
    """
    out = []
    for sym in on.columns:
        s, r = on[sym].astype(bool), ret[sym].fillna(0.0)
        opened, n = None, 0
        for t in s.index:
            if s[t]:
                if opened is None:
                    opened, n = t, 0
                n += 1
            elif opened is not None:
                out.append((sym, opened, t, float(r[opened:t].sum()), n))
                opened = None
        if opened is not None:
            out.append((sym, opened, None, float(r[opened:].sum()), n))
    return out


def deployed_apr(closed: list[tuple]) -> float:
    """Return on deployed capital of closed trades: total net return (entry
    AND exit costs included) per period hedged, annualised."""
    periods = sum(x[4] for x in closed)
    return sum(x[3] for x in closed) / periods * PERIODS_PER_YEAR if periods else float("nan")


def main() -> None:
    fund = funding_buckets(pd.DataFrame({s: load_funding(s, DATA_START) for s in SYMBOLS}).sort_index())
    spot_ok = load_spot_tradable(SYMBOLS, DATA_START)
    fund = fund[fund.index <= spot_ok.index.max()]               # no spot data yet -> unknown
    prem = pd.DataFrame({s: load_premium(s, DATA_START) for s in SYMBOLS}).sort_index().reindex(fund.index)
    ret, on = S.funding_carry(fund, THETA_ON, premium=prem, return_state=True, tradable=spot_ok)
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
    all_trades = trades(ret, on_b)
    closed = [x for x in all_trades if x[2] is not None]
    apr = deployed_apr(closed)
    days = (ret.index[-1] - ret.index[0]).total_seconds() / 86400
    print(f"Carry forward test since {FORWARD_START.date()} ({days:.1f} days, last settlement {ret.index[-1]})")
    print(f"  sleeve-periods hedged: {int(on_b.values.sum())} of {int(ret.notna().values.sum())} "
          f"({on_b.values.sum() / max(ret.notna().values.sum(), 1):.1%})")
    print(f"  trades: {len(closed)} closed, {len(all_trades) - len(closed)} open")
    if closed:
        mean_trade = sum(x[3] for x in closed) / len(closed)
        print(f"  return on deployed capital (closed trades, after all costs): {apr:+.1%} APR")
        print(f"  mean net return per closed trade: {mean_trade:+.3%}")
    open_now = [x for x in all_trades if x[2] is None]
    spot_now = spot_ok.reindex(index=fund.index, columns=fund.columns).eq(True).iloc[-1]
    latest = fund.rolling(9, min_periods=9).mean().iloc[-1][spot_now]
    print("  hedged now: " + (", ".join(f"{x[0]} (since {x[1]:%m-%d %H:%M})" for x in open_now) or "none"))
    print("  top 3-day funding now: " + ", ".join(f"{k} {v * 100:.4f}%/8h" for k, v in latest.nlargest(5).items()))
    if len(closed) < MIN_CLOSED_TRADES:
        print(f"  verdict: insufficient data ({len(closed)}/{MIN_CLOSED_TRADES} closed trades)")
    else:
        ok = (sum(x[3] for x in closed) / len(closed) > 0) and apr >= MIN_DEPLOYED_APR
        print(f"  verdict: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
