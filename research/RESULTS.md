# Signal research — results (2026-10-07)

**Question:** does any simple, well-known signal beat realistic trading costs on
the live bot's 50-coin universe, on data it was not designed on?

**Short answer:** no price-based signal does. The only real edge found is
perpetual-futures **funding carry**. It is market-neutral and structural, but it
depends on the market regime: strong in euphoric bull markets and mostly idle now.

## Setup (pre-registered in commit `46529d2`, before any result was seen)

- **Data:** Binance spot 1h candles, 2022-10 → 2026-10, for all 50 symbols. That
  includes the 4 since-delisted pairs, kept up to their delisting date. Perp
  funding rates and premium index come from Binance futures.
- **Periods:** a design period (IS) of 2023-01-01 → 2025-04-01, and a held-out
  period (OOS) of 2025-04-01 → 2026-10-07.
- **Costs:**
  - 0.10% fee plus 0.08% slippage per fill, the same as the bot's paper model.
  - Shorts pay 0.03% a day in borrow interest.
  - Every result is also stress-tested at 2× costs.
- **Strategies:** a grid of 19, covering time-series momentum (long/short and
  long-only), cross-sectional momentum and reversal, Donchian breakout, 1h
  mean reversion, and funding carry.
- **Decision rules:**
  - A strategy becomes a candidate if its IS Sharpe is ≥ 1 and its alpha
    t-stat against the equal-weight market is ≥ 2.
  - A candidate is confirmed if, on OOS data, its Sharpe is ≥ 0.75, its alpha
    is above 0, its Sharpe at 2× costs is ≥ 0.5, and a neighbouring parameter
    setting is also stable.

The data-handling decisions were also fixed before any results:
- `7e4fd17`: funding settlements are summed into 8h windows labelled by their end.
- `1a9de8f`: a daily bar is kept if at least 22 of its 24 hours are present.

## Market context (OOS)

The OOS period was a bear market for alts:
- **Equal-weight alt basket:** −20% to −26% a year, with a drawdown of about −70%.
- **BTC:** roughly flat (+2.5% a year) with a −53% drawdown.

## Results

| | Run 1: exact daily rebalancing | Run 2: no-trade band 0.5 |
|---|---|---|
| Price-based strategies (17) | all rejected | all rejected |
| Best IS alpha t-stat | 1.46 (`tsmom_lo_28`) | 1.59 (`tsmom_lo_28`) |
| Best OOS Sharpe | +0.10 (`xsmom_14`) | +0.32 (`xsmom_14`) |
| Reversal, 1h mean reversion | −65% to −100%/yr (costs) | same |
| `carry_2` | passes the written rules | passes the written rules |

Run 2 came after a sanity check found a methodology flaw. With no costs, an
equal-weight long book and an equal-weight short book *both* lost about 33% over
OOS. The cause was exact daily rebalancing: it makes volatile shorts decay like an
inverse ETF and inflates turnover. A no-trade band holds positions until the
signal changes or the drift is large. The grid and the rules stayed the same, and
only one band value was ever tried.

**Carry passes the rules but earns very little.** `carry_2` posted an OOS Sharpe
of 8.7, and 4.4 at 2× costs, but its OOS portfolio return was only **+0.2% a
year**. The rules demanded a high Sharpe but no minimum return, so a near-riskless,
tiny-return strategy could pass.

### Carry with basis risk (exploratory, post-hoc: `research/carry_basis.py`)

This re-runs only the two pre-registered thresholds, adding the hedge's basis P&L
from the perp premium index:

| threshold (per 8h) | period | active | return on deployed capital | deployed Sharpe |
|---|---|---|---|---|
| 0.02% | IS | 21% of sleeve-time | ~+18% APR | 9.2 |
| 0.02% | OOS | 2% of sleeve-time | ~+8% APR | 2.2 |

The basis P&L was slightly positive (+0.6% to +1.1% APR), because premiums tend to
fall after a high-funding entry. It still lowers the Sharpe.

### The live bot's own entry signal (`scripts/entry_edge.py`)

Across 991 live entries since 2026-10-01, the bot's entries are indistinguishable
from shorting at random times at every horizon from 5 minutes to 4 hours. At
+60 minutes the gross return was −0.026% ± 0.044%, against about 0.36% in
round-trip costs.

## Conclusions

1. **Tuning won't make the directional bot profitable.** Its entries have no edge,
   and no price signal tested clears costs out of sample. Better exits, sizing or
   learning can't fix that. Never put real money behind it as it stands.
2. **Funding carry is the one structural edge found,** and it is regime-dependent.
   The forward test is frozen in `research/carry_forward.py`: it starts
   2026-10-07, and its PASS rule (≥ 20 closed trades, mean trade > 0, deployed APR
   ≥ 5%) was fixed in advance.

## Lessons for future studies

- Decision rules need a minimum economic return as well as a risk-adjusted one.
- Sanity-check the execution model before running a grid. Constant-weight shorts
  decay in volatile markets.

## Caveats

- **Survivorship:** the universe was chosen in 2026, although delisted pairs are
  included up to their delisting.
- **Fills:** close-to-close with fixed slippage. Real slippage on small alts can be
  worse.
- **Carry model:** it ignores margin top-ups and liquidation handling on the perp
  leg, and exchange risk.

## Re-running

```
.venv/bin/python -m research.data_check          # refresh data cache + quality report
.venv/bin/python -m research.study [--band 0.5]  # the pre-registered grid
.venv/bin/python -m research.carry_basis         # carry economics incl. basis (exploratory)
.venv/bin/python -m research.carry_forward       # frozen forward test of carry
.venv/bin/python scripts/entry_edge.py           # live bot's entries vs random
```

Each study run writes `report.txt`, `summary.json` and `daily_equity.csv` under
`/home/admin/trading_data/research/results/<timestamp>/`.
