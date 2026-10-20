# Signal research — results (2026-10-07, corrected 2026-10-11)

> **Follow-up (2026-10-20):** the owner's "smart money" intraday playbook (liquidity sweep, market structure shift, fair value gap entry) was tested in a separate pre-registered study. All 8 short decision units were rejected. See [RESULTS_SMC.md](RESULTS_SMC.md).
> Any version that loosens the rules to get more trades (other killzones, looser displacement,
> other target or bias rules) is a new study: it needs its own pre-registration and PM review,
> and is never a re-run of this one on the same data.

**Question:** does any simple, well-known signal beat realistic trading costs on
the live bot's 50-coin universe, on data it was not designed on?

**Short answer (corrected):** no strategy passes out of sample. No price-based
signal does, and neither does perpetual-futures **funding carry**. Carry had a
real in-sample edge in the 2023–24 bull-market funding regime. Out of sample
(2025–26) funding rarely cleared its entry threshold, and the few hedges it
opened lost money after costs.

> **Correction (2026-10-11): carry's out-of-sample "confirmation" was a phantom
> hedge.** An audit found that the carry backtest let funding data alone decide
> whether a hedge could be held. MATICUSDT's spot pair was delisted on
> 2024-09-10, but Binance kept printing its perp funding at exactly 0.01% per 8h
> (the default rate, premium 0) for two more years. The exit rule (3-day mean
> < 0.01%) is strict, so a hedge opened before the delisting never closed. It
> collected riskless funding on a position that could not exist. 1,664 of
> `carry_2`'s 1,696 OOS hedged sleeve-periods were MATIC. AGIX left a similar
> phantom hedge in `carry_1` for a year after its July 2024 delisting. BLUR could
> also be "hedged" before its spot pair listed in November 2023.
>
> **Fix:** a carry sleeve is now live only in 8h windows where its spot pair
> traded (`research.study.spot_tradable`). Thresholds, grid and decision rules
> are unchanged. Flat 0.01% funding is not treated as missing, and the exit rule
> is unchanged. With the fix, `carry_2`'s OOS Sharpe goes from **+8.7 to −2.4**
> and `carry_1`'s from **+1.2 to −3.5**. Nothing is confirmed. The original
> figures are kept below, marked *superseded*.

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

Added after the audit (2026-10-11, post-hoc):
- An 8h funding window counts as tradable for a symbol only if its spot pair has
  1h closes in that window. Carry hedges are forced flat (paying the exit cost)
  outside tradable windows, and those sleeves drop out of the portfolio average.

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
| `carry_1` (corrected) | candidate, fails OOS (Sharpe −3.51) | same |
| `carry_2` (corrected) | candidate, fails OOS (Sharpe −2.36) | same |
| ~~`carry_2`~~ (superseded) | ~~passes the written rules~~ | ~~passes the written rules~~ |

Run 2 came after a sanity check found a methodology flaw. With no costs, an
equal-weight long book and an equal-weight short book *both* lost about 33% over
OOS. The cause was exact daily rebalancing: it makes volatile shorts decay like an
inverse ETF and inflates turnover. A no-trade band holds positions until the
signal changes or the drift is large. The grid and the rules stayed the same, and
only one band value was ever tried.

### Carry, corrected for spot tradability (2026-10-11)

Same pre-registered thresholds and rules, with carry masked to windows where the
spot pair trades. The two runs are identical for carry, because the band only
affects the price strategies.

| strategy | IS Sharpe | IS alpha t | OOS Sharpe | OOS Sharpe 2× costs | OOS CAGR | verdict |
|---|---|---|---|---|---|---|
| `carry_1` (0.01%/8h) | 11.91 *(was 12.18)* | 10.99 | **−3.51** *(was +1.24)* | −4.62 *(was −2.17)* | −0.2% *(was +0.1%)* | candidate, fails OOS |
| `carry_2` (0.02%/8h) | 10.00 *(was 10.13)* | 9.01 | **−2.36** *(was +8.72)* | −2.69 *(was +4.35)* | −0.0% *(was +0.2%)* | candidate, fails OOS *(was CONFIRMED)* |

- **In sample**, the phantom hedges were a small share of the ~24,000 hedged
  sleeve-periods (`carry_2`), so the IS edge stands. It comes from the 2023–24
  bull-market funding regime.
- **Out of sample**, `carry_2` held a real hedge for only 32 sleeve-periods,
  under 0.1% of sleeve-time, and lost its round-trip costs. `carry_1` was hedged
  1.6% of the time and lost money net.
- One smaller accounting fix ships with the mask. A hedge forced flat because
  its funding stops (perp delisted) now pays its exit cost. Before, that cost
  was dropped. On its own this moves `carry_2`'s unmasked OOS Sharpe from 8.73
  to 8.28. The mask accounts for the rest of the change.
- Corrected runs: `results/20261011T001031` (band 0) and `results/20261011T001040`
  (band 0.5). The superseded runs are `20261007T005803` and `20261007T010623`.

*Superseded (2026-10-07, before the audit):* **Carry passes the rules but earns
very little.** `carry_2` posted an OOS Sharpe of 8.7, and 4.4 at 2× costs, but
its OOS portfolio return was only **+0.2% a year**. The rules demanded a high
Sharpe but no minimum return, so a near-riskless, tiny-return strategy could
pass. *That Sharpe came from the phantom MATIC hedge. The lesson about a minimum
return still stands.*

### Carry with basis risk (exploratory, post-hoc: `research/carry_basis.py`)

This re-runs only the two pre-registered thresholds, adding the hedge's basis P&L
from the perp premium index. Corrected for spot tradability (2026-10-11):

| threshold (per 8h) | period | active | return on deployed capital | deployed Sharpe |
|---|---|---|---|---|
| 0.01% | IS | 36% of sleeve-time | +13.4% APR | 7.1 |
| 0.01% | OOS | 1.6% of sleeve-time | −7.9% APR | −1.6 |
| 0.02% | IS | 21% of sleeve-time | +18.3% APR | 9.3 |
| 0.02% | OOS | under 0.1% (32 sleeve-periods) | −66% APR | −2.7 |

The 0.02% OOS line collapses. Without the phantom MATIC hedge it was almost never
active, and on so few periods the entry and exit costs dominate. In sample, the
basis P&L was slightly positive (+0.6% to +0.9% APR), because premiums tend to
fall after a high-funding entry. It still lowers the Sharpe.

*Superseded (2026-10-07, before the audit):*

| threshold (per 8h) | period | active | return on deployed capital | deployed Sharpe |
|---|---|---|---|---|
| ~~0.02%~~ | ~~IS~~ | ~~21% of sleeve-time~~ | ~~+18% APR~~ | ~~9.2~~ |
| ~~0.02%~~ | ~~OOS~~ | ~~2% of sleeve-time~~ | ~~+8% APR~~ | ~~2.2~~ |

### The live bot's own entry signal (`scripts/entry_edge.py`)

Across 991 live entries since 2026-10-01, the bot's entries are indistinguishable
from shorting at random times at every horizon from 5 minutes to 4 hours. At
+60 minutes the gross return was −0.026% ± 0.044%, against about 0.36% in
round-trip costs.

## Conclusions

1. **Tuning won't make the directional bot profitable.** Its entries have no edge,
   and no price signal tested clears costs out of sample. Better exits, sizing or
   learning can't fix that. Never put real money behind it as it stands.
2. **No strategy passes out of sample, and that includes funding carry.** Carry
   earned about +18% APR on deployed capital in sample, net of costs and basis.
   That edge belongs to the 2023–24 bull-market funding regime. Out of sample it
   was rarely active and lost money when it was. The earlier "confirmed" verdict
   was a phantom delisted-MATIC hedge (see the correction above). Treat carry as
   an idea to revisit if funding returns to bull-market levels, not as a
   confirmed edge.
3. **The forward test keeps running, unchanged in its rules.**
   `research/carry_forward.py` starts 2026-10-07, and its PASS rule (≥ 20 closed
   trades, mean trade > 0, deployed APR ≥ 5%) was fixed in advance. Two fixes
   were made on 2026-10-11, when 0 trades had closed, so no verdict was affected.
   The forward test now uses the same spot-tradability mask. Deployed APR is now
   computed from closed trades, as total net return per period hedged. The old
   formula left out exit costs.

*Superseded (2026-10-07):* ~~2. **Funding carry is the one structural edge
found,** and it is regime-dependent.~~

## Lessons for future studies

- Decision rules need a minimum economic return as well as a risk-adjusted one.
- Sanity-check the execution model before running a grid. Constant-weight shorts
  decay in volatile markets.
- Restrict every leg to the hours it can actually trade. Exchanges keep printing
  data (here, default funding rates) for instruments you can no longer hold. An
  exit threshold that sits exactly on such a default value keeps a phantom
  position open forever.
- Check where a result comes from before believing it. One delisted symbol made
  up 98% of the "confirmed" strategy's active periods.

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
`/home/admin/trading_data/research/results/<timestamp>/`. Set `RESEARCH_OFFLINE=1`
to re-run on exactly the cached data, with no requests and no cache writes. The
2026-10-11 corrections were run this way, on the cache as of 2026-10-07.
