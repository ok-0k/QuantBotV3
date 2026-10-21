# Perp cost re-score of past studies — EXPLORATORY — NOT EVIDENCE

**Date:** 2026-10-21. **Branch:** `research/perp-rescore` (from `86ee599`).
**Script:** `research/explore_perp.py`. **Tests:** `tests/test_research_explore_perp.py`.

> **EXPLORATORY — NOT EVIDENCE.** Every out-of-sample period used here has
> already been seen, in RESULTS.md, RESULTS_FUNDING.md and RESULTS_SMC.md.
> The perp cost model was applied after the spot results were known. Nothing
> here confirms, rejects or re-opens any pre-registered verdict, and nothing
> here changes the bot. The only question is whether any idea deserves a
> **new, pre-registered forward test** if it were traded on Binance USDT-M
> perpetuals instead of spot.

## Short answer

- **108 rows screened:**
  - 48 crowding configs;
  - 17 price strategies × 2 no-trade bands (34 rows);
  - 2 carry configs;
  - 24 SMC units (8 short decision units, plus 16 long/both units for context).
- **48 pass screen (a)** (gross > 0 in IS and in OOS), and **29 pass (a) and
  (b)** (at least 30 events per 8 weeks). Of those 29, 21 are also
  non-negative OOS at perp cost. Most of the 29 are the same few ideas at
  neighbouring parameters.
- **Two forward-test candidates are proposed** (details below):
  1. `fund_lvl_ls_k3_24h`: funding level, long/short, rebalanced daily.
  2. `xsmom_14` (band 0.5): 14-day cross-sectional momentum, dollar-neutral.
- **SMC fails (b)** (at most 6.2 OOS trades per 8 weeks), and so does
  **carry** (at most 5.9 events per 8 weeks).
- **Moving to perps is the owner's decision.** The bot trades spot today. Any
  forward test of these ideas needs a new perp rebalancing executor (paper
  first), with its own review.

## Cost model (fixed before any re-scored number was computed)

| level | fees | slippage | borrow | funding |
|---|---|---|---|---|
| `zero` (gross) | none | none | none | none (price P&L only) |
| `perp` | taker 0.05% per fill; maker 0.02% only where the original model used a limit fill (SMC entries, TP1/TP2) | 0.02% per market-type fill (an **assumption** for liquid perps) | none | actual 8h (or 4h) settlement rates for every settlement a position spans |
| `perp_s5` | as `perp` | 0.05% per market-type fill | none | as `perp` |

- **Fills.** The original spot model was 0.10% fee + 0.08% slippage per fill
  plus 0.03% a day borrow on shorts. Every fill in the crowding and price
  books is a market-type fill, so it pays taker + slippage: 0.07% (0.10% at
  `perp_s5`) instead of 0.18%.
- **SMC.**
  - Limit entries and TP1/TP2 exits pay maker 0.02%, with no slippage.
  - Stops, breakeven, time and session exits pay taker 0.05% + slippage.
- **Carry** is long spot / short perp, so its spot leg keeps the spot cost.
  - Only its perp leg moves to 0.05% + 0.02% (0.05%).
  - A switch costs 0.25% (0.28%) of notional instead of 0.31%.
  - Its `zero` level has no switch cost; funding is its income at every
    level.
- **Funding timing.**
  - A position held over a bar pays `weight × rate` for each settlement in
    that bar. A long pays a positive rate; a short receives it.
  - The crowding books fill at 01:00, 09:00 and 17:00 UTC, so no fill sits on
    a settlement.
  - For SMC, funding is charged on the entry notional for settlements
    strictly between the fill-bar open and the exit. It is charged on full
    size even after a `pbe` partial, a small overstatement. It is
    economically zero, because the session ends at 16:00–17:00 UTC.
- **Prices stay Binance spot.** The perp basis is ignored. For the
  funding-level books the diagnostics below estimate it, and it was a gain
  (+1.5% to +4.4% a year), so ignoring it understates them.

## Universe: symbols without a perp

All 50 symbols have a USDT-M perp at some point. A symbol-time counts as
perp-listed from 8h before its first funding settlement in the cache up to its
last. Where the spot pair trades but the perp does not, the symbol is
**excluded** for that stretch:

| symbol | no perp | effect |
|---|---|---|
| FETUSDT | before 2023-01-16 | first 2 weeks of IS |
| STXUSDT | before 2023-02-21 | first 7 weeks of IS |
| ILVUSDT | before 2023-11-10 | first 10 months of IS |
| MKRUSDT | after 2025-09-08 (spot to 2025-09-15) | last week before its spot delisting |

- AGIX, OCEAN and MATIC show perp funding long after their spot delisting.
  Spot governs there, as in the studies, so the phantom-hedge problem cannot
  recur.
- In the **price strategies** the exclusion is applied to the tradable mask.
- In **SMC**, 6 of 1,454 setups were dropped.
- The **crowding** books need funding or premium data to rank a name, so
  their eligibility is used unchanged. The one leak is MKR, whose premium
  index kept printing after 2025-09-08: `prem_*` books could hold MKR for at
  most 7 days. That is immaterial, and no `prem_*` row is proposed.

## Method and checks

- **Unchanged code.** The committed code is reused without changes: signals,
  weights, schedules, eligibility, `research.backtest.metrics`, the SMC
  detectors, order simulation and walk, and the carry sleeve model.
- **Only two pieces are new:**
  - `run_perp` is `research.backtest.run` copied line for line, plus a
    funding term and sign-change counters. With funding off and the same
    costs it equals `backtest.run` exactly. This is checked in the run and in
    the tests.
  - `perp_net_r` is `research.smc.net_r` with maker/taker fees and funding
    instead of one fee and borrow.
- **Reproduction checks.** Each study was first reproduced at its original
  base cost on its original universe, and the run asserts equality with the
  stored outputs. All checks passed:
  - **Crowding:** the data fingerprint is `4a7bd3e0…`. Base IS/OOS alpha and
    zero-cost OOS alpha were checked for all 48 configs.
  - **Price strategies:** base IS/OOS Sharpe and alpha for all 17 × 2 bands
    and both carry configs, against `results/20261011T001031` and
    `results/20261011T001040`.
  - **SMC:** base IS/OOS trade counts and expectancy for all 24
    config × side units, per-symbol scope, against
    `runs/smc_run1_20261020T060848Z`.
- **One carry subtlety.** The stored carry runs saw funding settlements up to
  the last spot 8h window (2026-10-07 08:00 UTC). Capping funding at the last
  1h bar instead moved `carry_1`'s OOS Sharpe by 0.001, so the cap now
  follows `research.study.main`.
- **"Gross" is price P&L only.** It has no fees and no funding. "Alpha" is
  annualised alpha against each study's own equal-weight benchmark. For SMC
  the measure is expectancy in R per trade.

## The screen, as applied

- **(a)** Gross alpha (gross expectancy for SMC) must be > 0 in IS **and** in
  OOS.
- **(b)** At least 30 events per 8 weeks, at the OOS rate:
  - **SMC:** fills.
  - **Portfolio books:** signal-driven rebalances, meaning bars on which some
    position opens, closes or flips. Pure drift trades do not count.
  - So a 72h book can make at most 18.7 rebalances per 8 weeks and fails
    (b). A daily book can make at most 56.
- **Perp OOS ≥ 0** is reported but never decisive, because the OOS data have
  been seen.
- **Ranking.** Survivors are ranked by the OOS alpha t-stat at `perp` cost.
- **A looser reading of (b)** would count symbol-level entries instead of
  rebalances. It would also pass 8 of the 72h crowding books. None of them
  ranks above the proposals (best OOS perp t +1.32, `prem_lvl_ls_k3_72h`).

## Best configurations across studies

| config | study | IS gross | OOS gross | IS perp | OOS perp | OOS perp (0.05% slip) | OOS perp t | events/8wk | screen |
|---|---|---|---|---|---|---|---|---|---|
| `fund_lvl_ls_k3_24h` | crowding | +1.4% | +52.1% | +6.1% | +62.2% | +54.5% | +2.02 | 53 | pass |
| `fund_lvl_ls_k3_8h` | crowding | +0.7% | +42.8% | -0.5% | +54.9% | +43.9% | +1.76 | 123 | pass |
| `xsmom_14@band0.5` | price | +24.3% | +41.3% | +13.9% | +31.5% | +26.6% | +1.75 | 55 | pass |
| `fund_lvl_ls_k5_24h` | crowding | +12.5% | +20.0% | +14.1% | +25.6% | +18.8% | +1.18 | 55 | pass |
| `xsmom_28@band0.5` | price | +32.0% | +23.0% | +24.4% | +16.6% | +13.5% | +0.89 | 54 | pass |
| `tsmom_ls_28@band0.5` | price | +66.7% | +23.9% | +58.0% | +18.6% | +16.3% | +0.47 | 49 | pass |
| `fund_chg_ls_k3_24h` | crowding | +26.9% | +49.8% | +0.0% | +28.6% | +14.4% | +0.92 | 56 | pass |
| `xsrev_4h@band0` | price | +33.5% | +38.1% | -204.7% | -200.2% | -302.1% | -11.49 | 336 | pass (perp < 0) |
| `carry_1` | carry | +5.8% | +0.1% | +4.6% | -0.1% | -0.1% | -2.96 | 6 | fail (b) |
| `smc_5m_tp2_nobias:short` | smc | +0.247R | +0.549R | +0.169R | +0.428R | +0.391R | +1.63 | 6 | fail (b) |

Alpha is annualised; SMC is R per trade. The full tables for all 108 rows are
in the appendix.

## Proposed forward-test candidates (2)

### 1. `fund_lvl_ls_k3_24h`: funding level, long/short, daily (rank 1 of 29)

**What it is.** Once a day, the book shorts the 3 coins with the highest
3-day mean funding and buys the 3 with the lowest, at gross 1 and net 0.

**What the numbers show.**
- On spot it failed because costs (52% a year OOS) ate a gross OOS alpha of
  +52%.
- On perps two things change: the cost falls to about 18% a year, and the
  book **collects funding on both legs**. It received +19% a year IS and +31%
  a year OOS.
- Perp alpha is +6.1% IS and +62.2% OOS (t +2.02), and +54.5% at 0.05%
  slippage.

**Weaknesses.**
- Its price-only IS alpha is essentially zero (+1.4%, t +0.1). It passes (a)
  by a hair, so in sample the case rests on the funding income.
- OOS funding is concentrated on the long leg in AXS and ENJ, two
  4h-settlement coins with deeply negative funding. Most of it came in
  Jan–Apr 2026 (monthly funding +5% to +10% against +0.3% to +2.4%
  otherwise).
- Without AXS and ENJ the OOS perp alpha is still +31.7% (t +1.16), and the
  IS perp alpha rises to +18.6%.
- The estimated basis is a gain, not a loss: +1.9% a year IS and +4.4% OOS.

**Siblings.** The whole `fund_lvl_ls` family is positive at perp cost in
OOS (+20% to +62%), and 5 of its 6 books are positive in IS. The 10-name
sibling, `fund_lvl_ls_k5_24h`, is more diversified (IS +14.1%, OOS +25.6%,
t +1.18) but leans on AXS and ENJ even more (+7.0% OOS without them).

**Forward test.**
- A pre-registration should fix one k in advance, or treat both as decision
  units with a corrected bar.
- It should use a funding-concentration rule.
- Slippage needs care: the long leg sits in squeezed, less liquid names,
  where 0.02% is optimistic.
- It makes about 53 rebalances per 8 weeks.

### 2. `xsmom_14` at band 0.5: cross-sectional momentum, daily (rank 3 of 29)

**What it is.** Each day the book is long the top 20% and short the bottom
20% of 14-day returns, skipping the last day, at gross 1 and net 0.

**What the numbers show.**
- It was already the best OOS Sharpe in the spot study (+0.32), but it was
  never an IS candidate.
- Gross alpha is +24.3% IS (t +1.57) and +41.3% OOS (t +2.31). Costs fall
  from about 35% a year (spot) to about 11% a year (perp).
- Perp alpha is +13.9% IS and +31.5% OOS (t +1.75), and +26.6% at 0.05%
  slippage.
- Funding is a small +1.4% a year.

**Concentration.**
- Unlike candidate 1, the OOS result is spread out. The top symbol (ARB)
  carries about 20% of the OOS price P&L, and dropping it leaves +30.5%
  (t +1.67).
- 6 of 7 OOS quarters are positive.
- The 28-day neighbour also passes (perp IS +24.4%, OOS +16.6%). The 7-day
  one fails (a): its IS gross is −2.2%.

**Weakness.** The IS t-stat is modest, and 2025–26 was a strong dispersion
regime for alts.

**Forward test.** About 55 rebalances per 8 weeks. This is the cleaner of the
two ideas: it does not depend on funding and has no single-symbol driver.

### Why not a third

The rank-2 row, `fund_lvl_ls_k3_8h`, is candidate 1 rebalanced every 8h, and
its IS perp alpha is negative (−0.5%). After that, the next distinct ideas
are weaker:
- **`fund_chg_ls_k3_24h` (rank 7):** its IS perp alpha is +0.0%, and its
  family is incoherent. 3 of its 5 siblings are negative gross in IS or OOS.
- **The time-series momentum family (`tsmom_*`):** large IS gross alphas
  (+23% to +67%) from the 2023–24 trend, but OOS perp t at most +0.50. It is
  also partly beta: the long-only versions have β ≈ 0.5.

Proposing a third row would mostly add a multiple-comparisons tax. The same
goes for the 72h crowding books, which only pass under the loose reading of
(b).

### Not candidates, for the record

- **`xsrev_4h`.** Gross positive in both periods (+33% / +38%), but it turns
  over about 3,400× its capital a year. That costs about 240% a year even at
  perp rates, so perp OOS is −200%. It would need maker execution, which is a different
  strategy.
- **SMC.**
  - Perp costs cut the cost per trade from about 0.2–0.4R to about 0.06–0.12R.
  - The 5m short units are then positive in IS and OOS: `smc_5m_tp2_nobias`
    short is +0.169R IS and +0.428R OOS.
  - But SMC fails (b): at most 6.2 OOS trades per 8 weeks, and 15.8 for
    long+short together.
  - The concentration problems in RESULTS_SMC.md remain (one symbol and one
    day carry the OOS result).
- **Carry.** `carry_1` passes (a), but OOS it switched only 5.9 times per 8
  weeks, and its perp OOS alpha is still negative.

## Multiple comparisons

- **The scale.** 108 rows were screened, all on data whose OOS has been seen,
  and under a cost model chosen after the spot verdicts. With that many
  correlated tries, a best OOS t of +2.02 is what chance alone would
  produce. The Bonferroni one-sided 5% bar for 108 tests is about t = 3.3,
  and no row reaches it.
- **Cumulative count.** The four pre-registered studies have now had 75
  decision units fail. This re-score adds 108 more looks.
- **What a candidate earns.** A candidate here earns only the right to a new
  pre-registration, judged on data nobody has seen yet. It earns nothing
  else.
- **Power.** 8 weeks of daily rebalances gives about 56 observations. That
  is enough to catch a broken executor or a gross cost mismatch, but a ±30%
  a year alpha will not reach significance in 8 weeks. The pre-registration
  should say what an 8-week result can and cannot decide.

## Caveats of the perp model

- **Slippage.** 0.02% per fill is an assumption for liquid perps. The
  universe includes small alts (AXS, ENJ, FLOW, ILV, GMT), where real
  slippage can be several times that. `perp_s5` shows the sensitivity only
  up to 0.05%.
- **Fees.** Fees assume Binance's base tier (taker 0.05%, maker 0.02%), with
  no BNB discount and no VIP tier.
- **Basis.** Spot prices stand in for perp prices; the perp basis is ignored.
  The funding-level diagnostics estimate the basis from the 8h premium index
  on the held weights, and it was a gain there.
- **Not modelled:**
  - margin, liquidation, ADL, exchange or counterparty risk;
  - funding caps, and changes in settlement interval;
  - the 1h fill lag is the studies' own.
- **Survivorship.** The universe is still the 2026 list, as in every study.

## Owner decision

The bot trades spot. Trading either candidate means perps, both longs and
shorts, with margin. That is a different product, a different risk profile
and a new executor. It is the owner's decision. Even then, the next step is
a pre-registered **paper** forward test with a new executor, never a change
to the live bot on the strength of this re-score.

## Runtime

Wall clock was about 14 minutes on the Pi, under `nice -n 10`. The three
parts ran in parallel, then the diagnostics:

| part | runtime |
|---|---|
| crowding | 743 s |
| price + carry | 37 s |
| SMC | 272 s |
| diagnostics | 75 s |

## Re-running

```
S=<scratch>
TRADING_ENV_FILE=/nonexistent DISCORD_WEBHOOK_URL= RESEARCH_OFFLINE=1 \
TRADING_DATA_DIR=$S/td TRADING_LOG_DIR=$S/tl nice -n 10 \
/home/admin/trading_bot/.venv/bin/python -m research.explore_perp {crowding|price|smc|diag} --out $S/out
... -m research.explore_perp report --out $S/out    # report.md (the appendix) + screen.csv
```

Each part refuses to run without `RESEARCH_OFFLINE=1`, and asserts that it
reproduces the stored base-cost results before re-scoring.

## Appendix: all 108 rows (generated by `explore_perp report`)

Columns for the portfolio books:
- **IS / OOS gross α:** zero cost, price P&L only.
- **perp α:** includes funding.
- **OOS funding/yr and OOS cost/yr:** at `perp`.
- **events/8wk:** signal-driven rebalances.
- **entries/8wk:** symbol-level positions opened or flipped.

For SMC, the numbers are R per trade and the units use per-symbol limits.

### Crowding (48 configs)

| config | IS gross α | OOS gross α | IS perp α | OOS perp α | OOS perp α t | OOS perp s5 α | OOS funding/yr | OOS cost/yr | events/8wk | entries/8wk | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fund_lvl_so_k3_8h` | +17.4% | +44.4% | +5.7% | +17.5% | +0.680 | +3.3% | +6.6% | +33.5% | 100.9 | 127.8 | ✓ | ✓ |
| `fund_lvl_so_k3_24h` | -1.4% | +41.1% | -4.9% | +24.7% | +1.025 | +15.1% | +6.1% | +22.5% | 48.4 | 89.4 | ✗ | ✓ |
| `fund_lvl_so_k3_72h` | +17.7% | +3.8% | +17.0% | -3.6% | -0.133 | -9.0% | +5.2% | +12.7% | 18.4 | 53.8 | ✓ | ✗ |
| `fund_lvl_so_k5_8h` | +16.3% | +27.0% | +6.9% | +3.7% | +0.182 | -8.4% | +6.1% | +29.5% | 122.0 | 173.8 | ✓ | ✓ |
| `fund_lvl_so_k5_24h` | +14.0% | +10.5% | +12.4% | -3.5% | -0.175 | -11.8% | +5.7% | +19.6% | 53.1 | 118.0 | ✓ | ✓ |
| `fund_lvl_so_k5_72h` | +22.7% | +0.3% | +26.7% | -6.5% | -0.313 | -11.7% | +4.9% | +11.7% | 18.6 | 72.7 | ✓ | ✗ |
| `fund_lvl_ls_k3_8h` | +0.7% | +42.8% | -0.5% | +54.9% | +1.762 | +43.9% | +36.1% | +25.9% | 122.6 | 188.4 | ✓ | ✓ |
| `fund_lvl_ls_k3_24h` | +1.4% | +52.1% | +6.1% | +62.2% | +2.021 | +54.5% | +31.3% | +18.1% | 53.2 | 134.6 | ✓ | ✓ |
| `fund_lvl_ls_k3_72h` | +16.9% | +5.2% | +24.6% | +24.6% | +0.847 | +20.6% | +27.4% | +10.7% | 18.6 | 82.4 | ✓ | ✗ |
| `fund_lvl_ls_k5_8h` | +9.1% | +30.2% | +5.6% | +32.0% | +1.469 | +21.9% | +25.0% | +23.5% | 144.0 | 276.3 | ✓ | ✓ |
| `fund_lvl_ls_k5_24h` | +12.5% | +20.0% | +14.1% | +25.6% | +1.183 | +18.8% | +21.2% | +15.9% | 55.2 | 191.1 | ✓ | ✓ |
| `fund_lvl_ls_k5_72h` | +13.3% | +10.3% | +18.6% | +20.0% | +0.903 | +15.7% | +19.2% | +9.8% | 18.6 | 119.6 | ✓ | ✗ |
| `fund_chg_so_k3_8h` | +2.8% | -6.9% | -42.5% | -70.8% | -1.976 | -91.4% | -16.1% | +48.3% | 118.5 | 160.2 | ✗ | ✓ |
| `fund_chg_so_k3_24h` | +34.0% | +29.6% | +6.2% | -15.5% | -0.446 | -27.8% | -16.3% | +28.9% | 50.7 | 96.4 | ✓ | ✓ |
| `fund_chg_so_k3_72h` | +6.1% | -20.1% | -7.8% | -57.2% | -1.248 | -62.7% | -22.2% | +12.8% | 18.3 | 41.0 | ✗ | ✗ |
| `fund_chg_so_k5_8h` | -5.3% | -45.9% | -44.7% | -101.2% | -4.133 | -121.0% | -9.3% | +46.2% | 147.1 | 254.7 | ✗ | ✓ |
| `fund_chg_so_k5_24h` | -13.5% | -4.1% | -37.8% | -43.3% | -1.783 | -55.8% | -10.2% | +29.2% | 55.4 | 160.8 | ✗ | ✓ |
| `fund_chg_so_k5_72h` | -12.9% | -10.5% | -23.0% | -36.4% | -1.306 | -41.7% | -13.3% | +12.2% | 18.6 | 65.9 | ✗ | ✗ |
| `fund_chg_ls_k3_8h` | -12.5% | -19.1% | -57.3% | -56.7% | -2.118 | -79.9% | +16.6% | +54.5% | 159.0 | 363.2 | ✗ | ✓ |
| `fund_chg_ls_k3_24h` | +26.9% | +49.8% | +0.0% | +28.6% | +0.924 | +14.4% | +12.1% | +33.3% | 55.8 | 222.4 | ✓ | ✓ |
| `fund_chg_ls_k3_72h` | +18.5% | +3.5% | +7.7% | -6.2% | -0.197 | -12.1% | +4.1% | +13.8% | 18.6 | 90.4 | ✓ | ✗ |
| `fund_chg_ls_k5_8h` | -8.7% | -19.2% | -51.5% | -59.0% | -2.961 | -80.8% | +11.1% | +51.0% | 165.9 | 563.5 | ✗ | ✓ |
| `fund_chg_ls_k5_24h` | -5.0% | +12.3% | -32.2% | -12.2% | -0.602 | -26.0% | +7.9% | +32.5% | 56.0 | 358.5 | ✗ | ✓ |
| `fund_chg_ls_k5_72h` | +3.9% | +14.5% | -7.4% | +4.2% | +0.179 | -1.5% | +2.8% | +13.2% | 18.6 | 145.5 | ✓ | ✗ |
| `prem_lvl_so_k3_8h` | +15.6% | -13.7% | -9.8% | -43.8% | -1.790 | -57.7% | +3.0% | +32.6% | 94.3 | 106.7 | ✗ | ✓ |
| `prem_lvl_so_k3_24h` | -3.8% | -18.0% | -13.0% | -34.0% | -1.184 | -42.1% | +2.3% | +18.7% | 44.5 | 61.0 | ✗ | ✓ |
| `prem_lvl_so_k3_72h` | -0.1% | +24.5% | -0.6% | +15.8% | +0.559 | +11.5% | +1.2% | +9.8% | 18.5 | 32.0 | ✗ | ✗ |
| `prem_lvl_so_k5_8h` | +4.8% | -39.0% | -16.8% | -67.3% | -3.444 | -80.7% | +3.5% | +32.1% | 129.4 | 175.0 | ✗ | ✓ |
| `prem_lvl_so_k5_24h` | +8.0% | -37.5% | +1.2% | -52.5% | -2.265 | -60.5% | +2.9% | +18.3% | 52.7 | 99.4 | ✗ | ✓ |
| `prem_lvl_so_k5_72h` | +0.3% | -2.0% | +0.9% | -9.9% | -0.483 | -14.5% | +2.4% | +10.3% | 18.6 | 56.1 | ✗ | ✗ |
| `prem_lvl_ls_k3_8h` | +6.8% | -6.7% | -10.9% | -10.3% | -0.391 | -25.7% | +31.0% | +34.6% | 138.3 | 227.6 | ✗ | ✓ |
| `prem_lvl_ls_k3_24h` | -6.8% | -29.6% | -10.3% | -22.2% | -0.827 | -30.9% | +27.1% | +20.1% | 54.2 | 132.2 | ✗ | ✓ |
| `prem_lvl_ls_k3_72h` | +6.4% | +21.9% | +9.2% | +37.4% | +1.324 | +32.8% | +23.6% | +10.8% | 18.6 | 70.9 | ✓ | ✗ |
| `prem_lvl_ls_k5_8h` | +5.2% | -32.8% | -14.0% | -44.4% | -2.334 | -58.6% | +21.4% | +33.3% | 159.8 | 364.7 | ✗ | ✓ |
| `prem_lvl_ls_k5_24h` | -3.3% | -24.5% | -9.8% | -26.0% | -1.346 | -34.3% | +17.7% | +19.3% | 55.8 | 210.8 | ✗ | ✓ |
| `prem_lvl_ls_k5_72h` | +4.9% | +10.8% | +5.1% | +15.9% | +0.842 | +11.2% | +16.2% | +10.9% | 18.6 | 119.2 | ✓ | ✗ |
| `prem_chg_so_k3_8h` | +49.8% | -24.4% | -22.1% | -112.1% | -3.356 | -143.3% | -15.0% | +73.4% | 152.6 | 240.8 | ✗ | ✓ |
| `prem_chg_so_k3_24h` | +29.7% | -8.2% | -7.2% | -63.4% | -1.627 | -80.1% | -16.0% | +39.2% | 55.5 | 128.4 | ✗ | ✓ |
| `prem_chg_so_k3_72h` | -9.4% | -17.9% | -19.8% | -53.1% | -1.119 | -59.4% | -18.4% | +14.8% | 18.4 | 47.7 | ✗ | ✗ |
| `prem_chg_so_k5_8h` | +37.1% | -27.4% | -26.8% | -102.4% | -4.289 | -131.0% | -8.8% | +67.3% | 164.6 | 367.6 | ✗ | ✓ |
| `prem_chg_so_k5_24h` | +20.9% | +7.2% | -12.9% | -40.4% | -1.635 | -56.6% | -10.1% | +37.8% | 56.0 | 206.2 | ✓ | ✓ |
| `prem_chg_so_k5_72h` | +5.0% | -11.7% | -3.6% | -37.4% | -1.278 | -43.5% | -11.1% | +14.2% | 18.6 | 76.7 | ✗ | ✗ |
| `prem_chg_ls_k3_8h` | +18.6% | -25.2% | -54.6% | -87.8% | -3.355 | -119.9% | +12.5% | +75.5% | 166.2 | 495.9 | ✗ | ✓ |
| `prem_chg_ls_k3_24h` | +22.8% | -12.4% | -15.8% | -46.5% | -1.803 | -63.9% | +6.6% | +40.9% | 56.0 | 268.7 | ✗ | ✓ |
| `prem_chg_ls_k3_72h` | +9.6% | -8.1% | -3.1% | -20.6% | -0.616 | -27.0% | +2.6% | +15.0% | 18.6 | 98.4 | ✗ | ✗ |
| `prem_chg_ls_k5_8h` | +14.0% | -13.4% | -53.6% | -73.5% | -3.891 | -102.8% | +8.3% | +68.7% | 168.0 | 752.7 | ✗ | ✓ |
| `prem_chg_ls_k5_24h` | +16.3% | +2.2% | -21.3% | -32.2% | -1.784 | -48.9% | +4.4% | +39.0% | 56.0 | 427.1 | ✓ | ✓ |
| `prem_chg_ls_k5_72h` | +16.6% | -7.6% | +3.5% | -20.2% | -0.902 | -26.4% | +1.7% | +14.4% | 18.6 | 157.9 | ✗ | ✗ |

### Price strategies (17 x band 0 / 0.5)

| config | IS gross α | OOS gross α | IS perp α | OOS perp α | OOS perp α t | OOS perp s5 α | OOS funding/yr | OOS cost/yr | events/8wk | entries/8wk | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tsmom_ls_7@band0` | +49.0% | +14.0% | +36.2% | +3.2% | +0.074 | -1.3% | -0.3% | +10.5% | 52.1 | 378.7 | ✓ | ✓ |
| `tsmom_ls_7@band0.5` | +55.2% | +21.1% | +43.1% | +11.3% | +0.270 | +7.0% | -0.2% | +9.5% | 52.1 | 378.7 | ✓ | ✓ |
| `tsmom_lo_7@band0` | +23.5% | +7.1% | +13.6% | +3.7% | +0.172 | +1.6% | +1.7% | +5.0% | 49.7 | 190.0 | ✓ | ✓ |
| `tsmom_lo_7@band0.5` | +25.5% | +8.5% | +15.6% | +5.2% | +0.229 | +3.2% | +1.7% | +4.8% | 49.7 | 190.0 | ✓ | ✓ |
| `xsmom_7@band0` | -5.8% | +32.7% | -22.7% | +16.6% | +0.962 | +9.1% | +1.5% | +17.5% | 55.9 | 268.0 | ✗ | ✓ |
| `xsmom_7@band0.5` | -2.2% | +33.3% | -18.5% | +18.2% | +1.013 | +11.1% | +1.6% | +16.7% | 55.9 | 268.0 | ✗ | ✓ |
| `tsmom_ls_14@band0` | +58.5% | +4.5% | +48.0% | -3.7% | -0.086 | -6.9% | -0.6% | +7.6% | 50.6 | 261.0 | ✓ | ✓ |
| `tsmom_ls_14@band0.5` | +59.9% | +7.2% | +49.6% | +0.2% | +0.004 | -2.8% | -0.4% | +6.7% | 50.6 | 261.0 | ✓ | ✓ |
| `tsmom_lo_14@band0` | +28.3% | +2.0% | +19.6% | -0.0% | -0.002 | -1.6% | +1.5% | +3.6% | 48.3 | 130.5 | ✓ | ✓ |
| `tsmom_lo_14@band0.5` | +28.9% | +5.2% | +20.3% | +3.5% | +0.153 | +2.1% | +1.5% | +3.3% | 48.3 | 130.5 | ✓ | ✓ |
| `xsmom_14@band0` | +20.8% | +39.1% | +9.3% | +27.9% | +1.564 | +22.6% | +1.1% | +12.4% | 55.2 | 182.3 | ✓ | ✓ |
| `xsmom_14@band0.5` | +24.3% | +41.3% | +13.9% | +31.5% | +1.746 | +26.6% | +1.4% | +11.3% | 55.2 | 182.3 | ✓ | ✓ |
| `tsmom_ls_28@band0` | +63.6% | +17.3% | +54.4% | +10.6% | +0.254 | +8.0% | -0.7% | +6.0% | 48.7 | 196.0 | ✓ | ✓ |
| `tsmom_ls_28@band0.5` | +66.7% | +23.9% | +58.0% | +18.6% | +0.468 | +16.3% | -0.4% | +5.0% | 48.7 | 196.0 | ✓ | ✓ |
| `tsmom_lo_28@band0` | +30.8% | +8.6% | +22.8% | +7.3% | +0.353 | +6.2% | +1.5% | +2.7% | 44.1 | 97.2 | ✓ | ✓ |
| `tsmom_lo_28@band0.5` | +32.6% | +12.1% | +24.4% | +11.4% | +0.501 | +10.4% | +1.4% | +2.4% | 44.1 | 97.2 | ✓ | ✓ |
| `xsmom_28@band0` | +28.8% | +16.7% | +20.5% | +9.7% | +0.524 | +5.7% | +2.4% | +9.4% | 53.8 | 133.2 | ✓ | ✓ |
| `xsmom_28@band0.5` | +32.0% | +23.0% | +24.4% | +16.6% | +0.891 | +13.5% | +2.4% | +8.3% | 53.8 | 133.2 | ✓ | ✓ |
| `xsrev_1@band0` | -1.4% | -51.4% | -41.1% | -91.4% | -4.572 | -108.2% | -1.1% | +39.1% | 56.0 | 626.2 | ✗ | ✓ |
| `xsrev_1@band0.5` | -1.9% | -49.7% | -41.3% | -90.1% | -4.403 | -106.7% | -1.6% | +38.8% | 56.0 | 626.2 | ✗ | ✓ |
| `xsrev_3@band0` | -2.3% | -37.2% | -26.5% | -62.1% | -3.259 | -72.4% | -1.0% | +24.0% | 56.0 | 375.2 | ✗ | ✓ |
| `xsrev_3@band0.5` | -3.9% | -37.4% | -27.3% | -62.1% | -3.160 | -72.1% | -1.5% | +23.3% | 56.0 | 375.2 | ✗ | ✓ |
| `xsrev_4h@band0` | +33.5% | +38.1% | -204.7% | -200.2% | -11.486 | -302.1% | +0.1% | +239.5% | 336.0 | 3850.0 | ✓ | ✓ |
| `xsrev_4h@band0.5` | +33.2% | +36.8% | -204.4% | -201.3% | -11.318 | -302.9% | -0.2% | +238.8% | 336.0 | 3850.0 | ✓ | ✓ |
| `donchian_ls_20@band0` | +38.6% | -17.1% | +32.3% | -20.0% | -0.580 | -21.2% | -0.3% | +2.6% | 33.8 | 73.2 | ✗ | ✓ |
| `donchian_ls_20@band0.5` | +44.3% | -10.5% | +38.9% | -12.6% | -0.372 | -13.3% | -0.2% | +1.8% | 33.8 | 73.2 | ✗ | ✓ |
| `donchian_lo_20@band0` | +23.6% | -5.4% | +17.7% | -5.6% | -0.328 | -6.1% | +0.9% | +1.1% | 22.5 | 34.7 | ✗ | ✗ |
| `donchian_lo_20@band0.5` | +26.3% | -4.7% | +20.8% | -4.7% | -0.248 | -5.1% | +0.8% | +0.8% | 22.5 | 34.7 | ✗ | ✗ |
| `donchian_ls_55@band0` | +17.9% | -5.5% | +11.7% | -7.1% | -0.228 | -7.8% | -0.1% | +1.5% | 23.2 | 29.2 | ✗ | ✗ |
| `donchian_ls_55@band0.5` | +20.2% | +3.2% | +15.3% | +2.1% | +0.074 | +1.8% | +0.0% | +0.8% | 23.2 | 29.2 | ✓ | ✗ |
| `donchian_lo_55@band0` | +11.2% | -0.0% | +5.3% | +0.7% | +0.052 | +0.5% | +1.2% | +0.5% | 12.9 | 13.7 | ✗ | ✗ |
| `donchian_lo_55@band0.5` | +13.6% | +1.6% | +8.0% | +2.1% | +0.146 | +1.9% | +0.9% | +0.3% | 12.9 | 13.7 | ✓ | ✗ |
| `mr_1h@band0` | +2.5% | -27.8% | -23.8% | -53.6% | -2.235 | -64.8% | +0.3% | +26.1% | 628.3 | 999.6 | ✗ | ✓ |
| `mr_1h@band0.5` | +2.9% | -31.0% | -22.2% | -55.7% | -2.262 | -66.4% | +0.2% | +25.0% | 628.3 | 999.6 | ✗ | ✓ |

### Carry (2)

| config | IS gross α | OOS gross α | IS perp α | OOS perp α | OOS perp α t | OOS perp s5 α | OOS funding/yr | OOS cost/yr | events/8wk | entries/8wk | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `carry_1` | +5.8% | +0.1% | +4.6% | -0.1% | -2.965 | -0.1% | — | — | 5.9 | 3.7 | ✓ | ✗ |
| `carry_2` | +4.4% | -0.0% | +3.6% | -0.0% | -2.169 | -0.0% | — | — | 1.0 | 0.5 | ✗ | ✗ |

### SMC (8 configs x short / long / both; per-symbol scope)

| unit | IS n | OOS n | IS gross | OOS gross | IS perp | OOS perp | OOS perp s5 | funding R/trade OOS | trades/8wk | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `smc_5m_pbe_bias:long` | 26 | 26 | -0.084 | +0.069 | -0.170 | -0.013 | -0.036 | -0.000 | 2.6 | ✗ | ✗ |
| `smc_5m_pbe_bias:short` | 24 | 20 | +0.043 | +0.531 | -0.025 | +0.431 | +0.406 | +0.000 | 2.0 | ✓ | ✗ |
| `smc_5m_pbe_bias:both` | 50 | 46 | -0.023 | +0.270 | -0.101 | +0.180 | +0.156 | -0.000 | 4.6 | ✗ | ✗ |
| `smc_5m_pbe_nobias:long` | 97 | 95 | -0.006 | -0.147 | -0.095 | -0.236 | -0.262 | +0.000 | 9.6 | ✗ | ✗ |
| `smc_5m_pbe_nobias:short` | 85 | 61 | +0.174 | +0.343 | +0.102 | +0.232 | +0.201 | -0.000 | 6.2 | ✓ | ✗ |
| `smc_5m_pbe_nobias:both` | 182 | 156 | +0.078 | +0.044 | -0.003 | -0.053 | -0.081 | +0.000 | 15.8 | ✓ | ✗ |
| `smc_5m_tp2_bias:long` | 26 | 26 | -0.191 | +0.084 | -0.283 | -0.004 | -0.032 | -0.000 | 2.6 | ✗ | ✗ |
| `smc_5m_tp2_bias:short` | 24 | 20 | +0.100 | +0.789 | +0.027 | +0.676 | +0.643 | +0.000 | 2.0 | ✓ | ✗ |
| `smc_5m_tp2_bias:both` | 50 | 46 | -0.052 | +0.391 | -0.135 | +0.292 | +0.262 | -0.000 | 4.6 | ✗ | ✗ |
| `smc_5m_tp2_nobias:long` | 97 | 95 | -0.011 | -0.147 | -0.107 | -0.240 | -0.270 | +0.000 | 9.6 | ✗ | ✗ |
| `smc_5m_tp2_nobias:short` | 85 | 61 | +0.247 | +0.549 | +0.169 | +0.428 | +0.391 | -0.000 | 6.2 | ✓ | ✗ |
| `smc_5m_tp2_nobias:both` | 182 | 156 | +0.110 | +0.125 | +0.022 | +0.021 | -0.012 | -0.000 | 15.8 | ✓ | ✗ |
| `smc_15m_pbe_bias:long` | 9 | 7 | +0.208 | +0.009 | +0.155 | -0.051 | -0.068 | +0.000 | 0.7 | ✓ | ✗ |
| `smc_15m_pbe_bias:short` | 12 | 6 | -0.391 | +0.157 | -0.453 | +0.070 | +0.043 | -0.000 | 0.6 | ✗ | ✗ |
| `smc_15m_pbe_bias:both` | 21 | 13 | -0.134 | +0.078 | -0.192 | +0.005 | -0.017 | +0.000 | 1.3 | ✗ | ✗ |
| `smc_15m_pbe_nobias:long` | 40 | 24 | +0.135 | -0.073 | +0.075 | -0.136 | -0.155 | -0.000 | 2.4 | ✗ | ✗ |
| `smc_15m_pbe_nobias:short` | 28 | 22 | -0.171 | +0.047 | -0.227 | -0.019 | -0.039 | -0.000 | 2.2 | ✗ | ✗ |
| `smc_15m_pbe_nobias:both` | 68 | 46 | +0.009 | -0.016 | -0.050 | -0.080 | -0.100 | -0.000 | 4.6 | ✗ | ✗ |
| `smc_15m_tp2_bias:long` | 9 | 7 | +0.235 | +0.030 | +0.176 | -0.035 | -0.054 | +0.000 | 0.7 | ✓ | ✗ |
| `smc_15m_tp2_bias:short` | 12 | 6 | -0.430 | +0.258 | -0.493 | +0.169 | +0.141 | -0.000 | 0.6 | ✗ | ✗ |
| `smc_15m_tp2_bias:both` | 21 | 13 | -0.145 | +0.135 | -0.206 | +0.059 | +0.036 | +0.000 | 1.3 | ✗ | ✗ |
| `smc_15m_tp2_nobias:long` | 40 | 24 | +0.086 | -0.039 | +0.021 | -0.106 | -0.127 | -0.000 | 2.4 | ✗ | ✗ |
| `smc_15m_tp2_nobias:short` | 28 | 22 | -0.206 | +0.108 | -0.267 | +0.038 | +0.015 | -0.000 | 2.2 | ✗ | ✗ |
| `smc_15m_tp2_nobias:both` | 68 | 46 | -0.034 | +0.031 | -0.097 | -0.037 | -0.059 | -0.000 | 4.6 | ✗ | ✗ |

### Screen

- screened: 108 rows (48 crowding, 34 price x band, 2 carry, 24 SMC units of which 8 short)
- pass (a) gross > 0 IS and OOS: 48
- pass (a) and (b) >= 30 events / 8 weeks: 29
- pass (a) but fail (b): fund_lvl_so_k3_72h, fund_lvl_so_k5_72h, fund_lvl_ls_k3_72h, fund_lvl_ls_k5_72h, fund_chg_ls_k3_72h, fund_chg_ls_k5_72h, prem_lvl_ls_k3_72h, prem_lvl_ls_k5_72h, donchian_ls_55@band0.5, donchian_lo_55@band0.5, carry_1, smc_5m_pbe_bias:short, smc_5m_pbe_nobias:short, smc_5m_pbe_nobias:both, smc_5m_tp2_bias:short, smc_5m_tp2_nobias:short, smc_5m_tp2_nobias:both, smc_15m_pbe_bias:long, smc_15m_tp2_bias:long

1. `fund_lvl_ls_k3_24h`: gross IS +0.0138, gross OOS +0.5208, perp OOS +0.6221, perp s5 OOS +0.5445, events/8wk 53.2, perp OOS >= 0: True
2. `fund_lvl_ls_k3_8h`: gross IS +0.0066, gross OOS +0.4280, perp OOS +0.5494, perp s5 OOS +0.4386, events/8wk 122.6, perp OOS >= 0: True
3. `xsmom_14@band0.5`: gross IS +0.2435, gross OOS +0.4134, perp OOS +0.3147, perp s5 OOS +0.2665, events/8wk 55.2, perp OOS >= 0: True
4. `xsmom_14@band0`: gross IS +0.2078, gross OOS +0.3913, perp OOS +0.2793, perp s5 OOS +0.2264, events/8wk 55.2, perp OOS >= 0: True
5. `fund_lvl_ls_k5_8h`: gross IS +0.0914, gross OOS +0.3020, perp OOS +0.3198, perp s5 OOS +0.2192, events/8wk 144.0, perp OOS >= 0: True
6. `fund_lvl_ls_k5_24h`: gross IS +0.1252, gross OOS +0.1998, perp OOS +0.2564, perp s5 OOS +0.1885, events/8wk 55.2, perp OOS >= 0: True
7. `fund_chg_ls_k3_24h`: gross IS +0.2687, gross OOS +0.4976, perp OOS +0.2863, perp s5 OOS +0.1441, events/8wk 55.8, perp OOS >= 0: True
8. `xsmom_28@band0.5`: gross IS +0.3204, gross OOS +0.2298, perp OOS +0.1656, perp s5 OOS +0.1353, events/8wk 53.8, perp OOS >= 0: True
9. `fund_lvl_so_k3_8h`: gross IS +0.1744, gross OOS +0.4437, perp OOS +0.1752, perp s5 OOS +0.0328, events/8wk 100.9, perp OOS >= 0: True
10. `xsmom_28@band0`: gross IS +0.2884, gross OOS +0.1667, perp OOS +0.0967, perp s5 OOS +0.0565, events/8wk 53.8, perp OOS >= 0: True
11. `tsmom_lo_28@band0.5`: gross IS +0.3259, gross OOS +0.1214, perp OOS +0.1139, perp s5 OOS +0.1037, events/8wk 44.1, perp OOS >= 0: True
12. `tsmom_ls_28@band0.5`: gross IS +0.6666, gross OOS +0.2387, perp OOS +0.1859, perp s5 OOS +0.1633, events/8wk 48.7, perp OOS >= 0: True
13. `tsmom_lo_28@band0`: gross IS +0.3083, gross OOS +0.0857, perp OOS +0.0735, perp s5 OOS +0.0620, events/8wk 44.1, perp OOS >= 0: True
14. `tsmom_ls_7@band0.5`: gross IS +0.5523, gross OOS +0.2109, perp OOS +0.1135, perp s5 OOS +0.0703, events/8wk 52.1, perp OOS >= 0: True
15. `tsmom_ls_28@band0`: gross IS +0.6355, gross OOS +0.1727, perp OOS +0.1060, perp s5 OOS +0.0803, events/8wk 48.7, perp OOS >= 0: True
16. `tsmom_lo_7@band0.5`: gross IS +0.2546, gross OOS +0.0850, perp OOS +0.0524, perp s5 OOS +0.0321, events/8wk 49.7, perp OOS >= 0: True
17. `fund_lvl_so_k5_8h`: gross IS +0.1625, gross OOS +0.2696, perp OOS +0.0367, perp s5 OOS -0.0842, events/8wk 122.0, perp OOS >= 0: True
18. `tsmom_lo_7@band0`: gross IS +0.2348, gross OOS +0.0706, perp OOS +0.0372, perp s5 OOS +0.0159, events/8wk 49.7, perp OOS >= 0: True
19. `tsmom_lo_14@band0.5`: gross IS +0.2893, gross OOS +0.0518, perp OOS +0.0350, perp s5 OOS +0.0209, events/8wk 48.3, perp OOS >= 0: True
20. `tsmom_ls_7@band0`: gross IS +0.4904, gross OOS +0.1397, perp OOS +0.0321, perp s5 OOS -0.0128, events/8wk 52.1, perp OOS >= 0: True
21. `tsmom_ls_14@band0.5`: gross IS +0.5986, gross OOS +0.0717, perp OOS +0.0016, perp s5 OOS -0.0284, events/8wk 50.6, perp OOS >= 0: True
22. `tsmom_lo_14@band0`: gross IS +0.2829, gross OOS +0.0202, perp OOS -0.0004, perp s5 OOS -0.0157, events/8wk 48.3, perp OOS >= 0: False
23. `tsmom_ls_14@band0`: gross IS +0.5854, gross OOS +0.0450, perp OOS -0.0369, perp s5 OOS -0.0694, events/8wk 50.6, perp OOS >= 0: False
24. `fund_lvl_so_k5_24h`: gross IS +0.1402, gross OOS +0.1046, perp OOS -0.0348, perp s5 OOS -0.1184, events/8wk 53.1, perp OOS >= 0: False
25. `fund_chg_so_k3_24h`: gross IS +0.3396, gross OOS +0.2961, perp OOS -0.1547, perp s5 OOS -0.2784, events/8wk 50.7, perp OOS >= 0: False
26. `prem_chg_so_k5_24h`: gross IS +0.2087, gross OOS +0.0721, perp OOS -0.4045, perp s5 OOS -0.5657, events/8wk 56.0, perp OOS >= 0: False
27. `prem_chg_ls_k5_24h`: gross IS +0.1628, gross OOS +0.0221, perp OOS -0.3224, perp s5 OOS -0.4890, events/8wk 56.0, perp OOS >= 0: False
28. `xsrev_4h@band0.5`: gross IS +0.3320, gross OOS +0.3677, perp OOS -2.0128, perp s5 OOS -3.0292, events/8wk 336.0, perp OOS >= 0: False
29. `xsrev_4h@band0`: gross IS +0.3348, gross OOS +0.3814, perp OOS -2.0023, perp s5 OOS -3.0214, events/8wk 336.0, perp OOS >= 0: False
