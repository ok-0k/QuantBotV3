# Pre-registration: crowding (funding, premium) as a cross-sectional short signal

**Status: design only.** Written and committed on 2026-10-13 on branch
`research/funding-signal` (based on `9b9d8ac`). It was committed **before any
backtest of these signals was run** and before any return, Sharpe or alpha for
them existed. The only data looked at so far is cache coverage: first and last
timestamps, row counts, and how often each perp settles funding, per symbol.
That is the same kind of data-only pass as `research/data_check.py`. A Project
Manager reviews this document before anything runs.

Code: `research/crowding.py` implements sections 3 to 9 and 11 (inputs,
signals, eligibility, weights, schedule, costs, grid, decision rules), with
synthetic-data tests in
`tests/test_research_crowding.py`. `research/backtest.py` gains one
default-off option, `trade_at`, so a book can hold its positions between
rebalances. The runner is written only after approval, and it must use these
functions and constants unchanged.

## 1. Question

Does crowding information predict which coins underperform, net of realistic
costs, out of sample? "Crowding information" here means perp funding and the
perp premium over spot.

The existing study (`46529d2`, corrected 2026-10-11) rejected all 17
price-based strategies out of sample. Hedged funding carry also fails out of
sample once the phantom hedges are removed. Until now funding and premium have
only been used for hedged carry, never as a directional, cross-sectional
signal. The live bot is short-only and its intraday entries show no edge
(`scripts/entry_edge.py`). So the useful question for it is whether shorting
the most crowded coins, and holding them for 8 to 72 hours, earns anything
beyond the market's own drift.

**Hypothesis.** Coins with the highest funding or premium (crowded longs)
underperform the equal-weight universe over the following 8 to 72 hours. The
long/short variant also tests the mirror case: coins with the lowest
funding or premium outperform.

## 2. Carried over unchanged from the existing study

| item | value |
|---|---|
| Universe | `config.SYMBOLS`: the live bot's 50 symbols, including the 4 delisted pairs up to their delisting |
| Data | Binance spot 1h klines from `DATA_START` = 2022-10-01 (warm-up), USDT-M perp funding settlements, perp premium index (8h klines), all from the cache under `/home/admin/trading_data/research/` |
| IS (design) | 2023-01-01 00:00 → 2025-04-01 00:00 UTC: 1h bars labelled in [IS_START, OOS_START) |
| OOS (held out) | 2025-04-01 00:00 → 2026-10-07 00:00 UTC: 1h bars labelled in [OOS_START, END] |
| Backtester | `research.backtest.run`. The weight on bar t is decided at t's close and earns bar t+1. Every change in position pays fee + slippage per unit traded. Short notional pays `borrow_daily / 24` per 1h bar |
| Metrics | `research.backtest.metrics` with `bars_per_year = 8760`. Sharpe uses 1h net returns. Beta and alpha come from regressing daily-compounded strategy returns on the benchmark: `alpha_yr` = mean daily residual × 365, and `alpha_t` is its t-stat |
| Benchmark | The equal-weight universe, built exactly like `bench["1h"]` in `research.study`. Each hour it is the mean 1h spot return of the coins that pass `strategies.tradable` at the previous hour. Alpha is measured against it |
| Funding bucketing | `research.study.funding_buckets` (`7e4fd17`). Settlements are summed into 8h windows labelled by their END |
| Spot mask | `research.study.spot_tradable` (audit, 2026-10-11) |
| Candidate / confirmation thresholds | `research.study.CANDIDATE` and `CONFIRM`, unchanged. Section 11 adds new rules on top |

## 3. Data and timestamp alignment

**Two grids.**
- **1h grid (backtest):** `build_panel(SYMBOLS, "1h", DATA_START)`. Row T is
  the spot bar (T−1h, T], with its close at T. The index is the full regular
  range, and missing bars are NaN.
- **8h grid (signals):** labels at 00:00, 08:00 and 16:00 UTC, anchored to the
  epoch. Label t stands for the window (t−8h, t]. Every label from
  DATA_START to END exists (`crowding.grid_8h`). A window with no data is
  NaN and is never forward-filled.

**Inputs on the 8h grid.**
- `F[t, s]`, funding: each settlement timestamp is rounded to the hour, then
  ceiled to 8h, and settlements with the same label are summed
  (`funding_buckets`; a window with no settlement stays NaN). So `F[t]` is the
  total funding paid in (t−8h, t], as a fraction of notional.
  - Perps that settle every 4h contribute two settlements per window. TIA,
    RENDER and ILV settle every 4h throughout; SOL, AXS, ENJ, GMT, FLOW and
    BLUR settle more often than every 8h for part of the period.
  - A settlement stamped 08:00:00.008 belongs to label 08:00.
- `P[t, s]`, premium: the close of the 8h premium-index kline that opened at
  t−8h. `research.data.load_premium` already labels it by its close time t.
  A label that falls off the 8h grid raises an error (`crowding.on_grid`).
- `spot_ok[t, s]` = `spot_tradable(close_1h)[t]`. It is True if the spot pair
  has at least one 1h close in (t−8h, t].
- `liquid[t, s]` = `strategies.tradable(p1h, 24)` at 1h row t. It requires:
  - listed for at least 30 days (at least 95% of the last 720 hours have a
    close);
  - a trailing 7-day average quote volume of at least $2M a day;
  - a close at t.

**Truncation.** All inputs are cut at **END = 2026-10-07 00:00 UTC**: 1h rows
≤ END and 8h labels ≤ END. The cache now runs to 2026-10-13, because the carry
forward test refreshes it, so this hard cap is what keeps the OOS window equal
to the existing study's. `crowding.load_inputs` refuses to run unless
`RESEARCH_OFFLINE=1` is set, which means no requests and no cache writes.

**Information rule.** The decision at label t uses `F`, `P`, `spot_ok` and
`liquid` at labels ≤ t only. Everything it uses had settled or closed by t:
- the last funding settlement, at t plus a few milliseconds;
- the premium bar that closed at t;
- the 1h spot bars that closed at or before t.

**Fill rule.** The target weights decided at t are written on **1h row t+1h**.
Under the backtester's convention they are filled at close(t+1h) and first earn
the bar (t+1h, t+2h]. The position never earns the bar (t, t+1h], which is the
execution bar: the fill happens at the close of the next 1h bar.
- Example: the decision at 2025-06-01 08:00 UTC uses funding settled up to
  08:00 and premium bars closed up to 08:00. It is filled at the 09:00 close and
  earns from 09:00 onward.
- Why one hour and not zero: it absorbs the millisecond-to-second jitter of
  settlement timestamps, and the time a real executor needs to fetch funding
  and premium. It also means no fill uses a price printed at the same instant
  as the decision's last input.

**Between rebalances nothing trades.** `backtest.run(..., trade_at=)` is True
only on fill rows. On every other bar the book keeps its drifted weights,
which is the same as holding fixed quantities.

**Continuity.** The backtest runs without a break from DATA_START, and metrics
are taken on the IS and OOS windows. Positions open at OOS_START carry over
from IS, as in the existing study. Nothing is force-closed at END.

## 4. Signals (exact)

`M_n(X)[t] = (1/n) * sum_{j=0..n-1} X[t - j*8h]`. It is defined only if all
n values are present; otherwise it is NaN (`crowding.trailing_mean`).

| signal | formula | meaning |
|---|---|---|
| `fund_lvl` | `M_9(F)` | mean funding per 8h over the last 3 days |
| `fund_chg` | `M_3(F) − M_21(F)` | the last day's mean funding minus the last week's |
| `prem_lvl` | `M_9(P)` | mean premium over the last 3 days (9 closes) |
| `prem_chg` | `M_3(P) − M_21(P)` | the last day's mean premium minus the last week's |

- **Direction:** a higher score means a more crowded long, which makes the
  coin a short candidate.
- **Lookbacks:** fixed without looking at data. 9 windows is the carry study's
  existing 3-day trail; 3 and 21 windows are one day and one week. No other
  lookback will be tried.
- **Why premium as well as funding:** Binance funding = average premium +
  clamp(0.01% − average premium, ±0.05%) per 8h. So funding sits at exactly
  0.01% whenever the average premium is within about [−0.04%, +0.06%].
  - In calm markets many coins therefore tie at the default funding rate.
  - Premium is continuous, so it still ranks them.

## 5. Eligibility and skipped rebalances

`eligible[t, s] = score[t, s] is not NaN AND spot_ok[t, s] AND liquid[t, s]`
(`crowding.eligible`).

- **Perp data as the signal needs.** A funding signal needs all 9 (or 21)
  funding windows. A premium signal needs all 9 (or 21) premium closes. A
  funding signal does not need premium data, and the reverse.
- **Spot mask.** This is what excludes perp data for coins that could not be
  traded on spot at the time:
  - MATIC: spot delisted 2024-09-10, but funding kept printing to 2026-10.
  - OCEAN and AGIX: spot ended 2024-07-01, funding ran to 2025-06.
  - MKR: spot ended 2025-09-15.
  - BLUR: perp data from 2023-04-27, but spot only listed 2023-11-24.
- **Liquidity and listing age.** This is the existing study's universe filter
  (a judgement call; see section 13).
- **Fewer than 10 eligible coins at a scheduled decision: the rebalance is
  skipped.** Nothing trades, and the book keeps its drifted positions (or stays
  flat if it has none yet, as during warm-up).
  - *Why hold rather than go flat:* a thin cross-section is a data or market
    problem (an exchange outage, missing perp history), not a signal.
    Liquidating would charge a round trip of costs because of a data gap.
    Re-ranking a thin universe would trade on noise.
  - Holding is also what an executor does when its inputs are unavailable. A
    stale book lasts at most until the next executed rebalance.
  - Skips are counted and reported, and rule N4 below bounds them.
- **Positions that drop out.** A held coin that is no longer eligible at a
  later executed rebalance is closed (target 0) and pays its exit cost.
- **Missing prices.** A held coin whose spot price disappears between
  rebalances (a delisting or a gap) earns 0, because the backtester treats a
  missing return as 0. It is closed at the next executed rebalance.

## 6. Ranking and the two books

At an executed rebalance, take the eligible coins' scores `x`, rounded to 12
decimal places (`SCORE_DECIMALS`).

- **`slots_k(x)`** (`crowding._slots`) picks the k largest values.
  - Coins strictly above the k-th largest value get a full slot.
  - Coins tied with the k-th largest value share the remaining slots equally.
  - The slots always sum to k.
- **Short-only `so`:** `w = −slots_k(x) / k`. This shorts the top k most
  crowded coins at −1/k each and leaves everything else flat. Gross exposure
  is 1 and net exposure is −1.
- **Long/short `ls`:** `w = (0.5/k) * (slots_k(−x) − slots_k(x))`. This shorts
  the top k at −0.5/k and goes long the bottom k at +0.5/k. Gross exposure is
  1 and net exposure is 0.
  - A coin in both tie blocks nets out. If every eligible coin ties, the `ls`
    book is flat.
- **Why share ties.** Ties at the default 0.01% funding are common, and any
  order among tied coins (alphabetical, column order) would be arbitrary.
  - Rounding to 1e-12 stops float noise in rolling means from breaking real
    ties. Binance publishes funding rates with 8 decimals, so genuinely
    different 9-window means differ by at least about 1e-9.
- **k ∈ {3, 5}.** With at least 10 eligible coins, the top k and bottom k can
  only overlap through ties.
- **Scale.** All thresholds are defined at gross exposure 1. A deployment may
  scale the book down; Sharpe and alpha t-stats do not depend on scale.

## 7. Rebalance schedule and the no-trade band

| period | decision labels t (UTC) | per year |
|---|---|---|
| 8h | every 8h label: 00:00, 08:00, 16:00 | 1095 |
| 24h | 00:00 every day | 365 |
| 72h | 00:00 on days whose number since 1970-01-01 is divisible by 3 (e.g. 2025-03-31, 2025-04-03, 2025-04-06; 2025-04-01 is day 20179) | ~122 |

- **Anchors:** the schedule is anchored to the epoch, not to the data.
- **Fill row:** each decision at t is filled on 1h row t+1h, where `trade_at`
  is True. It is False there if the rebalance was skipped.
- **No-trade band: `band = 0.5`**, the `research.backtest.run` band, applied on
  fill rows only.
  - A coin that stays in the book with the same sign keeps its drifted weight
    while `|held − target| ≤ 0.5·|target|`. Otherwise it is traded back to
    its target.
  - New entries go straight to target, and exits go to 0.
  - 0.5 is the only band value the existing study ever used (its Run 2), and
    it is the only value used here. It is not a grid dimension.

## 8. Costs

Every configuration is run under all three cases (`crowding.COSTS`):

| case | fee per side | slippage per side | borrow on short notional |
|---|---|---|---|
| base | 0.10% | 0.08% | 0.03% a day |
| 2× stress | 0.20% | 0.16% | 0.06% a day |
| borrow stress | 0.10% | 0.08% | **0.10% a day** |

- **How borrow is charged:** every hour, on the drifted short notional. The
  `ls` book pays it on its short half only.
- **Borrow stress:** crowded coins are the costliest to borrow, and 0.10%/day
  is about 36.5% a year on short notional.
- **No funding is credited or charged.** The bot shorts spot on margin, so its
  shorts receive no funding.
  - If a candidate were instead built with perps, its shorts would receive the
    mostly positive funding of the crowded coins. That is a different product,
    outside this study, and it is not credited here.
- **Unverified assumption:** whether each coin can actually be borrowed on
  margin, and at what rate, cannot be checked offline (see section 13).

## 9. The grid: 48 configurations, complete

4 signals × 2 books × 2 values of k × 3 rebalance periods = 48 configurations.
The band is fixed at 0.5. Names follow `{signal}_{book}_k{k}_{period}`
(`crowding.grid()`).

| # | signal | book | k=3, 8h | k=3, 24h | k=3, 72h | k=5, 8h | k=5, 24h | k=5, 72h |
|---|---|---|---|---|---|---|---|---|
| 1–6 | fund_lvl | so | `fund_lvl_so_k3_8h` | `fund_lvl_so_k3_24h` | `fund_lvl_so_k3_72h` | `fund_lvl_so_k5_8h` | `fund_lvl_so_k5_24h` | `fund_lvl_so_k5_72h` |
| 7–12 | fund_lvl | ls | `fund_lvl_ls_k3_8h` | `fund_lvl_ls_k3_24h` | `fund_lvl_ls_k3_72h` | `fund_lvl_ls_k5_8h` | `fund_lvl_ls_k5_24h` | `fund_lvl_ls_k5_72h` |
| 13–18 | fund_chg | so | `fund_chg_so_k3_8h` | `fund_chg_so_k3_24h` | `fund_chg_so_k3_72h` | `fund_chg_so_k5_8h` | `fund_chg_so_k5_24h` | `fund_chg_so_k5_72h` |
| 19–24 | fund_chg | ls | `fund_chg_ls_k3_8h` | `fund_chg_ls_k3_24h` | `fund_chg_ls_k3_72h` | `fund_chg_ls_k5_8h` | `fund_chg_ls_k5_24h` | `fund_chg_ls_k5_72h` |
| 25–30 | prem_lvl | so | `prem_lvl_so_k3_8h` | `prem_lvl_so_k3_24h` | `prem_lvl_so_k3_72h` | `prem_lvl_so_k5_8h` | `prem_lvl_so_k5_24h` | `prem_lvl_so_k5_72h` |
| 31–36 | prem_lvl | ls | `prem_lvl_ls_k3_8h` | `prem_lvl_ls_k3_24h` | `prem_lvl_ls_k3_72h` | `prem_lvl_ls_k5_8h` | `prem_lvl_ls_k5_24h` | `prem_lvl_ls_k5_72h` |
| 37–42 | prem_chg | so | `prem_chg_so_k3_8h` | `prem_chg_so_k3_24h` | `prem_chg_so_k3_72h` | `prem_chg_so_k5_8h` | `prem_chg_so_k5_24h` | `prem_chg_so_k5_72h` |
| 43–48 | prem_chg | ls | `prem_chg_ls_k3_8h` | `prem_chg_ls_k3_24h` | `prem_chg_ls_k3_72h` | `prem_chg_ls_k5_8h` | `prem_chg_ls_k5_24h` | `prem_chg_ls_k5_72h` |

**Neighbours** (`crowding.neighbours`) share a configuration's signal and book
and are one step away in k (3 ↔ 5) or in period (8h ↔ 24h ↔ 72h). For example,
`fund_lvl_so_k3_8h` has the neighbours `fund_lvl_so_k5_8h` and
`fund_lvl_so_k3_24h`.

**Reference books.** These are not configurations and can never become
candidates. They are reported next to the grid:
- `bench_ew`: the 1h equal-weight universe.
- `bench_btc`: BTC buy-and-hold.
- `ref_short_all_{8h,24h,72h}`: short every `fund_lvl`-eligible coin at equal
  weight, with the same schedule, skip rule, band and costs as the grid.

The `ref_short_all` books show what a short-only book earns from market drift
and the book mechanics alone.

## 10. What is reported

All 48 configurations are reported, not only the winners, plus the reference
books. Output goes to `results/<timestamp>-crowding/`: `report.txt`,
`summary.json` and `daily_equity.csv`.

For each configuration:
- **IS and OOS:** Sharpe, CAGR, volatility, max drawdown, beta, alpha a year,
  alpha t-stat, turnover a year, cost a year (fees + slippage + borrow), gross
  return a year, and average gross and net exposure.
- **OOS stress cases:** Sharpe and alpha a year at 2× costs, and Sharpe and
  alpha a year under borrow stress.
- **OOS halves:** alpha a year at base costs for H1 = [2025-04-01, 2026-01-01)
  and H2 = [2026-01-01, 2026-10-07 00:00]. These are 275 and 279 days. Beta is
  re-estimated within each half.
- **OOS alpha at zero cost**, as a diagnostic. It separates "no signal" from
  "a signal eaten by costs".
- **Coverage:** rebalances executed and skipped in IS and in OOS, and the mean
  and minimum eligible count.
- **Concentration:** the five symbols with the largest share of OOS
  short-holding time.
- **Multiple testing:** whether the IS alpha t-stat clears the Bonferroni level
  of 3.08 (informational only; see section 12).
- **Verdict:** the verdict and the list of failed checks (`crowding.verdict`).

For the run as a whole: the data fingerprint (`crowding.fingerprint`), the
commit hash, END and the band.

For PASSED configurations only, as a non-decisional diagnostic: the same
configuration with the other schedule phases (24h at 08:00 and at 16:00 UTC;
72h on days ≡ 1 and ≡ 2 mod 3). It is reported, never used to decide.

## 11. Decision rules (fixed now)

All rules use base costs unless a rule says otherwise. The thresholds live in
`research.study.CANDIDATE`, `CONFIRM` and `research.crowding.PASS`, and the
logic in `crowding.verdict`.

**CANDIDATE (IS only), unchanged from the existing study:**

| check | rule |
|---|---|
| `is_sharpe` | IS Sharpe ≥ 1.0 |
| `is_alpha_t` | IS alpha t-stat vs the equal-weight universe ≥ 2.0 |

**PASSED: a candidate that meets every check below.**

| check | rule | origin |
|---|---|---|
| `oos_sharpe` | OOS Sharpe ≥ 0.75 | existing |
| `oos_alpha_pos` | OOS alpha > 0 | existing |
| `oos_sharpe_2x` | OOS Sharpe at 2× costs ≥ 0.5 | existing |
| `neighbour` | at least one neighbour has IS Sharpe ≥ 0.5 **and** OOS alpha > 0 | existing; the OOS-alpha part is new |
| `oos_alpha_min` | **OOS alpha ≥ +10% a year**, net of base costs, against the equal-weight universe | new: minimum economic return |
| `oos_h1_alpha`, `oos_h2_alpha` | alpha > 0 in **both** OOS halves | new |
| `borrow_sharpe`, `borrow_alpha` | at 0.10%/day borrow: OOS Sharpe ≥ 0.5 and OOS alpha > 0 | new |
| `coverage` | at most 5% of OOS rebalances skipped (decision labels in [OOS_START, END)) | new |
| `drop_one` | run only if every check above passes. Re-run the configuration once per symbol that held a non-zero target on any OOS fill row, with that symbol's column removed from every input except the benchmark. The lowest of these OOS alphas must be > 0 | new: the MATIC lesson |

**Verdicts:**
- `rejected`: fails the candidate gate.
- `candidate (failed OOS)`: passes the candidate gate but fails at least one
  PASSED check.
- `PASSED (candidate only; needs independent re-run)`: passes everything.

**Why alpha and not CAGR.** OOS was a bear market for alts: the equal-weight
basket returned −20% to −26% a year. A short-only basket gains on that drift
alone, with no selection skill at all. Alpha is the regression intercept
against the equal-weight universe, so that drift is removed before the
economic test.

**Why 10% a year.**
- It has to beat parking USDT (about 5%; the carry forward test also uses a 5%
  bar).
- It also has to leave room for frictions the backtest does not model:
  - slippage on crowded small caps worse than 0.08%;
  - borrow that is unavailable or expensive;
  - margin top-ups.
- A smaller alpha is not worth building a new executor for.

**Caveat for the short-only book at the candidate gate.** IS (2023-01 →
2025-04) was mostly a bull market for crypto. A short-only book's raw IS
Sharpe therefore carries negative market drift, and it can reach IS Sharpe ≥ 1
only if its selection alpha outweighs that drift.
- The gate is kept unchanged on purpose, as required. A short book that loses
  in bull markets and wins in bear markets is mostly beta, not an edge.
- The consequence is that the short-only variant faces a much higher bar than
  the `ls` variant. The `ls` book is the cleaner test of the crowding
  hypothesis itself.

## 12. Multiple testing

There are 48 configurations. They are strongly correlated: the same signal at
neighbouring k and period values.

- **Standard errors:** under no edge, the Sharpe estimate's standard error is
  about 0.67 in sample (2.25 years) and about 0.81 out of sample (1.52 years).
- **OOS Sharpe alone:** a configuration with no edge clears OOS Sharpe ≥ 0.75
  by luck about 18% of the time, so on its own that rule is weak evidence.
- **The IS candidate gate:**
  - It needs alpha t ≥ 2.0, about p = 0.023 one-sided per test.
  - A Bonferroni correction for 48 independent tests would need t ≥ 3.08.
  - Even with the correlation, one or two configurations could clear the gate
    by chance alone.
  - The report flags whether each candidate clears 3.08. That flag is
    informational: the pre-registered gate stays at 2.0.
- **The defences:**
  - the stacked OOS rules, especially the minimum alpha, the both-halves rule
    and the drop-one rule;
  - then the independent re-run and a forward paper test (section 14).
- **Bottom line:** even a PASSED configuration is weak evidence until it
  survives a forward test.

## 13. Judgement calls and unverified assumptions

**Judgement calls (made here, before any result):**
1. **One-hour fill lag** (section 3) rather than a fill at the decision bar's
   own close.
2. **Skipped rebalances hold the book** (section 5) rather than going flat.
3. **The existing study's liquidity and listing-age filter** is added to
   eligibility on top of perp data and the spot mask. New listings have
   extreme funding and borrow that is usually unavailable.
4. **Ties share slots, and scores are rounded to 1e-12** (section 6).
5. **The band is fixed at 0.5** and applies only at rebalances. There is no
   rank hysteresis, and the band is not a grid dimension.
6. **k ∈ {3, 5}, the lookbacks 9 / 3 / 21, and "change" defined as the 1-day
   minus the 7-day mean.**
7. **Borrow stress passes at Sharpe ≥ 0.5 and alpha > 0.** This mirrors the 2×
   cost rule. A stricter reading is that alpha must still be ≥ 10% at
   0.10%/day borrow. For a short-only book that needs about 35% alpha at base
   borrow. **The PM may switch to the stricter reading before the run.**
8. **Added rules:** the neighbour's OOS alpha > 0, the drop-one rule and the
   coverage rule.
9. **OOS split at 2026-01-01**, giving 275 and 279 days.
10. **Schedule anchors:** 24h at 00:00 UTC; 72h anchored to the epoch.
11. **Benchmark:** the existing equal-weight universe (`bench["1h"]`), not a
    separate one per signal.
12. **The END hard cap** at 2026-10-07 00:00 UTC.

**Unverified assumptions:**
- **Margin borrowability per coin cannot be verified offline.** The study
  assumes every eligible coin could be shorted on Binance margin at every
  rebalance, at 0.03%/day (0.10%/day under stress).
  - Historical margin availability and rates are not in the cache.
  - Crowded coins are exactly the ones most likely to be unavailable to borrow,
    or expensive.
- **Fills:** at the 1h close with 0.08% slippage. Squeezes on crowded small
  caps could be worse.
- **Data timing:** funding and premium are assumed to be published within the
  one-hour lag.
- **Squeezes:** there is no margin-call or forced-liquidation model for the
  shorts. A k=3 short book holds about 33% of equity per coin.
- **Survivorship:** the universe was chosen in 2026, although the delisted
  pairs are kept up to their delisting.
- **Missing bars:** a missing spot bar earns 0 return.

## 14. What a pass means

**A PASSED configuration is a CANDIDATE only.** Before it means anything:

1. **A second, independent re-run from a clean checkout is required.** Someone
   other than the first runner does it, from a fresh `git worktree` at the run
   commit, with `RESEARCH_OFFLINE=1`.
   - It must reproduce the data fingerprint, every verdict, and every reported
     metric to 1e-9 relative.
   - Any difference voids the run until it is explained.
2. **Bringing a candidate into the bot means building a new 8–72h rebalancing
   executor, with its own review.** The bot's current machinery opens and
   closes intraday positions one at a time and cannot hold a scheduled,
   cross-sectional book.
   - The executor needs its own design review and tests.
   - It also needs live checks of margin borrow availability and rates for the
     coins it would short.
3. **A forward paper test, with its pass rule fixed in advance** (as
   `research/carry_forward.py` did), comes before any real money. No real
   money goes behind a backtest alone.

If nothing passes, RESULTS.md records that crowding signals fail as
pre-registered, and there is no re-tuning.

## 15. What may not change after results are seen

None of the following may change after results are seen:
- the grid;
- the signal formulas and lookbacks;
- k, the periods and the schedule anchors;
- the band and the fill lag;
- the eligibility rules and the minimum of 10 eligible coins;
- the costs, the periods, END and the benchmark;
- every threshold.

**Bug fixes** are allowed only under all of these conditions:
- a test demonstrates the bug;
- the fix is documented with before and after numbers for all 48
  configurations;
- it is labelled as a post-hoc correction, like the carry correction of
  2026-10-11.

**Any further analysis** is labelled exploratory and cannot produce a PASS.

## 16. Running (after approval only)

The runner, `research/crowding_study.py`, is to be written after approval. It
must:
- load inputs with `crowding.load_inputs`;
- build the scores with `crowding.signals` and the eligibility with
  `crowding.eligible`;
- build each configuration's book with `crowding.targets`;
- simulate with `backtest.run(W, p1h["close"], 24, cost, band=BAND, trade_at=ta)`
  for each `cost` in `COSTS`;
- report as in section 10 and decide with `crowding.verdict`.

```
RESEARCH_OFFLINE=1 .venv/bin/python -m research.crowding_study
```
