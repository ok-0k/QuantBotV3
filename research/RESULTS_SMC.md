# SMC playbook ("smart money" intraday): results

**Date:** 2026-10-20.

| | |
|---|---|
| Pre-registration | `research/PREREG_SMC.md`: design `8e0af86`, revision 1 `1fc108a`, approved by the PM |
| Runner | `research/smc_study.py`, commit `2151577`, committed with its synthetic end-to-end test before any real-data run |
| Run of record | `research/runs/smc_run1_20261020T060848Z/`, committed unchanged in `901bff5`. It is the only real-data run; there was no smoke test and no re-run |
| Data | Offline cache (`RESEARCH_OFFLINE=1`), Binance spot 5m klines, END 2026-10-07 00:00 UTC, 50 symbols, 943 NY trading days. Fingerprint `eafd756afa4c3c6dbe2d47b78a3e74a1c435edfede1f7180d10242a4c7d4fc15` |
| Runtime | 247 s under `nice -n 10`, exit code 0, no warnings |

All numbers are **R per trade net of base costs** unless stated (fee 0.10% per
fill, +0.08% slippage on market-type fills, 0.03%/day borrow on shorts).
Expectancy is the unweighted mean net R per trade. t is the day-clustered
t-stat (PREREG §10). OOS runs from 2025-04-01 to 2026-10-07: H1 to 2026-01-01,
then H2.

## Short answer

**No decision unit passes.** All 8 short-side configurations are
**rejected** under the written rules. None even clears the in-sample gate:
every short unit has a negative IS expectancy. The best is −0.003R
(`smc_5m_tp2_nobias`).

- **Too few trades to be decided.** The playbook is very selective on this
  universe. The most active short unit filled 148 orders in 943 days across
  50 coins: 87 IS and 61 OOS. Every unit fails `oos_n` (fewer than 100 OOS
  trades) and `oos_t` (no OOS t-stat above +0.81, against a bar of 2.50).
- **Positive OOS means are noise.** The three short units with a positive
  OOS mean (5m only: +0.15R to +0.41R) all have:
  - a negative IS mean;
  - a negative OOS H2;
  - one symbol carrying 55–74% of the OOS net R;
  - one single NY day carrying more than all of it (110–159%: the other days
    lost in total).
  That is noise on 20–61 trades, not an edge.
- **Costs consume what gross edge there is.** Cost runs about 0.2–0.4R per
  trade (OOS 0.21–0.40R, IS 0.20–0.26R). That is in line with the
  pre-registered 2× bar (about 0.25R on 15m, about 0.45R on 5m).
  - On 15m the shorts lose even before costs in sample (gross IS expectancy
    −0.17R to −0.43R).
  - On 5m the gross IS expectancy is +0.04R to +0.26R, and costs turn it
    negative.

**Verdict: the playbook fails as pre-registered.** Nothing goes to an
independent re-run, a forward test or an executor, and nothing is re-tuned.
That covers both entry timeframes, both management modes, and the bias gate
on and off, for shorts, longs and both together.

**Sanity counters: all three are 0** across all 96 simulations
(`fvg_invalid` cancels 0, two-loss binds 0, `spot_tradable` mismatches 0), so
the verdicts stand.

## The 8 decision units (config × short): binding rules

✓ = passes, ✗ = fails. Base costs and per-symbol limits, except `oos_2x_nonneg`
(2× costs) and `port_exp_pos` (portfolio-scope limits, short, OOS, base cost).

| unit (short) | `is_exp_pos` | `oos_exp_min` | `oos_t` | `oos_n` | `oos_h1_pos` | `oos_h2_pos` | `oos_2x_nonneg` | `sym_share` | `day_share` | `oos_exp_w_pos` | `port_exp_pos` | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `smc_5m_pbe_bias` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | 4 of 11 |
| `smc_5m_pbe_nobias` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | 2 of 11 |
| `smc_5m_tp2_bias` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | 5 of 11 |
| `smc_5m_tp2_nobias` | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | 3 of 11 |
| `smc_15m_pbe_bias` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0 of 11 |
| `smc_15m_pbe_nobias` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0 of 11 |
| `smc_15m_tp2_bias` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 1 of 11 |
| `smc_15m_tp2_nobias` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 1 of 11 |

Number of the 8 short units passing each rule:

| rule | passing |
|---|---|
| `is_exp_pos` | 0 |
| `oos_exp_min` | 3 |
| `oos_t` | 0 |
| `oos_n` | 0 |
| `oos_h1_pos` | 5 |
| `oos_h2_pos` | 1 |
| `oos_2x_nonneg` | 1 |
| `sym_share` | 0 |
| `day_share` | 0 |
| `oos_exp_w_pos` | 4 |
| `port_exp_pos` | 2 |

**Verdict per the written rules: all 8 rejected.** No unit passes more than
5 of the 11 rules.

## The 8 decision units: metrics

- **Top-symbol share and top-day share** are a symbol's (or a NY day's) net
  R divided by the unit's total OOS net R. They are blank (—) when the total
  is ≤ 0, and a share above 1 means the other symbols or days lost in total.
- **Max drawdown in R** comes from cumulative net R in exit-time order.
- **Gross** is the zero-cost expectancy of the same trades; cost/trade is
  gross minus net.
- **Portfolio exp** is the OOS expectancy under the portfolio-scope limits
  (one account, 2 orders a day), with its trade count in brackets.

| unit (short) | IS exp | OOS n | OOS exp | OOS t | H1 exp | H2 exp | 2× exp | risk-wtd exp | portfolio exp (n) | top-symbol share | top-day share | OOS max DD (R) | gross exp | cost/trade | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `smc_5m_pbe_bias` | -0.194 | 20 | +0.174 | +0.48 | +0.333 | -0.302 | -0.182 | +0.174 | +0.047 (19) | +0.74 | +1.59 | -3.8 | +0.531 | +0.357 | rejected |
| `smc_5m_pbe_nobias` | -0.075 | 61 | -0.038 | -0.19 | +0.116 | -0.219 | -0.418 | +0.032 | -0.260 (50) | — | — | -10.9 | +0.343 | +0.381 | rejected |
| `smc_5m_tp2_bias` | -0.145 | 20 | +0.412 | +0.81 | +0.717 | -0.504 | +0.034 | +0.412 | +0.196 (19) | +0.55 | +1.10 | -3.2 | +0.789 | +0.377 | rejected |
| `smc_5m_tp2_nobias` | -0.003 | 61 | +0.153 | +0.55 | +0.390 | -0.127 | -0.244 | +0.236 | -0.134 (50) | +0.69 | +1.18 | -9.9 | +0.549 | +0.397 | rejected |
| `smc_15m_pbe_bias` | -0.599 | 6 | -0.128 | -0.25 | -0.234 | -0.076 | -0.414 | -0.128 | -0.128 (6) | — | — | -1.7 | +0.157 | +0.286 | rejected |
| `smc_15m_pbe_nobias` | -0.365 | 22 | -0.168 | -0.69 | -0.017 | -0.294 | -0.382 | -0.159 | -0.120 (19) | — | — | -6.0 | +0.047 | +0.215 | rejected |
| `smc_15m_tp2_bias` | -0.641 | 6 | -0.030 | -0.05 | -0.284 | +0.096 | -0.319 | -0.030 | -0.030 (6) | — | — | -1.7 | +0.258 | +0.289 | rejected |
| `smc_15m_tp2_nobias` | -0.408 | 22 | -0.114 | -0.42 | +0.042 | -0.243 | -0.335 | -0.096 | -0.058 (19) | — | — | -5.8 | +0.108 | +0.222 | rejected |

## Sanity counters

| counter | total over 8 configs × 3 sides × 2 scopes × 2 costs | must be |
|---|---|---|
| `fvg_invalid` cancels | 0 | 0 |
| two-loss rule binds | 0 | 0 |
| `spot_tradable` mismatches (symbol-days with a placed order) | 0 | 0 |

No simulation had a non-zero counter, so the verdicts are not frozen.

## Why so few trades: the funnel

All symbols, 943 trading days, before the bias gate and the limits
(`summary.json` → `funnel`):

| stage | 5m long | 5m short | 15m long | 15m short |
|---|---|---|---|---|
| marked levels first traded through on the day | 139,578 | 133,854 | 97,314 | 92,124 |
| … breakout or no reclaim (no sweep) | 50,659 | 50,724 | 30,875 | 31,123 |
| valid displacement MSS after a sweep | 4,000 | 3,590 | 3,105 | 2,862 |
| … with placement inside 09:30–11:00 | 1,142 | 943 | 731 | 683 |
| … dropped: nearest draw on liquidity < 2R | 597 | 471 | 507 | 460 |
| setups (all required items) | 542 | 472 | 219 | 221 |
| … grade A+ / B / below 7 | 135 / 358 / 49 | 117 / 315 / 40 | 55 / 153 / 11 | 76 / 132 / 13 |

**Orders (short side, per-symbol limits, base cost, IS + OOS):**

| config | orders placed | filled | cancelled: TP1 reached first | cancelled: killzone end / expired | refused by the bias gate |
|---|---|---|---|---|---|
| `smc_5m_*_bias` | 119 | 44 (37%) | 64 (54%) | 11 / 0 | 353 |
| `smc_5m_*_nobias` | 432 | 148 (34%) | 241 (56%) | 42 / 1 | 40 |
| `smc_15m_*_bias` | 77 | 18 (23%) | 44 (57%) | 15 / 0 | 144 |
| `smc_15m_*_nobias` | 208 | 50 (24%) | 98 (47%) | 59 / 1 | 13 |

Management mode changes only exits, so `pbe` and `tp2` place and fill the
same orders.

**Two things shrink the sample.**
1. **The move leaves without a retrace.** About half of all limit orders at
   the FVG's CE are cancelled because price reaches +1R (TP1) before it comes
   back to the CE. The playbook's own rule: the move left without us.
2. **The bias gate almost never says Bullish or Bearish** (next section).

## Bias: the share of days per bias, and Neutral-day trades

Eligible symbol-days by bias at 09:30 NY (§5.4):

| period | Bullish | Bearish | Neutral | symbol-days |
|---|---|---|---|---|
| IS | 489 (1.9%) | 604 (2.4%) | 24,170 (95.7%) | 25,263 |
| OOS H1 | 159 (2.0%) | 267 (3.4%) | 7,427 (94.6%) | 7,853 |
| OOS H2 | 161 (2.6%) | 122 (2.0%) | 5,910 (95.4%) | 6,193 |
| all | 809 (2.1%) | 993 (2.5%) | 37,507 (95.4%) | 39,309 |

- **Neutralised days.** On a further 100 Bullish/Bearish symbol-days, a 15m
  MSS against the bias turned the day Neutral during the killzone (§5.5).
- **Ineligible symbol-days** (§3.6):
  - 3,268 had missing bars before the session. These are almost all days
    before a listing or after a delisting, plus the 2023-03-24 outage.
  - 4,414 were illiquid (under $2M a day).
  - 117 were listed for less than 30 days.
  - 42 were missing a level.

**Why the bias is so rarely directional.** The bias table needs agreeing
timeframes together with either an overnight sweep or price in the right
half of the daily range. A daily "up" state starts with a close above the
last daily swing high, which puts price at the top of the daily dealing
range, in premium, so "up + discount" is rare. I flagged this to the PM from
synthetic data before the run. The rule was kept as approved.

**Neutral-day share of bias-on trades** (per-symbol limits, base cost, IS + OOS):

| bias-on config | short | long | both |
|---|---|---|---|
| `smc_5m_pbe_bias` / `smc_5m_tp2_bias` | 41 of 44 (93%) | 46 of 53 (87%) | 87 of 97 (90%) |
| `smc_15m_pbe_bias` / `smc_15m_tp2_bias` | 18 of 18 (100%) | 16 of 16 (100%) | 34 of 34 (100%) |
| all bias-on trades (all cells summed) | | | 484 of 524 (92.4%) |

In OOS alone the short shares are 19 of 20 (5m) and 6 of 6 (15m). The
"bias as written" configurations are therefore mostly "A+ setups on Neutral
days at half risk". The bias gate mainly acts as a filter to A+ grades.

## Multiple testing

There are 8 decision units, strongly correlated. The binding t bar is 2.50,
the Bonferroni level for 8 units at one-sided 5% (PREREG §12).
- No short unit comes near it: the best OOS t is +0.81.
- No short unit reaches even an uncorrected t of 2.
- Across four pre-registered studies, this universe has now had 75
  decision units fail out of sample: 17 price strategies, 2 carry, 48
  crowding, and these 8. That excludes the 16 informational long/both units
  here.

There is nothing to correct for: nothing came close.

## Context only: long and both (cannot produce a candidate)

These are not decision units (PREREG §11). They are shown for completeness,
with per-symbol limits and base cost; "2×" is OOS at 2× costs.

| config | sides | IS exp | IS n | OOS n | OOS exp | OOS t | 2× exp | portfolio exp | OOS max DD (R) | checks failed (of 11) |
|---|---|---|---|---|---|---|---|---|---|---|
| `smc_5m_pbe_bias` | long | -0.339 | 27 | 26 | -0.208 | -0.76 | -0.485 | -0.357 | -8.5 | 10 |
| `smc_5m_pbe_nobias` | long | -0.288 | 98 | 95 | -0.437 | -3.19 | -0.727 | -0.455 | -41.8 | 11 |
| `smc_5m_tp2_bias` | long | -0.460 | 27 | 26 | -0.204 | -0.59 | -0.491 | -0.381 | -10.6 | 10 |
| `smc_5m_tp2_nobias` | long | -0.308 | 98 | 95 | -0.445 | -2.66 | -0.743 | -0.501 | -42.7 | 11 |
| `smc_15m_pbe_bias` | long | +0.042 | 9 | 7 | -0.196 | -0.46 | -0.402 | -0.196 | -3.3 | 10 |
| `smc_15m_pbe_nobias` | long | -0.056 | 40 | 24 | -0.282 | -1.34 | -0.490 | -0.358 | -8.7 | 11 |
| `smc_15m_tp2_bias` | long | +0.060 | 9 | 7 | -0.183 | -0.32 | -0.395 | -0.183 | -4.2 | 9 |
| `smc_15m_tp2_nobias` | long | -0.112 | 40 | 24 | -0.254 | -1.00 | -0.468 | -0.324 | -9.1 | 11 |
| `smc_5m_pbe_bias` | both | -0.271 | 51 | 46 | -0.042 | -0.19 | -0.354 | -0.160 | -10.0 | 10 |
| `smc_5m_pbe_nobias` | both | -0.188 | 185 | 156 | -0.281 | -2.46 | -0.606 | -0.368 | -45.3 | 10 |
| `smc_5m_tp2_bias` | both | -0.311 | 51 | 46 | +0.064 | +0.22 | -0.263 | -0.100 | -13.1 | 9 |
| `smc_5m_tp2_nobias` | both | -0.165 | 185 | 156 | -0.211 | -1.41 | -0.548 | -0.337 | -43.5 | 10 |
| `smc_15m_pbe_bias` | both | -0.324 | 21 | 13 | -0.165 | -0.52 | -0.408 | -0.165 | -4.0 | 11 |
| `smc_15m_pbe_nobias` | both | -0.183 | 68 | 46 | -0.227 | -1.45 | -0.439 | -0.245 | -13.7 | 11 |
| `smc_15m_tp2_bias` | both | -0.340 | 21 | 13 | -0.112 | -0.29 | -0.360 | -0.112 | -4.9 | 11 |
| `smc_15m_tp2_nobias` | both | -0.234 | 68 | 46 | -0.187 | -1.03 | -0.404 | -0.197 | -13.2 | 11 |

- **Longs are worse than shorts.** Every long unit has a negative OOS
  expectancy, from −0.18R to −0.45R. The two 5m long units without the bias
  gate reach OOS t −3.19 and −2.66, with drawdowns over 40R.
- **No combined "both" unit has a positive IS expectancy.** The only positive
  OOS mean among them (+0.064R, `smc_5m_tp2_bias` both, t +0.22) fails 9 of
  11 rules.

## Every simulation (config × side × scope × cost)

All 96 simulations, with the IS and OOS summaries. Per-period details
(median, win rate, profit factor, gross expectancy, cost per trade, exit
reasons, worst day) are in `summary.json`. Every trade is in
`trades.csv.gz`.

| config | sides | scope | cost | IS n | IS exp | OOS n | OOS exp | OOS t | H1 exp | H2 exp | OOS max DD (R) | OOS max DD (U) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `smc_5m_pbe_bias` | short | symbol | base | 24 | -0.194 | 20 | +0.174 | +0.48 | +0.333 | -0.302 | -3.8 | -1.9 |
| `smc_5m_pbe_bias` | short | symbol | 2x | 24 | -0.432 | 20 | -0.182 | -0.48 | -0.045 | -0.595 | -6.1 | -3.1 |
| `smc_5m_pbe_bias` | short | portfolio | base | 24 | -0.194 | 19 | +0.047 | +0.16 | +0.171 | -0.302 | -3.8 | -1.9 |
| `smc_5m_pbe_bias` | short | portfolio | 2x | 24 | -0.432 | 19 | -0.311 | -0.98 | -0.210 | -0.595 | -7.2 | -3.6 |
| `smc_5m_pbe_bias` | long | symbol | base | 27 | -0.339 | 26 | -0.208 | -0.76 | +0.158 | -0.900 | -8.5 | -5.2 |
| `smc_5m_pbe_bias` | long | symbol | 2x | 27 | -0.615 | 26 | -0.485 | -1.78 | -0.112 | -1.190 | -13.8 | -7.9 |
| `smc_5m_pbe_bias` | long | portfolio | base | 27 | -0.339 | 20 | -0.357 | -1.78 | -0.023 | -0.860 | -8.8 | -5.4 |
| `smc_5m_pbe_bias` | long | portfolio | 2x | 27 | -0.615 | 20 | -0.632 | -3.03 | -0.281 | -1.158 | -13.8 | -8.2 |
| `smc_5m_pbe_bias` | both | symbol | base | 51 | -0.271 | 46 | -0.042 | -0.19 | +0.240 | -0.686 | -10.0 | -5.3 |
| `smc_5m_pbe_bias` | both | symbol | 2x | 51 | -0.529 | 46 | -0.354 | -1.60 | -0.081 | -0.977 | -17.3 | -9.7 |
| `smc_5m_pbe_bias` | both | portfolio | base | 51 | -0.271 | 39 | -0.160 | -0.90 | +0.082 | -0.645 | -9.0 | -5.0 |
| `smc_5m_pbe_bias` | both | portfolio | 2x | 51 | -0.529 | 39 | -0.476 | -2.55 | -0.243 | -0.942 | -19.6 | -11.1 |
| `smc_5m_pbe_nobias` | short | symbol | base | 87 | -0.075 | 61 | -0.038 | -0.19 | +0.116 | -0.219 | -10.9 | -6.3 |
| `smc_5m_pbe_nobias` | short | symbol | 2x | 87 | -0.330 | 61 | -0.418 | -1.84 | -0.323 | -0.531 | -28.6 | -15.3 |
| `smc_5m_pbe_nobias` | short | portfolio | base | 70 | -0.130 | 50 | -0.260 | -1.37 | -0.154 | -0.376 | -15.5 | -7.5 |
| `smc_5m_pbe_nobias` | short | portfolio | 2x | 70 | -0.403 | 50 | -0.662 | -2.96 | -0.614 | -0.713 | -34.6 | -20.0 |
| `smc_5m_pbe_nobias` | long | symbol | base | 98 | -0.288 | 95 | -0.437 | -3.19 | -0.390 | -0.540 | -41.8 | -24.1 |
| `smc_5m_pbe_nobias` | long | symbol | 2x | 98 | -0.575 | 95 | -0.727 | -5.29 | -0.662 | -0.868 | -69.1 | -40.8 |
| `smc_5m_pbe_nobias` | long | portfolio | base | 81 | -0.448 | 62 | -0.455 | -4.17 | -0.455 | -0.455 | -29.3 | -18.1 |
| `smc_5m_pbe_nobias` | long | portfolio | 2x | 81 | -0.749 | 62 | -0.755 | -6.38 | -0.731 | -0.792 | -47.4 | -28.6 |
| `smc_5m_pbe_nobias` | both | symbol | base | 185 | -0.188 | 156 | -0.281 | -2.46 | -0.219 | -0.385 | -45.3 | -26.8 |
| `smc_5m_pbe_nobias` | both | symbol | 2x | 185 | -0.460 | 156 | -0.606 | -5.02 | -0.548 | -0.705 | -95.2 | -54.9 |
| `smc_5m_pbe_nobias` | both | portfolio | base | 149 | -0.291 | 112 | -0.368 | -3.55 | -0.333 | -0.415 | -43.3 | -24.9 |
| `smc_5m_pbe_nobias` | both | portfolio | 2x | 149 | -0.578 | 112 | -0.713 | -6.02 | -0.684 | -0.753 | -80.5 | -48.1 |
| `smc_5m_tp2_bias` | short | symbol | base | 24 | -0.145 | 20 | +0.412 | +0.81 | +0.717 | -0.504 | -3.2 | -1.6 |
| `smc_5m_tp2_bias` | short | symbol | 2x | 24 | -0.389 | 20 | +0.034 | +0.07 | +0.314 | -0.804 | -5.8 | -2.9 |
| `smc_5m_tp2_bias` | short | portfolio | base | 24 | -0.145 | 19 | +0.196 | +0.52 | +0.446 | -0.504 | -3.2 | -1.6 |
| `smc_5m_tp2_bias` | short | portfolio | 2x | 24 | -0.389 | 19 | -0.184 | -0.46 | +0.037 | -0.804 | -6.4 | -3.2 |
| `smc_5m_tp2_bias` | long | symbol | base | 27 | -0.460 | 26 | -0.204 | -0.59 | +0.255 | -1.070 | -10.6 | -6.1 |
| `smc_5m_tp2_bias` | long | symbol | 2x | 27 | -0.747 | 26 | -0.491 | -1.46 | -0.029 | -1.364 | -14.6 | -8.3 |
| `smc_5m_tp2_bias` | long | portfolio | base | 27 | -0.460 | 20 | -0.381 | -1.48 | +0.065 | -1.051 | -9.7 | -5.8 |
| `smc_5m_tp2_bias` | long | portfolio | 2x | 27 | -0.747 | 20 | -0.662 | -2.49 | -0.200 | -1.354 | -15.1 | -8.8 |
| `smc_5m_tp2_bias` | both | symbol | base | 51 | -0.311 | 46 | +0.064 | +0.22 | +0.471 | -0.867 | -13.1 | -6.9 |
| `smc_5m_tp2_bias` | both | symbol | 2x | 51 | -0.578 | 46 | -0.263 | -0.89 | +0.132 | -1.164 | -17.9 | -9.4 |
| `smc_5m_tp2_bias` | both | portfolio | base | 51 | -0.311 | 39 | -0.100 | -0.44 | +0.270 | -0.840 | -11.9 | -6.2 |
| `smc_5m_tp2_bias` | both | portfolio | 2x | 51 | -0.578 | 39 | -0.429 | -1.81 | -0.072 | -1.143 | -19.1 | -10.9 |
| `smc_5m_tp2_nobias` | short | symbol | base | 87 | -0.003 | 61 | +0.153 | +0.55 | +0.390 | -0.127 | -9.9 | -6.1 |
| `smc_5m_tp2_nobias` | short | symbol | 2x | 87 | -0.267 | 61 | -0.244 | -0.81 | -0.071 | -0.448 | -19.8 | -11.3 |
| `smc_5m_tp2_nobias` | short | portfolio | base | 70 | -0.083 | 50 | -0.134 | -0.54 | -0.007 | -0.271 | -11.0 | -6.4 |
| `smc_5m_tp2_nobias` | short | portfolio | 2x | 70 | -0.365 | 50 | -0.551 | -1.95 | -0.491 | -0.616 | -30.7 | -17.5 |
| `smc_5m_tp2_nobias` | long | symbol | base | 98 | -0.308 | 95 | -0.445 | -2.66 | -0.317 | -0.722 | -42.7 | -26.1 |
| `smc_5m_tp2_nobias` | long | symbol | 2x | 98 | -0.609 | 95 | -0.743 | -4.50 | -0.597 | -1.059 | -70.5 | -41.7 |
| `smc_5m_tp2_nobias` | long | portfolio | base | 81 | -0.508 | 62 | -0.501 | -3.90 | -0.418 | -0.632 | -32.5 | -19.9 |
| `smc_5m_tp2_nobias` | long | portfolio | 2x | 81 | -0.822 | 62 | -0.809 | -5.95 | -0.701 | -0.979 | -51.0 | -31.1 |
| `smc_5m_tp2_nobias` | both | symbol | base | 185 | -0.165 | 156 | -0.211 | -1.41 | -0.079 | -0.435 | -43.5 | -28.3 |
| `smc_5m_tp2_nobias` | both | symbol | 2x | 185 | -0.448 | 156 | -0.548 | -3.53 | -0.420 | -0.764 | -86.0 | -48.6 |
| `smc_5m_tp2_nobias` | both | portfolio | base | 149 | -0.302 | 112 | -0.337 | -2.54 | -0.251 | -0.451 | -41.0 | -23.2 |
| `smc_5m_tp2_nobias` | both | portfolio | 2x | 149 | -0.601 | 111 | -0.705 | -4.79 | -0.635 | -0.798 | -80.0 | -47.9 |
| `smc_15m_pbe_bias` | short | symbol | base | 12 | -0.599 | 6 | -0.128 | -0.25 | -0.234 | -0.076 | -1.7 | -0.8 |
| `smc_15m_pbe_bias` | short | symbol | 2x | 12 | -0.807 | 6 | -0.414 | -0.72 | -0.428 | -0.407 | -3.0 | -1.5 |
| `smc_15m_pbe_bias` | short | portfolio | base | 7 | -0.684 | 6 | -0.128 | -0.25 | -0.234 | -0.076 | -1.7 | -0.8 |
| `smc_15m_pbe_bias` | short | portfolio | 2x | 7 | -0.969 | 6 | -0.414 | -0.72 | -0.428 | -0.407 | -3.0 | -1.5 |
| `smc_15m_pbe_bias` | long | symbol | base | 9 | +0.042 | 7 | -0.196 | -0.46 | -0.142 | -0.269 | -3.3 | -1.6 |
| `smc_15m_pbe_bias` | long | symbol | 2x | 9 | -0.124 | 7 | -0.402 | -0.92 | -0.331 | -0.497 | -4.5 | -2.3 |
| `smc_15m_pbe_bias` | long | portfolio | base | 8 | +0.049 | 7 | -0.196 | -0.46 | -0.142 | -0.269 | -3.3 | -1.6 |
| `smc_15m_pbe_bias` | long | portfolio | 2x | 8 | -0.129 | 7 | -0.402 | -0.92 | -0.331 | -0.497 | -4.5 | -2.3 |
| `smc_15m_pbe_bias` | both | symbol | base | 21 | -0.324 | 13 | -0.165 | -0.52 | -0.173 | -0.158 | -4.0 | -2.0 |
| `smc_15m_pbe_bias` | both | symbol | 2x | 21 | -0.514 | 13 | -0.408 | -1.21 | -0.363 | -0.446 | -7.0 | -3.5 |
| `smc_15m_pbe_bias` | both | portfolio | base | 15 | -0.293 | 13 | -0.165 | -0.52 | -0.173 | -0.158 | -4.0 | -2.0 |
| `smc_15m_pbe_bias` | both | portfolio | 2x | 15 | -0.521 | 13 | -0.408 | -1.21 | -0.363 | -0.446 | -7.0 | -3.5 |
| `smc_15m_pbe_nobias` | short | symbol | base | 28 | -0.365 | 22 | -0.168 | -0.69 | -0.017 | -0.294 | -6.0 | -3.8 |
| `smc_15m_pbe_nobias` | short | symbol | 2x | 28 | -0.560 | 22 | -0.382 | -1.50 | -0.200 | -0.534 | -9.9 | -6.4 |
| `smc_15m_pbe_nobias` | short | portfolio | base | 20 | -0.306 | 19 | -0.120 | -0.51 | +0.123 | -0.339 | -4.4 | -3.0 |
| `smc_15m_pbe_nobias` | short | portfolio | 2x | 20 | -0.539 | 19 | -0.337 | -1.33 | -0.051 | -0.595 | -7.6 | -4.4 |
| `smc_15m_pbe_nobias` | long | symbol | base | 40 | -0.056 | 24 | -0.282 | -1.34 | -0.206 | -0.388 | -8.7 | -6.0 |
| `smc_15m_pbe_nobias` | long | symbol | 2x | 40 | -0.247 | 24 | -0.490 | -2.25 | -0.387 | -0.635 | -13.5 | -9.0 |
| `smc_15m_pbe_nobias` | long | portfolio | base | 23 | -0.226 | 21 | -0.358 | -1.55 | -0.331 | -0.388 | -9.4 | -6.3 |
| `smc_15m_pbe_nobias` | long | portfolio | 2x | 23 | -0.436 | 21 | -0.574 | -2.40 | -0.518 | -0.635 | -13.7 | -9.1 |
| `smc_15m_pbe_nobias` | both | symbol | base | 68 | -0.183 | 46 | -0.227 | -1.45 | -0.127 | -0.336 | -13.7 | -8.9 |
| `smc_15m_pbe_nobias` | both | symbol | 2x | 68 | -0.376 | 46 | -0.439 | -2.66 | -0.309 | -0.580 | -23.0 | -15.0 |
| `smc_15m_pbe_nobias` | both | portfolio | base | 43 | -0.264 | 40 | -0.245 | -1.49 | -0.127 | -0.364 | -13.0 | -7.9 |
| `smc_15m_pbe_nobias` | both | portfolio | 2x | 43 | -0.484 | 40 | -0.461 | -2.67 | -0.308 | -0.615 | -21.3 | -13.4 |
| `smc_15m_tp2_bias` | short | symbol | base | 12 | -0.641 | 6 | -0.030 | -0.05 | -0.284 | +0.096 | -1.7 | -0.9 |
| `smc_15m_tp2_bias` | short | symbol | 2x | 12 | -0.852 | 6 | -0.319 | -0.49 | -0.487 | -0.235 | -3.1 | -1.6 |
| `smc_15m_tp2_bias` | short | portfolio | base | 7 | -0.756 | 6 | -0.030 | -0.05 | -0.284 | +0.096 | -1.7 | -0.9 |
| `smc_15m_tp2_bias` | short | portfolio | 2x | 7 | -1.046 | 6 | -0.319 | -0.49 | -0.487 | -0.235 | -3.1 | -1.6 |
| `smc_15m_tp2_bias` | long | symbol | base | 9 | +0.060 | 7 | -0.183 | -0.32 | +0.126 | -0.594 | -4.2 | -2.1 |
| `smc_15m_tp2_bias` | long | symbol | 2x | 9 | -0.115 | 7 | -0.395 | -0.68 | -0.063 | -0.837 | -5.5 | -2.8 |
| `smc_15m_tp2_bias` | long | portfolio | base | 8 | +0.070 | 7 | -0.183 | -0.32 | +0.126 | -0.594 | -4.2 | -2.1 |
| `smc_15m_tp2_bias` | long | portfolio | 2x | 8 | -0.120 | 7 | -0.395 | -0.68 | -0.063 | -0.837 | -5.5 | -2.8 |
| `smc_15m_tp2_bias` | both | symbol | base | 21 | -0.340 | 13 | -0.112 | -0.29 | -0.011 | -0.199 | -4.9 | -2.5 |
| `smc_15m_tp2_bias` | both | symbol | 2x | 21 | -0.536 | 13 | -0.360 | -0.87 | -0.205 | -0.493 | -7.4 | -3.7 |
| `smc_15m_tp2_bias` | both | portfolio | base | 15 | -0.316 | 13 | -0.112 | -0.29 | -0.011 | -0.199 | -4.9 | -2.5 |
| `smc_15m_tp2_bias` | both | portfolio | 2x | 15 | -0.552 | 13 | -0.360 | -0.87 | -0.205 | -0.493 | -7.4 | -3.7 |
| `smc_15m_tp2_nobias` | short | symbol | base | 28 | -0.408 | 22 | -0.114 | -0.42 | +0.042 | -0.243 | -5.8 | -3.8 |
| `smc_15m_tp2_nobias` | short | symbol | 2x | 28 | -0.609 | 22 | -0.335 | -1.20 | -0.155 | -0.485 | -9.8 | -6.4 |
| `smc_15m_tp2_nobias` | short | portfolio | base | 20 | -0.365 | 19 | -0.058 | -0.22 | +0.188 | -0.279 | -4.4 | -3.0 |
| `smc_15m_tp2_nobias` | short | portfolio | 2x | 20 | -0.607 | 19 | -0.283 | -1.01 | -0.001 | -0.536 | -7.3 | -4.5 |
| `smc_15m_tp2_nobias` | long | symbol | base | 40 | -0.112 | 24 | -0.254 | -1.00 | -0.139 | -0.413 | -9.1 | -6.7 |
| `smc_15m_tp2_nobias` | long | symbol | 2x | 40 | -0.310 | 24 | -0.468 | -1.81 | -0.324 | -0.669 | -14.0 | -9.8 |
| `smc_15m_tp2_nobias` | long | portfolio | base | 23 | -0.284 | 21 | -0.324 | -1.15 | -0.242 | -0.413 | -9.8 | -7.0 |
| `smc_15m_tp2_nobias` | long | portfolio | 2x | 23 | -0.498 | 21 | -0.544 | -1.88 | -0.431 | -0.669 | -14.2 | -9.9 |
| `smc_15m_tp2_nobias` | both | symbol | base | 68 | -0.234 | 46 | -0.187 | -1.03 | -0.064 | -0.321 | -13.2 | -8.8 |
| `smc_15m_tp2_nobias` | both | symbol | 2x | 68 | -0.433 | 46 | -0.404 | -2.15 | -0.254 | -0.569 | -22.8 | -15.1 |
| `smc_15m_tp2_nobias` | both | portfolio | base | 43 | -0.322 | 40 | -0.197 | -1.02 | -0.049 | -0.346 | -12.5 | -7.8 |
| `smc_15m_tp2_nobias` | both | portfolio | 2x | 43 | -0.549 | 40 | -0.420 | -2.10 | -0.237 | -0.603 | -21.1 | -13.5 |

## What this means

- **This is a pre-registered negative.**
  - The playbook's entry model rarely fires on crypto spot: 18 to 148 short
    fills in 943 trading days across 50 coins, or 0.02 to 0.16 a day.
  - When it does fire, it shows no in-sample edge after costs.
  - The few out-of-sample positives rest on one or two symbols and days.
- **Nothing goes forward.** Nothing goes to the independent re-run, the
  forward paper test or an executor, and `VALIDATED_STRATEGIES` stays empty.
- **No re-tuning.** Under PREREG §15 the rules, constants and thresholds stay
  as they are. Any follow-up idea would need a new pre-registration and is
  not proposed here.
- **Possible follow-up, for the owner and PM to judge.** The two mechanisms
  that shrink the sample are the TP1-before-fill cancellations and the
  near-permanent Neutral bias. Both are properties of the playbook's rules as
  written, not of the code. If the owner wants to try a variant (for
  example, entering at the FVG edge instead of the CE, or a different bias
  definition), it is a new study with a new pre-registration and a new OOS
  window.
- **Data coverage.** The 5m cache (1,095 MB) stays in
  `/home/admin/trading_data/research/klines_5m/` for any later independent
  re-run of this record.
