# Crowding signals (funding, premium): results

**Date:** 2026-10-14.

| | |
|---|---|
| Pre-registration | `research/PREREG_FUNDING.md`, commit `98e8cec`, approved by the PM as written |
| Runner | `research/crowding_study.py`, commit `8b4ec7a`, committed before it read any real data |
| Run of record | `research/runs/crowding_run1_20261014T035310Z/`, committed unchanged in `a529c9f`. It is the only real-data run; there was no re-run |
| Data | Offline cache (`RESEARCH_OFFLINE=1`), END 2026-10-07 00:00 UTC, 50 symbols, 35,208 1h bars. Fingerprint `4a7bd3e083fc984323a8ad0c444b73f280ca472135ead750cf9b5f9dc84042e0` |
| Runtime | 686 s, exit code 0, no warnings |

## Short answer

**No configuration passes. None of the 48 even becomes an in-sample
candidate.**
- **In sample:** every IS Sharpe is negative. The best is −0.38
  (`fund_lvl_ls_k3_72h`).
- **Out of sample:** no OOS alpha is positive at base costs. The best is
  −4.0% a year (`fund_lvl_ls_k3_24h`).

Under the written rules, crowding information does not predict which coins
underperform once costs are paid. That covers funding and premium, level and
change, short-only and long/short books, and rebalancing every 8h, 24h or
72h. **Verdict: all 48 rejected.** Nothing goes on to an independent re-run or
to a rebalancing executor.

## Rule outcomes at a glance

Number of the 48 configurations that pass each rule:

| rule | passing |
|---|---|
| IS Sharpe ≥ 1.0 (`is_sharpe`) | 0 |
| IS alpha t ≥ 2.0 (`is_alpha_t`) | 0 |
| OOS Sharpe ≥ 0.75 (`oos_sharpe`) | 0 |
| OOS alpha > 0 (`oos_alpha_pos`) | 0 |
| OOS Sharpe at 2× costs ≥ 0.5 (`oos_sharpe_2x`) | 0 |
| a stable neighbour (`neighbour`) | 0 |
| OOS alpha ≥ 10% a year (`oos_alpha_min`) | 0 |
| alpha > 0 in OOS H1, 2025-04-01 → 2026-01-01 (`oos_h1_alpha`) | 10 |
| alpha > 0 in OOS H2, 2026-01-01 → 2026-10-07 (`oos_h2_alpha`) | 2 |
| Sharpe ≥ 0.5 at 0.10%/day borrow (`borrow_sharpe`) | 0 |
| alpha > 0 at 0.10%/day borrow (`borrow_alpha`) | 0 |
| at most 5% of OOS rebalances skipped (`coverage`) | 48 |
| drop-one-symbol check (`drop_one`) | not run: no configuration reached it |

The six best configurations by OOS alpha (base costs):

| config | OOS alpha/yr | OOS Sharpe | IS Sharpe | IS alpha t | H1 alpha | H2 alpha | rules failed | verdict |
|---|---|---|---|---|---|---|---|---|
| `fund_lvl_ls_k3_24h` | −4.0% | −0.09 | −1.30 | −1.99 | +21.4% | −28.3% | 10 of 12 run | rejected |
| `prem_lvl_ls_k3_72h` | −11.4% | −0.33 | −0.85 | −1.31 | −56.5% | +29.2% | 10 of 12 | rejected |
| `prem_lvl_so_k3_72h` | −12.2% | −0.18 | −1.08 | −1.40 | −40.9% | +12.3% | 10 of 12 | rejected |
| `fund_lvl_ls_k5_72h` | −20.2% | −0.78 | −0.58 | −0.91 | −11.2% | −28.5% | 11 of 12 | rejected |
| `prem_lvl_ls_k5_72h` | −23.3% | −0.95 | −1.16 | −1.77 | −43.7% | −5.3% | 11 of 12 | rejected |
| `fund_chg_ls_k5_72h` | −24.7% | −0.85 | −1.40 | −2.07 | +7.7% | −57.3% | 10 of 12 | rejected |

## All 48 configurations: metrics

- **Cost basis:** base costs unless noted. Base is 0.10% fee + 0.08%
  slippage per side and 0.03%/day borrow on short notional. "2×" doubles all
  three. "Borrow" means 0.10%/day borrow, with fees at base.
- **Alpha:** measured against the equal-weight universe, annualised, from
  daily returns (`research.backtest.metrics`).
- **Halves:** H1 and H2 are the two OOS halves, each with its own beta.
- **Skipped:** the share of OOS rebalances skipped because fewer than 10 coins
  were eligible.
- **Zero-cost OOS alpha:** a diagnostic only (section 10 of the
  pre-registration). It is not a rule.

| config | IS Sharpe | IS alpha/yr | IS alpha t | OOS Sharpe | OOS alpha/yr | OOS alpha t | H1 alpha | H2 alpha | 2× Sharpe | 2× alpha | borrow Sharpe | borrow alpha | skipped | OOS cost/yr | zero-cost OOS alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fund_lvl_so_k3_8h` | -1.29 | -63.8% | -2.51 | -0.65 | -52.5% | -2.03 | +36.6% | -140.5% | -1.90 | -150.2% | -0.99 | -78.4% | 0.0% | 97% | +44.4% |
| `fund_lvl_so_k3_24h` | -1.24 | -61.6% | -2.31 | -0.31 | -27.4% | -1.13 | +37.6% | -91.1% | -1.18 | -96.1% | -0.64 | -53.0% | 0.0% | 69% | +41.1% |
| `fund_lvl_so_k3_72h` | -0.76 | -20.4% | -0.74 | -0.48 | -39.5% | -1.44 | -18.8% | -60.9% | -1.04 | -83.0% | -0.81 | -64.8% | 0.0% | 44% | +3.8% |
| `fund_lvl_so_k5_8h` | -1.27 | -56.6% | -2.68 | -0.77 | -58.9% | -2.91 | +1.5% | -117.6% | -1.95 | -147.3% | -1.12 | -85.2% | 0.0% | 87% | +27.0% |
| `fund_lvl_so_k5_24h` | -1.06 | -40.9% | -1.90 | -0.64 | -50.7% | -2.53 | -4.9% | -95.1% | -1.49 | -114.5% | -0.99 | -76.3% | 0.0% | 61% | +10.5% |
| `fund_lvl_so_k5_72h` | -0.76 | -16.3% | -0.75 | -0.52 | -40.7% | -1.94 | -3.1% | -77.5% | -1.06 | -82.2% | -0.86 | -66.6% | 0.0% | 41% | +0.3% |
| `fund_lvl_ls_k3_8h` | -1.79 | -59.1% | -2.79 | -0.74 | -29.4% | -0.96 | +0.8% | -58.4% | -2.56 | -101.0% | -1.07 | -42.1% | 0.0% | 72% | +42.8% |
| `fund_lvl_ls_k3_24h` | -1.30 | -43.1% | -1.99 | -0.09 | -4.0% | -0.13 | +21.4% | -28.3% | -1.43 | -56.5% | -0.42 | -16.8% | 0.0% | 52% | +52.1% |
| `fund_lvl_ls_k3_72h` | -0.38 | -13.0% | -0.62 | -0.77 | -27.7% | -0.98 | -26.3% | -29.6% | -1.68 | -60.8% | -1.13 | -40.5% | 0.0% | 33% | +5.2% |
| `fund_lvl_ls_k5_8h` | -1.82 | -45.5% | -2.73 | -1.30 | -35.5% | -1.65 | -19.4% | -51.0% | -3.72 | -102.7% | -1.76 | -48.0% | 0.0% | 66% | +30.2% |
| `fund_lvl_ls_k5_24h` | -1.05 | -26.1% | -1.60 | -0.98 | -26.3% | -1.23 | -15.2% | -36.3% | -2.67 | -72.6% | -1.44 | -38.7% | 0.0% | 46% | +20.0% |
| `fund_lvl_ls_k5_72h` | -0.58 | -14.8% | -0.91 | -0.78 | -20.2% | -0.93 | -11.2% | -28.5% | -1.92 | -50.5% | -1.27 | -32.9% | 0.0% | 31% | +10.3% |
| `fund_chg_so_k3_8h` | -2.04 | -133.4% | -4.59 | -1.59 | -141.5% | -3.94 | -152.8% | -131.1% | -3.09 | -276.1% | -1.88 | -167.0% | 0.0% | 135% | -6.9% |
| `fund_chg_so_k3_24h` | -1.12 | -53.0% | -2.00 | -0.58 | -56.0% | -1.62 | -42.9% | -70.1% | -1.53 | -141.8% | -0.87 | -81.9% | 0.0% | 85% | +29.6% |
| `fund_chg_so_k3_72h` | -0.98 | -40.2% | -1.44 | -0.68 | -63.8% | -1.48 | +29.3% | -154.4% | -1.13 | -107.8% | -0.95 | -89.7% | 0.0% | 44% | -20.1% |
| `fund_chg_so_k5_8h` | -2.19 | -132.1% | -6.19 | -2.21 | -175.4% | -7.13 | -141.4% | -207.9% | -3.80 | -303.9% | -2.53 | -200.9% | 0.0% | 130% | -45.9% |
| `fund_chg_so_k5_24h` | -1.77 | -98.6% | -4.47 | -1.08 | -88.6% | -3.68 | -76.0% | -102.2% | -2.13 | -175.2% | -1.40 | -114.6% | 0.0% | 86% | -4.1% |
| `fund_chg_so_k5_72h` | -1.25 | -57.7% | -2.39 | -0.64 | -52.9% | -1.97 | +18.3% | -122.4% | -1.15 | -95.6% | -0.95 | -78.9% | 0.0% | 42% | -10.5% |
| `fund_chg_ls_k3_8h` | -4.64 | -154.9% | -7.02 | -4.23 | -163.9% | -6.20 | -151.3% | -176.8% | -7.83 | -308.9% | -4.56 | -176.7% | 0.0% | 146% | -19.1% |
| `fund_chg_ls_k3_24h` | -1.91 | -64.1% | -2.93 | -0.96 | -41.0% | -1.34 | -53.2% | -30.3% | -3.09 | -132.2% | -1.26 | -53.7% | 0.0% | 91% | +49.8% |
| `fund_chg_ls_k3_72h` | -0.68 | -23.1% | -1.04 | -0.97 | -37.2% | -1.21 | +7.0% | -81.5% | -2.01 | -78.1% | -1.30 | -49.9% | 0.0% | 41% | +3.5% |
| `fund_chg_ls_k5_8h` | -5.62 | -140.1% | -8.60 | -5.61 | -155.1% | -7.85 | -124.9% | -183.9% | -10.23 | -291.3% | -6.08 | -167.8% | 0.0% | 137% | -19.2% |
| `fund_chg_ls_k5_24h` | -3.60 | -92.2% | -5.32 | -2.76 | -76.3% | -3.83 | -71.2% | -81.8% | -5.76 | -165.4% | -3.22 | -89.1% | 0.0% | 89% | +12.3% |
| `fund_chg_ls_k5_72h` | -1.40 | -36.6% | -2.07 | -0.85 | -24.7% | -1.08 | +7.7% | -57.3% | -2.18 | -64.2% | -1.30 | -37.6% | 0.0% | 39% | +14.5% |
| `prem_lvl_so_k3_8h` | -1.68 | -94.8% | -3.43 | -1.51 | -109.3% | -4.46 | -88.6% | -130.0% | -2.78 | -203.3% | -1.85 | -134.8% | 0.0% | 95% | -13.7% |
| `prem_lvl_so_k3_24h` | -1.42 | -71.9% | -2.59 | -1.03 | -78.9% | -2.76 | -43.5% | -112.4% | -1.80 | -137.4% | -1.37 | -104.5% | 0.0% | 59% | -18.0% |
| `prem_lvl_so_k3_72h` | -1.08 | -44.1% | -1.40 | -0.18 | -12.2% | -0.43 | -40.9% | +12.3% | -0.69 | -49.6% | -0.55 | -38.6% | 0.0% | 36% | +24.5% |
| `prem_lvl_so_k5_8h` | -1.75 | -95.0% | -4.82 | -1.80 | -131.6% | -6.66 | -97.0% | -166.0% | -3.05 | -225.6% | -2.14 | -156.9% | 0.0% | 94% | -39.0% |
| `prem_lvl_so_k5_24h` | -1.28 | -55.0% | -2.77 | -1.28 | -95.0% | -4.11 | -80.3% | -109.9% | -2.05 | -153.9% | -1.62 | -121.4% | 0.0% | 58% | -37.5% |
| `prem_lvl_so_k5_72h` | -1.10 | -40.9% | -1.85 | -0.54 | -39.7% | -1.92 | -13.6% | -67.1% | -1.08 | -79.5% | -0.90 | -65.6% | 0.0% | 38% | -2.0% |
| `prem_lvl_ls_k3_8h` | -2.58 | -86.6% | -4.08 | -2.81 | -100.3% | -3.92 | -100.5% | -101.0% | -5.48 | -197.6% | -3.19 | -113.7% | 0.0% | 95% | -6.7% |
| `prem_lvl_ls_k3_24h` | -1.92 | -63.4% | -2.95 | -2.40 | -86.2% | -3.30 | -89.4% | -83.5% | -3.91 | -141.5% | -2.77 | -99.4% | 0.0% | 57% | -29.6% |
| `prem_lvl_ls_k3_72h` | -0.85 | -28.7% | -1.31 | -0.33 | -11.4% | -0.42 | -56.5% | +29.2% | -1.31 | -45.3% | -0.71 | -24.4% | 0.0% | 33% | +21.9% |
| `prem_lvl_ls_k5_8h` | -3.34 | -82.2% | -5.39 | -4.94 | -125.0% | -6.74 | -109.4% | -141.0% | -8.46 | -217.2% | -5.46 | -138.2% | 0.0% | 91% | -32.8% |
| `prem_lvl_ls_k5_24h` | -2.32 | -56.3% | -3.70 | -3.10 | -79.1% | -4.18 | -90.1% | -68.5% | -5.19 | -135.0% | -3.61 | -92.2% | 0.0% | 55% | -24.5% |
| `prem_lvl_ls_k5_72h` | -1.16 | -28.3% | -1.77 | -0.95 | -23.3% | -1.26 | -43.7% | -5.3% | -2.28 | -56.5% | -1.48 | -36.2% | 0.0% | 34% | +10.8% |
| `prem_chg_so_k3_8h` | -2.36 | -161.2% | -6.00 | -2.52 | -222.6% | -6.64 | -211.7% | -231.6% | -4.74 | -420.7% | -2.81 | -247.9% | 0.0% | 200% | -24.4% |
| `prem_chg_so_k3_24h` | -1.60 | -89.2% | -3.25 | -1.21 | -119.6% | -3.12 | -101.2% | -134.5% | -2.37 | -231.4% | -1.48 | -145.1% | 0.0% | 112% | -8.2% |
| `prem_chg_so_k3_72h` | -1.29 | -59.9% | -2.07 | -0.80 | -66.8% | -1.49 | +12.6% | -139.1% | -1.33 | -116.0% | -1.08 | -92.5% | 0.0% | 49% | -17.9% |
| `prem_chg_so_k5_8h` | -2.46 | -156.1% | -7.99 | -2.59 | -209.6% | -8.76 | -189.2% | -228.2% | -4.84 | -392.3% | -2.91 | -235.0% | 0.0% | 184% | -27.4% |
| `prem_chg_so_k5_24h` | -1.74 | -93.3% | -4.47 | -1.17 | -100.4% | -4.13 | -88.4% | -111.7% | -2.45 | -208.4% | -1.48 | -125.9% | 0.0% | 108% | +7.2% |
| `prem_chg_so_k5_72h` | -1.16 | -43.6% | -1.96 | -0.73 | -59.0% | -2.09 | -10.4% | -104.9% | -1.30 | -105.7% | -1.03 | -83.8% | 0.0% | 48% | -11.7% |
| `prem_chg_ls_k3_8h` | -5.63 | -187.7% | -9.51 | -5.95 | -223.6% | -8.60 | -216.1% | -230.7% | -10.83 | -422.0% | -6.28 | -236.2% | 0.0% | 200% | -25.2% |
| `prem_chg_ls_k3_24h` | -2.78 | -91.2% | -4.34 | -3.20 | -122.6% | -4.82 | -128.7% | -115.9% | -5.92 | -233.3% | -3.54 | -135.3% | 0.0% | 111% | -12.4% |
| `prem_chg_ls_k3_72h` | -1.09 | -35.1% | -1.68 | -1.37 | -52.1% | -1.59 | -52.1% | -52.6% | -2.49 | -96.3% | -1.70 | -64.9% | 0.0% | 44% | -8.1% |
| `prem_chg_ls_k5_8h` | -7.00 | -174.4% | -12.19 | -7.42 | -194.8% | -10.37 | -180.9% | -207.4% | -13.53 | -376.2% | -7.91 | -207.5% | 0.0% | 182% | -13.4% |
| `prem_chg_ls_k5_24h` | -3.78 | -92.7% | -5.73 | -3.89 | -103.2% | -5.82 | -104.0% | -102.1% | -7.45 | -209.0% | -4.38 | -115.9% | 0.0% | 106% | +2.2% |
| `prem_chg_ls_k5_72h` | -1.08 | -26.4% | -1.70 | -1.89 | -50.1% | -2.27 | -50.9% | -50.5% | -3.41 | -92.8% | -2.38 | -62.9% | 0.0% | 43% | -7.6% |

## All 48 configurations: rule by rule

✓ = pass, ✗ = fail, – = not run. The drop-one check runs only for a
configuration that passes everything else.

| config | `is_sharpe` | `is_alpha_t` | `oos_sharpe` | `oos_alpha_pos` | `oos_sharpe_2x` | `neighbour` | `oos_alpha_min` | `oos_h1_alpha` | `oos_h2_alpha` | `borrow_sharpe` | `borrow_alpha` | `coverage` | `drop_one` | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fund_lvl_so_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_so_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_so_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_so_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_so_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_so_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_ls_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_ls_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_ls_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_ls_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_ls_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_lvl_ls_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_so_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_so_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_so_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_so_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_so_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_so_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_ls_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_ls_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_ls_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_ls_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_ls_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `fund_chg_ls_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_so_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_so_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_so_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_so_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_so_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_so_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_ls_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_ls_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_ls_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_ls_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_ls_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_lvl_ls_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_so_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_so_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_so_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_so_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_so_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_so_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_ls_k3_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_ls_k3_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_ls_k3_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_ls_k5_8h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_ls_k5_24h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |
| `prem_chg_ls_k5_72h` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | – | rejected |

## References (never candidates)

| reference | IS Sharpe | IS CAGR | OOS Sharpe | OOS CAGR | OOS alpha/yr | OOS beta |
|---|---|---|---|---|---|---|
| `bench_ew` | +0.75 | +31.8% | -0.04 | -24.5% | -0.0% | +1.00 |
| `bench_btc` | +1.71 | +104.6% | +0.27 | +2.5% | +12.1% | +0.47 |
| `ref_short_all_8h` | -0.93 | -61.0% | -0.16 | -26.6% | -11.3% | -0.89 |
| `ref_short_all_24h` | -0.90 | -60.1% | -0.22 | -29.3% | -14.9% | -0.90 |
| `ref_short_all_72h` | -0.93 | -61.2% | -0.34 | -35.2% | -22.8% | -0.91 |

## What the numbers say (descriptive, not part of the decision)

1. **Costs dominate.**
   - Turnover is 140 to 1,080 times a year, and OOS costs run 31% to 200% a
     year.
   - The change signals at 8h are the worst. The `prem_chg` books pay about
     180–200% a year.
   - Rebalancing less often cuts costs, but no rebalancing period brings any
     configuration into positive alpha.
2. **Before costs, funding level carried a little information out of sample,
   in the expected direction. It is not an edge.**
   - The zero-cost OOS alpha of the 12 `fund_lvl` books ranges from +0.3% to
     +52% a year.
   - But costs exceeded it in every configuration.
   - Base-cost IS alpha was negative for all 12. Zero-cost IS alpha was not
     part of the pre-registered output and was not computed.
   - And the figure is being read after the fact out of a 48-configuration
     grid.
   - At most, this motivates a new, separately pre-registered hypothesis
     (below).
3. **Premium level ranks a structural feature, not crowding.** BNB is short
   17–28% of the time in every `prem_lvl` book, because its perp premium sits
   persistently above the others'.
4. **The OOS "alt bear market" was mostly volatility drag, not downward
   drift.**
   - The hourly-rebalanced equal-weight universe returned −24.5% a year
     (CAGR), but its OOS Sharpe was −0.04, so its arithmetic mean return was
     close to zero.
   - Shorting the whole universe also lost money: `ref_short_all` OOS CAGR
     −27% to −35%, alpha −11% to −23%.
   - The 8h version's −11.3% alpha is about its 12%-a-year cost (mostly
     borrow).
   - The 24h and 72h versions lose more. That reflects holding a drifting
     short basket against an hourly-rebalanced benchmark, a mechanical effect
     of the alpha definition that applies to every held book here.
   - So the pre-registration's worry that short-only books would gain on
     drift did not happen. Nothing needed the alpha floor to be rejected.
5. **Coverage was never a problem.** Every OOS decision had at least 23
   eligible coins (mean 36.8; in sample, mean 44.9 and minimum 36). No
   rebalance was skipped in any configuration.

## Multiple-testing caveat

There are 48 configurations, and they are correlated: the same signal at
neighbouring k and period values. The pre-registered concern was false
positives (section 12). With zero candidates there are none to discount.

The caveat now runs the other way: **do not mine these 48 rows for the one
that almost worked.**
- The best OOS alpha (−4.0%), the positive zero-cost alphas, and the 10
  positive H1 and 2 positive H2 alphas are the kind of scatter expected
  somewhere among 48 noisy, correlated rows with a negative mean.
- No configuration's IS alpha t-stat is even positive, let alone at the
  Bonferroni level of 3.08.

## What this means

- **For the bot:** on this universe, funding and premium give no edge in
  choosing which coins to short, once costs are paid. Do not build the 8–72h
  rebalancing executor for this.
- **Together with `research/RESULTS.md`:**
  - no price-based signal passes out of sample;
  - hedged funding carry does not pass out of sample;
  - and now no crowding signal passes out of sample either.
- **Follow-ups:** any follow-up, for example a slower, lower-turnover
  funding-level book, would be a new hypothesis that these results suggested.
  It needs its own pre-registration and must be judged on data not yet seen
  (a forward test), not on this sample.

## Deviations and things that went wrong

- **A reporting bug, fixed before the run in `8b4ec7a`.** The synthetic
  end-to-end test caught it: `crowding.verdict` listed `drop_one` as failed for
  configurations that had failed the IS gate. Drop-one never runs for those
  configurations. The verdict labels were already correct, and no real data had
  been read at that point.
- **The drop-one and schedule-phase stages never ran,** because no
  configuration got far enough. Only the synthetic tests exercise them.
- **The secrets file was loaded into memory once, during the design phase,**
  before the pre-registration commit.
  - One coverage-check command imported `config.py` without overriding
    `TRADING_ENV_FILE`. `config.py` therefore loaded
    `~/.config/quant-bot/env` into that process's environment at import.
  - Nothing was printed or written, and the process exited.
  - Every later command, including the run of record, set
    `TRADING_ENV_FILE=/nonexistent` and pointed the data and log dirs at
    scratch space.
- **The run itself:** exit code 0, no warnings, one run, no re-run. The run
  directory's `command.txt` shows the scratch paths as `<scratch>`.
- **The cache** now extends to 2026-10-13. The END cap kept the OOS window the
  same as the existing study's.

## Re-running

```
RESEARCH_OFFLINE=1 TRADING_ENV_FILE=/nonexistent TRADING_DATA_DIR=<scratch>/td \
TRADING_LOG_DIR=<scratch>/tl .venv/bin/python -m research.crowding_study --out <dir>
```

Run it from a checkout of `8b4ec7a` or later. It should print the same data
fingerprint (`4a7bd3e0…`) and the same report. The pre-registration requires an
independent re-run only for a PASS (section 14), and there was none.
