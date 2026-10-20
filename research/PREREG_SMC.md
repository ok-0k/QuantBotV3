# Pre-registration: the owner's "smart money" intraday playbook on crypto spot

**Status: design only.** Written on 2026-10-20 on branch
`research/smc-playbook` (based on live commit `f2a051a`). It is committed
**before any backtest of this playbook was run on real data** and before any
trade list, return, win rate or expectancy for it existed. The only real data
looked at so far is cache coverage of the 5m klines: first and last
timestamps, bar counts, gaps and outages (Appendix A). That is the same kind
of data-only pass as `research/data_check.py`. A Project Manager (PM) reviews
this document before anything runs.

**Revision 1 (2026-10-20), after PM review of `8e0af86` (verdict: CHANGES,
small; no lookahead found).**
- Decisions are made on the 8 configurations × **short** only; `both` is now
  report-only, like `long`.
- The OOS t-stat bar is **2.50**, binding.
- Three binding OOS checks are added: portfolio-scope expectancy > 0,
  risk-weighted expectancy > 0, and no NY day > 25% of OOS net R.
- §5.5 wording is fixed.
- Three sanity counters are added, each of which must be 0.
- The PM's rulings on the §13 judgement calls are recorded there.

Still design only: no real-data strategy statistic has been computed.

Code: `research/smc.py` implements sections 3 to 11 (data handling,
detection, bias, order model, costs, limits, statistics, decision rules).
`tests/test_research_smc.py` tests it on synthetic data only: swing timing,
structure, displacement, FVG, sweeps, ticks, the fill model, costs, limits,
verdicts, and an exact long/short mirror test. No-lookahead is tested
several ways:
- future bars replaced by different data cannot change any earlier setup;
- at several cut times (09:30, 15m boundaries, 10:55), with the SMT
  reference symbol's future replaced too, neither the setups nor the orders
  `walk` places before the cut change, for every config, side, scope and
  cost case;
- SMT uses reference information only up to the reclaim;
- the walk is causal: perturbing any outcome not yet known at time T cannot
  change an order placed before T.
`research/smc_data_check.py` downloads the 5m klines and reports coverage.
The runner (`research/smc_study.py`) is written only after approval, and it
must use the functions and constants in `research/smc.py` unchanged.

## 1. Question

Does the owner's intraday "smart money" (SMC/ICT) playbook have an edge on
the bot's crypto spot universe after honest costs, out of sample?

The playbook was written for index futures and forex. This study adapts it to
Binance spot, keeping its rules, numbers and scoring. The large risk is cost:
on live candles the median 5m ATR(14) is about 0.28% of price (BTC about
0.11%) and the median 15m ATR about 0.53%. With a stop of about 2 ATR, a
0.36% round trip costs about 0.64R on 5m and 0.34R on 15m. Most of the
playbook's gross edge, if it has one, has to survive that.

Earlier pre-registered studies on this universe found nothing that passes out
of sample: 17 price strategies (`46529d2`), hedged funding carry (corrected
2026-10-11) and 48 funding/premium crowding configurations (`98e8cec`,
results `1e9f15e`). The bot is short-only and currently flat behind an
evidence gate that admits only strategies passing a pre-registered OOS test.

**Hypothesis.** Sweep of a marked liquidity level, then a displacement market
structure shift, then a limit entry at the consequent encroachment (CE) of the
fair value gap (FVG) the displacement left, in the NY morning killzone,
earns a positive expectancy in R net of costs. Short setups alone must be
decidable, because the bot can only short.

## 2. Universe, periods, data

| item | value |
|---|---|
| Universe | `config.SYMBOLS`: the live bot's 50 symbols, including the 4 delisted pairs (MATIC, MKR, OCEAN, AGIX) up to their delisting |
| Data | Binance **spot** 5m klines (`api.binance.com/api/v3/klines`, public) via `research.data.load_klines(symbol, "5m", start, end)`, cached as `/home/admin/trading_data/research/klines_5m/<symbol>.npz` |
| Warm-up | `DATA_START` = 2022-10-01 00:00 UTC (94 days before the first trading day) |
| IS (design) | NY trading days 2023-01-01 → 2025-04-01 (first trading day 2023-01-03, last 2025-03-31) |
| OOS (held out) | NY trading days 2025-04-01 → 2026-10-07 (last trading day 2026-10-06) |
| OOS halves | H1 = [2025-04-01, 2026-01-01), H2 = [2026-01-01, 2026-10-07) |
| Trading days (§3.2) | IS 562, OOS 381 (H1 190, H2 191); 943 in all |
| Hard END | **2026-10-07 00:00 UTC.** Only bars that open and close by END are loaded (`smc.load_symbol` builds the grid [DATA_START, END) and asserts no bar runs past it). The cache may run later; nothing after END is read |
| Offline | `smc.load_symbol` refuses to run unless `RESEARCH_OFFLINE=1` (no requests, no cache writes) |

A trade belongs to the NY trading day on which its order was placed. Every
trade is opened and closed between 09:30 and 12:00 NY, i.e. between 13:30 and
17:00 UTC of that same date, so the periods split cleanly by day and the last
OOS trade ends before END.

## 3. Data handling

### 3.1 Grid

- Every symbol sits on the same regular 5m grid of bar OPEN times
  [DATA_START, END) (`smc.to_grid`). A bar Binance did not return is NaN.
  Bars still forming are never cached (`load_klines` drops them).
- A bar's close time is its open plus 5 minutes. A decision "at the close of
  bar k" uses bars ≤ k only.

### 3.2 Time

- All session times are **America/New_York wall-clock, DST-aware**
  (`zoneinfo`; `smc.ny_day_minute`, `smc.ny_ms`).
- **NY day** = 00:00 to 24:00 NY (the "midnight open"). On the two DST-change
  Sundays it has 23 or 25 hours.
- **NY week** = Monday 00:00 to the next Monday 00:00 NY, weekend included
  (crypto trades 24/7).
- **Trading days** = Monday to Friday, excluding NYSE full-day holidays
  (`smc.NYSE_HOLIDAYS`, listed in the code; 2025-01-09, the national day of
  mourning, is included). The killzone is defined by the NY equity open, which
  does not happen on weekends or NYSE holidays. That gives 943 trading days,
  2023-01-03 to 2026-10-06.
- None of the session boundaries used (00:00, 02:00, 05:00, 09:30, 11:00,
  12:00, 18:00, 20:00) is a DST-ambiguous wall-clock time.

### 3.3 Higher timeframes

All built from **closed 5m bars only** (`smc.resample`, `smc.htf_bars`):

| bar | definition | complete if |
|---|---|---|
| 15m (entry TF) | 3 consecutive 5m bars from a quarter hour (UTC and NY quarter hours coincide) | all 3 present |
| 1h | clock hour | ≥ 90% of its 5m bars (11 of 12) |
| 4h | NY 00–04, 04–08, 08–12, 12–16, 16–20, 20–24 wall-clock | ≥ 90% of its 5m bars |
| 1d | NY day | ≥ 90% of its 5m bars |
| 1w | NY week | ≥ 90% of its 5m bars |
| Asia, London, overnight sessions | §4.7 | ≥ 90% of their 5m bars |

- An incomplete bar is NaN. It can be neither a swing nor a neighbour of one,
  and it cannot break structure.
- On a DST-change Sunday, the 4h block containing 02:00 has 3 or 5 hours and
  the day has 23 or 25. "Expected bars" counts real elapsed time.
- 90% matches the convention of `research.data.resample` in the earlier
  studies. The first and last block of the grid are always incomplete.
- A higher-timeframe bar is used only after its close time.

### 3.4 Tick size

No exchange metadata is used (there is no offline copy). **Rule
(`smc.infer_ticks`):** for each symbol and each UTC calendar month, the tick is
the largest power of ten such that every open, high, low and close printed
that month is a whole multiple of it (tolerance 1e-6 ticks).

- **Why this works.** Binance spot tick sizes are powers of ten, and every
  printed trade price is a multiple of the tick in force. A month has about
  35,000 prices, so all of them landing on a coarser grid by chance is
  practically impossible. The inferred tick can only be equal to or coarser
  than the true one, which makes "≥ 1 tick" rules slightly stricter, never
  looser.
- **Per month** because Binance has changed tick sizes on some pairs. In a
  month that contains a change, the finer of the two ticks is inferred.
- **Lookahead.** The tick is an exchange constant, published before trading,
  not price information. It is the only quantity computed from a whole month.
- **Rounding.** Computed prices are rounded onto the tick grid: the long entry
  and long stop are rounded down; TP1 is rounded up (shorts mirror). The
  rounding never makes a fill easier.

### 3.5 Spot tradability

A position can only fill, and only exit, on a 5m bar of that pair that exists.
So the purpose of `research.study.spot_tradable` (never hold a leg on a pair
that is not trading: the MATIC lesson) is met natively. Delisted pairs simply
stop producing bars, and no symbol-day is eligible without 33.5 hours of
complete bars before it (§3.6).

`spot_tradable` itself (8h windows on 1h closes) is a **sanity counter**
(§10, `smc.spot_mismatch_days`), applied to 1h closes resampled from the 5m
cache. It counts the symbol-days with a placed order whose window containing
09:30 NY is False. **It must be 0.** It is not a selection rule, because its
8h window can extend up to 8 hours past the decision time.

Missing bars during a trade: a pending order cannot fill on a missing bar. An
open position is exited by the first rule that fires on the next bar that
exists (a stop that gapped fills at that bar's open). If the data ends (a
delisting), the position exits at the last close as a market fill.

### 3.6 Symbol-day eligibility (coverage rules)

A symbol can trade on NY trading day D only if all of the following hold
(`smc._day_ctx`; the report counts each failure reason):

1. Every 5m bar from D−1 00:00 NY to D 09:30 NY exists (33.5 hours). This is
   backward-looking only: no data after 09:30 is required.
2. All Tier 1/2 levels exist (§4.7): the previous day, the previous week (at
   least 90% of its bars), Asia, London and overnight.
3. Listed for at least 30 days: the first bar in the cache is at least 30 days
   before D 00:00.
4. Liquid: the mean daily quote volume over the 7 NY days before D is at least
   $2M. This is the same filter as the earlier studies.

## 4. Market structure (entry timeframe, unless stated)

Every rule is written for the long side. The short side runs the same code on
mirrored prices (high ↔ low, every price negated), so it is the exact mirror
image (tested in `test_long_and_short_detection_are_exact_mirrors`).

### 4.1 ATR

- **ATR14** is Wilder's ATR on the bars of the timeframe in question: the seed
  is the mean of the first 14 true ranges, then
  `ATR += (TR − ATR) / 14` (`smc.atr`). Missing bars are skipped.
- **A candle is measured against the ATR through the previous candle**
  (`atr_prev[k]`), so a big candle never inflates its own yardstick. This
  applies to displacement bodies and FVG heights.
- Order parameters use the ATR through the placement candle (`atr[p]`).
  Equal-level tolerance uses the ATR at the second swing's confirmation.

### 4.2 Swings and structure

- **Swing high** at candle i: high[i] is strictly greater than the highs of
  the N candles before and the N after, and all of them exist. N = 2 on the
  entry TF and N = 3 on 1h, 4h and daily. Swing lows mirror (`smc.swings`).
- **Confirmation.** A swing at i is confirmed only at the **close of candle
  i+N** and is usable from candle i+N+1.
- **Structure (`smc.structure`).**
  - The *reference high* is the most recently confirmed swing high, until a
    candle **closes** above it. That close is a bullish **break**; a wick above
    is not. After a break there is no reference high until a newer swing high
    is confirmed. Lows mirror.
  - A newly confirmed swing replaces the reference, even if the older one was
    never broken.
  - The *state* is the direction of the last break (+1 up, −1 down, 0 none at
    the start of the data).
  - A break against the prior state is an **MSS** (change of character). A
    break in the same direction is a BOS.
  - If one close breaks both references (possible only when the newest swing
    high sits below the newest swing low), both are consumed and the state is
    unchanged.

### 4.3 Displacement

A bullish **displacement leg** ending at candle m (`smc.disp_leg`) is the
longest of:

- **(a)** 3 consecutive candles m−2..m, or
- **(b)** 2 consecutive candles m−1..m,

in which **every** candle is bullish, has body ≥ **1.0 × ATR**, and has an
upper (opposing) wick ≤ **0.25 × its body**; otherwise

- **(c)** the single candle m if its body ≥ **1.5 × ATR**.

The leg must start at or after the sweep candle.

**"The leg leaves an FVG"** is required too: at least one FVG (§4.4) whose
middle candle is in the leg and whose third candle is ≤ m+1.

A bearish **displacement candle**, used only by the breakout test (§4.5), is
the body part of the same rules: a bearish body ≥ 1.5 ATR, or membership in a
2–3 candle bearish run, each body ≥ 1.0 ATR with lower wick ≤ 0.25 × body
(`smc.bear_disp_member`). No FVG is required there.

### 4.4 Fair value gap

- **Bullish FVG** (candles c1, c2, c3 consecutive): high[c1] < low[c3]. The
  zone is [high[c1], low[c3]], with **bottom** = high[c1] and **top** =
  low[c3] (`smc.leg_fvgs`).
- Minimum height **0.25 × ATR** through c1.
- **CE** = (bottom + top) / 2.
- An FVG is known at the close of c3.
- **Invalid** once an entry-TF candle closes below its bottom (a body close
  fully through it, the opposite way).
- Bearish FVGs mirror: low[c1] > high[c3].

### 4.5 Sweep or breakout

A sell-side level L (§4.7) is **taken** by the first entry-TF candle j, at or
after the level's availability, whose low ≤ L − 1 tick. A level can be taken
only once, so whatever happens at j decides it for the rest of its life
(`smc.first_through_low`, `smc.classify_sweep`):

- **Breakout (no trade):** candles j and j+1 both close below L, and one of
  them is part of a bearish displacement (§4.3).
- **Sweep:** otherwise, the first of j, j+1, j+2 that closes above L is the
  reclaim candle r. The sweep is confirmed at r's close.
- **Neither (no trade):** no close above L by j+2. A close exactly at L is not
  above it.

### 4.6 MSS after a sweep

For a bullish sweep with sweep candle j and reclaim r (`smc.mss_after_sweep`):

- **The FIRST bullish break** (§4.2) among candles j..j+12 decides. It is the
  MSS only if all of these hold:
  - it flips a **down** state;
  - it comes at or after r;
  - it ends a displacement leg starting at or after j;
  - that leg leaves an FVG (§4.3).
- If the first break fails any of these, or no break comes within 12 candles,
  the sweep produces nothing.
- A break that was itself a BOS (the state was already up) is not an MSS, so
  it voids the sweep too.

### 4.7 Liquidity levels

Levels are computed per NY trading day D from 5m bars (`smc.day_levels`).
Each has an **availability time**, the moment it is fully formed. It can only
be swept, or count as untaken, after that time:

| tier | level | window | available from |
|---|---|---|---|
| 1 | PDH / PDL | previous NY calendar day (Sunday for Monday) | D 00:00 |
| 1 | PWH / PWL | previous NY week, Monday–Sunday | Monday 00:00 of D's week |
| 2 | Asia high / low | D−1 20:00 → D 00:00 | D 00:00 |
| 2 | London high / low | D 02:00 → 05:00 | D 05:00 |
| 2 | Overnight high / low | D−1 18:00 → D 09:30 | D 09:30 |
| 3 | Equal highs / lows | two entry-TF swing lows i1 < i2 whose candles open within 24h, with \|low1 − low2\| ≤ 0.1 × ATR, and no candle strictly between them trading ≥ 1 tick below the lower one. Level = the lower low | close of i2's confirmation candle, for 24h |

- The 18:00 overnight start mirrors the futures overnight (Globex) session.
  Old session levels are not carried over to later days.
- Tier 3 levels come from the entry timeframe (`smc.eq_levels`).
- Sell-side levels are the lows; buy-side levels are the highs.

### 4.8 Setup assembly (long; short mirrors)

Steps 1 to 10 below are `smc.find_candidates`. All MSS events with the same
MSS candle m are merged into one setup, using the earliest of their sweep
candles, j0.

1. **Placement candle** p = m+1. The order is placed at **p's close**, when
   every FVG the leg can leave is known.
   - **Killzone gate:** p's close must be in [09:30, 11:00) NY on D.
2. **FVG choice.** Among the leg's FVGs (third candle ≤ p), drop any whose CE
   was touched (low ≤ CE), or whose bottom was closed through, by a candle
   after its third candle and up to p. Choose the remaining FVG **closest to
   the broken swing price** (distance 0 if the swing lies inside the zone).
   Ties go to the later FVG.
3. **Entry:** a buy limit at CE, rounded down to the tick.
4. **Sweep extreme** = lowest low from j0 through p.
5. **Stop** = sweep extreme − max(2 ticks, 0.1 × ATR[p]), rounded down.
   **R** = entry − stop. It must be > 0.
6. **TP1** = entry + **1.0R**, rounded up.
7. **TP2 (draw on liquidity)** = the nearest **untaken Tier 1/2 buy-side
   level** above the entry. Candidates are PDH, PWH, Asia, London and
   overnight highs that are available by placement and not traded ≥ 1 tick
   through from their availability up to p.
   - The setup is **skipped unless TP2 − entry ≥ 2R**. Only the nearest level
     is checked; there is no searching further out.
8. **Score (§4.9)** and grade.
9. **Required items.** The sweep, the displacement MSS, the killzone and
   DOL ≥ 2R are required by construction: a setup without them never exists.
10. All of this is independent of the bias gate, the management mode and costs.

### 4.9 Scoring

| item | points | definition |
|---|---|---|
| Sweep | 2 (required) | §4.5 |
| MSS + displacement | 2 (required) | §4.6 |
| Killzone | 1 (required) | placement in [09:30, 11:00) |
| DOL ≥ 2R | 1 (required) | §4.8 step 7 |
| HTF bias aligned | 2 | the **daily** structure state (§5.1) at D−1's close equals the trade direction |
| Discount / premium | 1 | long: entry < daily dealing-range midpoint (§5.2); short: entry > it |
| SMT | 1 | §4.10 |

- **9–10 = A+**, **7–8 = B**, **6 = no trade**.
- "HTF bias aligned" uses the daily state, not the final bias. That way a
  Neutral day (§5.4) can still produce an A+ setup, which the playbook's
  "Neutral → A+ only" rule needs. If it used the final bias, a Neutral day
  could never score above 8.

### 4.10 SMT divergence

- The reference is BTC for every alt (ETH included), and ETH for BTC.
- A long setup earns SMT if, for at least one Tier 1/2 level type it swept
  (PD, PW, Asia, London, overnight), the reference symbol had **not** traded
  ≥ 1 tick through its own level of the same type by the close of this
  setup's reclaim candle (`smc.ref_taken_times`, measured on the reference's
  5m bars).
- In plain terms: this coin took the liquidity, but the market leader did not.
- Tier 3 sweeps and missing reference data give no SMT point. Shorts mirror.

## 5. Bias (exact algorithm; `smc._day_ctx`, `smc.bias_rule`)

Fixed at **09:30 NY** each trading day, except for §5.5.

### 5.1 Structure votes

- The structure state (§4.2, N = 3, built from closed bars only) on:
  - **daily**, as of D 00:00 (D−1's close);
  - **4h**, as of 09:30 (the 04–08 bar is the last one closed);
  - **1h**, as of 09:30 (the 08–09 bar is the last one closed).
- Let *up* and *dn* count the +1 and −1 votes among the three.
  - **conflict** if up ≥ 1 and dn ≥ 1.
  - **S = up** if up ≥ 2 and dn = 0, and **S = down** if dn ≥ 2 and up = 0.
  - **S = none** otherwise.

### 5.2 Premium / discount

- The **daily dealing range** runs from the most recent confirmed daily swing
  low to the most recent confirmed daily swing high, both as of D 00:00. Its
  midpoint is EQ.
- **p0** = close of the 09:25–09:30 5m bar.
- p0 < EQ is **discount**; p0 > EQ is **premium**.

### 5.3 Overnight liquidity

**Liq = bull** when both of these hold:

- at least one sell-side level among PDL, PWL and the Asia low was traded
  ≥ 1 tick through between max(its availability, D−1 18:00) and 09:30, and p0
  is back above it;
- none of PDH, PWH or the Asia high was traded through in its window.

**Liq = bear** is the mirror. Otherwise **Liq = none** (this includes both
sides taken).

### 5.4 Bias table

| S | Liq | bias |
|---|---|---|
| up | bull | **Bullish** |
| up | none | **Bullish** if discount, else Neutral |
| up | bear | Neutral |
| down | bear | **Bearish** |
| down | none | **Bearish** if premium, else Neutral |
| down | bull | Neutral |
| none | bull | **Bullish** if discount, else Neutral |
| none | bear | **Bearish** if premium, else Neutral |
| none | none | Neutral |
| conflict | any | **Neutral** |

In words, three routes lead to a Bullish bias:
- the timeframes agree up and the overnight session swept sell-side;
- the timeframes agree up and price is in discount, with no opposing sweep;
- the timeframes have no direction, but sell-side was swept and price is in
  discount.

Conflicting timeframes always give Neutral.

### 5.5 Intraday neutralisation

- **The trigger:** the bias is Bullish, and a **15m bearish displacement
  MSS** happens whose MSS candle opens in [09:30, 11:00).
  - This means a bearish break that flips an up state, ends a bearish
    displacement leg, and leaves an FVG.
  - The 15m bars, N = 2 and §4.2–4.4 apply, whatever the entry timeframe.
- **The effect:** the day is **Neutral from the close of candle m+1**, the
  candle after the 15m MSS candle m, onward (`smc.neutral_time`). Bearish
  mirrors.
  - m+1 is the latest candle that can be the FVG's third candle.
- It applies to orders placed at or after that time. Pending orders and open
  positions are not touched.

## 6. Order and fill model (`smc.simulate`)

**Execution is simulated on 5m bars for both entry timeframes.** 5m is the
finest data available, and it makes same-bar ambiguities rarer.

The order exists from the placement time T_p and can only fill on 5m bars
**opening at or after T_p**. It is live on bars opening before
min(T_p + 60 min, 11:00).

### 6.1 Pending order

Checks within each bar, in this order:

1. **TP1 touched** (high ≥ TP1) → cancel ("TP1 reached first"). This is
   checked before the fill, so a bar that both touches TP1 and trades through
   the limit counts as a cancel: no winner is credited from an unknowable
   order of events.
2. **Fill:** low ≤ limit − 1 tick, traded THROUGH by one tick. The fill is at
   the limit price, never better, even if the bar opened through it. There is
   no slippage.
3. **FVG invalidation:** the entry-TF candle ending with this bar closes below
   the FVG bottom → cancel.
   - With the limit at the CE, price must trade through the CE before it can
     close below the bottom, so check 2 always fires first. This cancel
     therefore cannot bind. It is kept because the playbook lists it, and its
     count is a **sanity counter (§10) that must be 0**.
4. If the live window ends unfilled → cancel ("expired" after 60 minutes, or
   "killzone_end").

### 6.2 Open position

- **The fill bar** can only stop out (its high cannot reach TP1, or check 1
  would have cancelled the order). If its low ≤ stop, the fill is followed by
  the stop in the same bar.
- **On each later bar, in order:**
  1. **Session exit:** the first bar at or after 12:00 → exit everything at
     its open, a market fill.
  2. **Time stop:** the position is exited at the open of the first bar at or
     after (fill bar close + 45 min), a market fill, unless a bar after the
     fill bar reached high ≥ entry + 0.5R before then.
  3. **Stop:** low ≤ stop → fill at the stop, or at the bar's open if it
     opened beyond the stop (gap), a market fill. **The stop is checked
     before any target**, so a bar touching both counts as the stop.
  4. **TP1** (management `pbe` only): high ≥ TP1 + 1 tick → 50% filled at
     TP1, a limit fill. The stop moves to **breakeven** (the entry price). If
     the same bar's low ≤ entry, the breakeven stop is assumed hit after TP1
     in that bar.
  5. **TP2:** high ≥ TP2 + 1 tick → the remainder filled at TP2, a limit fill.
  6. The +0.5R check for the time stop uses this bar's high.

### 6.3 Management (grid dimension)

- **`pbe`:** 50% at TP1, then the stop goes to breakeven, and the rest runs
  to TP2.
- **`tp2`:** full size to TP2, with the stop unchanged.
- Both use the time stop and the 12:00 exit.

## 7. Costs (`smc.net_r`, `smc.COSTS`)

| case | fee per fill | slippage (market-type fills only) | borrow on shorts |
|---|---|---|---|
| base | 0.10% | +0.08% | 0.03% / day |
| 2× stress | 0.20% | +0.16% | 0.06% / day |
| zero (diagnostic only) | 0 | 0 | 0 |

- **Fees** are paid on every fill: the entry, each partial and the final
  exit, on that fill's notional.
- **Market-type fills** are stops (including breakeven), time stops, session
  exits and data-end exits. They fill 0.08% worse. Limit fills (entry, TP1,
  TP2) have no slippage.
- **Borrow** is charged on short notional at the entry price, per started
  hour in the position (Binance margin charges hourly).
- **Net R** = net P&L ÷ (|entry − stop| × size). Costs therefore weigh more
  in R when the stop is tight relative to price, which is the playbook's main
  risk on crypto.
- **Futures-style per-contract commissions are NOT used.** Every cost is a
  percentage of notional, the same model as the bot's paper trading and the
  earlier studies.

## 8. Gates, sizing and risk limits

### 8.1 Gate and risk weight (`smc.gate`)

- The risk unit is **1U = 0.5% of equity**. A+ = 1U and B = 0.5U. The "A+ may
  use 1%" option is not taken: 0.5% is fixed.
- **Bias as written:**
  - Bullish → longs only (A+ 1U, B 0.5U).
  - Bearish → shorts only.
  - **Neutral → A+ only, at 0.5U**, either direction.
- **Bias off:** any direction, every day (A+ 1U, B 0.5U). The score, and with
  it the grade and the 7-point minimum, is unchanged. Only the gate and §5.5
  are switched off.

### 8.2 Limits (`smc.walk`)

**Decision: the limits apply PER SYMBOL** (scope `symbol`). Every symbol is
its own copy of the playbook account. The limits are:

- at most **2** filled-or-pending orders per NY day;
- no new order once the day's realised result is ≤ **−2U**, or its last two
  closed trades both lost;
- no new order once the NY week's realised result is ≤ **−5U** (this resumes
  the next Monday);
- **one order or position per symbol at a time**. In the "both" variant a
  long and a short also exclude each other.

A trade with net R < 0 counts as a loss, so a breakeven exit that only pays
costs is a loss.

**Why per symbol:**
- The playbook was written for one instrument at a time, and its limits
  protect a single-instrument account.
- Applied across 50 symbols, "2 trades a day" would turn the study into "the
  first two setups of the day anywhere". That keeps whichever coins happen to
  set up first, and throws away about 95% of the evidence.
- The 50 coins are strongly correlated, so their same-day trades are not
  independent. **Inference clusters by NY day** (§10), so 30 correlated alt
  trades on one morning count as one cluster, not 30 observations.

**The portfolio variant** (scope `portfolio`) is **binding** through one
check (§11 `port_exp_pos`: its OOS expectancy at base cost must be > 0). PM
ruling on §13 #17: the bot is one account.
- The same limits apply once across all symbols: 2 filled-or-pending orders
  a day, −2U a day, −5U a week, one order per symbol.
- Orders are processed in placement order; ties go by `config.SYMBOLS` order,
  then long before short.
- This variant shows what a single account following the playbook literally
  would have done. The report also shows the worst day in U and the most
  trades in a day for the per-symbol run, so its portfolio risk is visible.
- In addition, §11 `day_share` (no NY day > 25% of OOS net R) catches a
  result carried by one big correlated morning.

**Note:** with the 2-trade cap, "2 consecutive losses" can never bind (the
cap binds first). The −2U daily stop binds only after a single loss worse
than −2R, such as a gap or 2× costs. Both rules are implemented as written
anyway. **The number of times the two-loss rule binds is a sanity counter
(§10) that must be 0.**

### 8.3 News

**No news blackout is applied.** No calendar data is available offline. The
reasons:

1. The tier-1 US releases for crypto (CPI, NFP, PPI, retail sales, GDP,
   jobless claims) print at **08:30 ET**. With the usual ±30-minute blackout
   that ends by 09:00, before the 09:30 killzone. FOMC statements and minutes
   (14:00 ET) come after the 12:00 flat time.
2. The 10:00 ET releases (ISM, JOLTS, UMich, Conference Board, new home sales)
   do fall inside the killzone. Their dates cannot be listed reliably from
   memory: they shifted during the October–November 2025 US government
   shutdown. A partial or guessed list would be a filter applied
   inconsistently.
3. A fixed daily no-entry window around 10:00 (for example 09:55–10:05) would
   be an executable stand-in, but it would cut every day for a release that
   occurs on only some days. **PM ruling (2026-10-20): the window stays
   OFF.**

The forward test (§14) records whether each trade overlapped a 10:00 release,
so the effect can be checked going forward.

## 9. The grid: 8 configurations, complete

2 entry timeframes × 2 management modes × 2 bias settings = **8
configurations** (`smc.grid()`), named `smc_{tf}_{mgmt}_{bias|nobias}`:

| # | config | entry TF | management | bias gate |
|---|---|---|---|---|
| 1 | `smc_5m_pbe_bias` | 5m | 50% at TP1 + breakeven | as written |
| 2 | `smc_5m_pbe_nobias` | 5m | 50% at TP1 + breakeven | off |
| 3 | `smc_5m_tp2_bias` | 5m | full to TP2 | as written |
| 4 | `smc_5m_tp2_nobias` | 5m | full to TP2 | off |
| 5 | `smc_15m_pbe_bias` | 15m | 50% at TP1 + breakeven | as written |
| 6 | `smc_15m_pbe_nobias` | 15m | 50% at TP1 + breakeven | off |
| 7 | `smc_15m_tp2_bias` | 15m | full to TP2 | as written |
| 8 | `smc_15m_tp2_nobias` | 15m | full to TP2 | off |

**Sides.** Each configuration is simulated three times, as **separate
simulations**, because the limits interact:
- `long`: long setups only;
- `short`: short setups only (what the bot could trade today);
- `both`: the full playbook.

**Fixed, not grid dimensions:** the time stop (on), the risk unit (0.5%), the
news handling (none), TP1 = 1R, the limit scope (per symbol), and every
constant in §4. **Decision units:** 8 configurations × `short` = **8**
(§11). The `long` and `both` results are reported but cannot produce a
candidate (§11).

## 10. What is reported

Every configuration, every side and every period is reported, not only the
winners. Output goes to `results/<timestamp>-smc/`: `report.txt`,
`summary.json` and `trades.csv`, holding every trade of every simulation.

**For each config × side, for IS, OOS, OOS-H1 and OOS-H2, at base cost
(`smc.summarize`):**
- trades and NY days traded;
- expectancy (mean net R per trade, unweighted) and median R;
- the **day-clustered t-stat**:
  t = mean / SE, with SE = √(G/(G−1) · Σ_g (Σ_{i∈g} (R_i − mean))²) / N,
  where g runs over NY trading days, G is the number of days and N the number
  of trades (`smc.cluster_t`);
- win rate, average win and average loss, profit factor;
- total R, total U, and the risk-weighted expectancy Σ wR / Σ w;
- **max drawdown in R** (trade units) and **in U** (as sized), from
  cumulative sums in exit-time order;
- gross (zero-cost) expectancy of the same trades, and cost per trade in R;
- the worst day in U and the most trades in one day;
- the top symbol and its share of net R; the top NY day and its share of net R;
- the number of A+ trades and the exit reasons.

**OOS at 2× costs:** n, expectancy and t. This is a separate simulation,
because the limits react to the higher losses.

**The funnel, per entry TF and side:**
- levels taken, breakouts or failed reclaims;
- MSS events, MSS events in the killzone;
- setups dropped for no valid FVG, and for DOL < 2R;
- setups by grade;
- per config: orders placed, filled, and cancelled by reason.

**Context:**
- eligible symbol-days, and the reasons for ineligibility;
- the bias distribution (Bullish, Bearish and Neutral days), and the days
  neutralised under §5.5;
- the inferred tick per symbol (median tick ÷ price).

**The portfolio variant (§8.2), for every config × side:** IS and OOS n,
expectancy, t and max drawdown in U. The OOS expectancy feeds §11
`port_exp_pos`.

**Sanity counters (each must be 0).** Over every config, side, scope and cost
case of the run:

| counter | source | why it must be 0 |
|---|---|---|
| `fvg_invalid` cancels | outcome reason of placed orders (`walk` stats `outcome_fvg_invalid`) | a CE limit always fills first (§6.1) |
| two-loss rule binds | `walk` stats `two_loss_binds` | the 2-trade cap binds first (§8.2) |
| `spot_tradable` mismatches | `smc.spot_mismatch_days` over symbol-days with a placed order | eligibility needs complete bars before 09:30 (§3.5) |

**If any counter is not 0, every verdict is frozen.** It is then handled as
a §15 bug: a demonstrating test, a fix, before-and-after numbers for all
configurations, and a post-hoc label. No verdict stands until that is done.

**For the whole run:** the data fingerprint (`smc.fingerprint`), the commit
hash, END and the runtime.

## 11. Decision rules (fixed now)

**Scope.** The rules are applied to each of the **8 decision units**: the
8 configurations × side **short**.
- `long` and `both` are reported with the same checks for information. They
  cannot become candidates: the bot is short-only and cannot trade them, and
  a long-capable executor would need its own pre-registration.

All statistics are in **R per trade net of base costs**, unweighted, unless a
rule says otherwise (`smc.verdict`, thresholds in `smc.PASS`).

A decision unit is a **CANDIDATE** only if **every** check holds:

| check | rule |
|---|---|
| `is_exp_pos` | IS expectancy > 0 |
| `oos_exp_min` | OOS expectancy ≥ **+0.10R** |
| `oos_t` | OOS day-clustered t-stat ≥ **2.50** |
| `oos_n` | at least **100** OOS trades |
| `oos_h1_pos`, `oos_h2_pos` | expectancy > 0 in **both** OOS halves |
| `oos_2x_nonneg` | OOS expectancy ≥ 0 at **2× costs** (its own simulation) |
| `sym_share` | no single symbol contributes more than **25%** of OOS net R (total OOS net R must be > 0) |
| `day_share` | no single NY trading day contributes more than **25%** of OOS net R (total OOS net R must be > 0) |
| `oos_exp_w_pos` | OOS **risk-weighted** expectancy Σ wR / Σ w > 0, where A+ is sized at 2× B |
| `port_exp_pos` | OOS expectancy > 0 with **portfolio-scope** limits (§8.2), at base cost |

**Max drawdown in R** (and in U) is reported for every unit. It is not a
threshold.

**Verdicts:**
- `CANDIDATE (needs independent re-run + forward test)`;
- `rejected`, with the list of failed checks.

**Why these thresholds:**
- **0.10R** is a minimum economic edge after costs. It leaves room for
  frictions the backtest does not model (worse fills on small caps, borrow
  that is unavailable, queue position).
- **t ≥ 2.50 clustered by day.** Same-morning trades across 50 coins are
  correlated, and 2.50 is the Bonferroni bar for 8 units at one-sided 5%
  (§12). This universe has already had 65+ failed tests.
- **The 2× cost check is the real bar.** At 2× costs the expectancy is
  roughly the base expectancy minus one more cost per trade, so it needs
  base expectancy ≥ cost per trade. That is about 0.25R on 15m and about
  0.45R on 5m (PM estimate from the ATR figures in §1), well above the 0.10R
  floor. The 0.10R floor is confirmed as written.
- **Both halves** guard against a result carried by one regime.
- **2× costs** and the **25% concentration caps** guard against a result
  carried by optimistic costs, by one coin (the MATIC lesson) or by one
  morning.
- **Risk-weighted** and **portfolio** expectancy guard against an edge that
  disappears when trades are sized and limited the way one account actually
  would.

## 12. Multiple testing

There are 8 decision units (configurations × short), strongly correlated:
the same setups appear across management and bias settings.

- **The bar is t ≥ 2.50**, the Bonferroni correction for 8 units at one-sided
  5% (0.05 / 8 = 0.00625, which is z ≈ 2.50). It is binding.
- Under no edge, the chance that any one of 8 independent units clears it by
  luck is at most about 5%. The correlation between units makes it smaller.
- **The defences on top:** the stacked rules (minimum 0.10R, a positive IS,
  both OOS halves positive, 2× costs, at least 100 trades, symbol and day
  concentration caps, risk-weighted and portfolio expectancy), then the
  independent re-run and the forward test (§14).
- Even a CANDIDATE is weak evidence until it survives a forward paper test.

## 13. Judgement calls and unverified assumptions

**Judgement calls**, made here before any result. After the run none may
change.

**PM rulings (2026-10-20):**
- #1–3, #8, #9, #13, #15, #16, #20 and #21 are confirmed.
- #10: TP1 = entry + 1R is confirmed.
- #17: the portfolio variant becomes binding through `port_exp_pos`.
- #18: the time stop is always on.
- #19: the 09:55–10:05 window stays OFF.
- #24: the decision covers short only.
- The two rules that can never fire stay in, with their counts reported
  (§10).
- All other items stand as written.

1. **Day and week boundaries.** The NY day runs midnight to midnight (not a
   17:00 or 18:00 roll). The week runs Monday to Sunday with the weekend
   included. The previous day for Monday is Sunday.
2. **Trading days** are NYSE trading days. Weekends and the listed full-day
   holidays are excluded.
3. **Overnight** = 18:00 (D−1) → 09:30. Session levels are not carried over
   between days.
4. **A level is taken once.** Its first trade-through by ≥ 1 tick decides
   sweep, breakout or neither, and it can never be swept again.
5. **Tier 3 equal lows:** entry-TF swings within 24h of each other, marked
   for 24h, not traded through between them. They are sweep targets only,
   never a DOL or an SMT input.
6. **Displacement:** "2–3 bodies ≥ 1.0 ATR" is read as *each* body. The
   opposing wick must be ≤ 0.25 × body. Every candle is measured against the
   ATR through the previous candle.
7. **The breakout test** uses bearish displacement candles (body rules
   only, no FVG requirement).
8. **MSS:** the first break after the sweep decides. It must flip a down state
   (a BOS voids the sweep).
9. **The order is placed at the close of m+1,** when every FVG the leg can
   leave is known. An FVG whose CE was touched before placement is skipped.
   Ties in "closest to the broken swing" go to the later FVG.
10. **TP1 = entry + 1.0R** exactly, not an "internal high". The alternative,
    TP1 = the displacement leg's high, makes TP1 and the TP1-cancel rule
    depend on leg shape. *(PM: 1R confirmed.)*
11. **TP2 = the nearest untaken Tier 1/2 pool.** If it is < 2R away the setup
    is skipped, with no looking further out.
12. **The sweep extreme** is the lowest low from the earliest merged sweep
    candle through placement.
13. **The bias algorithm (§5),** including the liquidity route and P/D. **HTF
    aligned** means the daily state, so a Neutral day can produce an A+.
14. **The 15m neutralisation** affects new orders only.
15. **SMT** is measured against BTC (ETH for BTC) on the same level type,
    Tier 1/2 only.
16. **Risk:** 0.5% fixed for A+, 0.25% for B and for A+ on a Neutral day.
17. **The limits are per symbol** for the main statistics. *(PM: the
    portfolio variant is binding through `port_exp_pos` (§8.2, §11).)*
18. **The time stop is on** in every configuration. The playbook says
    "consider". It is not a grid dimension, to keep the grid at 8. *(PM:
    confirmed.)*
19. **No news blackout** (§8.3). *(PM: the 09:55–10:05 window stays OFF.
    The forward test logs 10:00 releases.)*
20. **Execution on 5m bars** for both entry timeframes.
21. **Same-bar ambiguities** are resolved against the strategy:
    - a TP1 touch before a fill means a cancel;
    - the stop comes before targets;
    - a breakeven touch after TP1 in the same bar is a hit;
    - a limit fills at the limit price, never better.
22. **A loss is net R < 0,** so a breakeven exit after costs counts as a
    loss.
23. **Expectancy is unweighted per trade.** The risk-weighted figure is
    reported.
24. **Sides are separate simulations.** *(PM: the decision covers short
    only; long and both are informational.)*
25. **Eligibility** (§3.6): a complete 33.5h of 5m bars before the session,
    a listing age of at least 30 days, and at least $2M a day of quote volume
    over the last 7 days.
26. **Completeness ≥ 90%** for 1h/4h/daily/weekly bars and sessions.
    Complete 15m bars only.
27. **The tick is inferred per symbol-month** from printed prices (§3.4).
28. **Borrow** is charged per started hour, on entry notional.

**Unverified assumptions:**
- **Margin borrow.** Every short is assumed borrowable on Binance margin at
  0.03% a day for every symbol on every day. Historical margin availability is
  not in the cache.
- **Liquidity at the limit.** A trade-through by one tick is assumed to fill
  the whole order. This holds for small size; it is not modelled for large
  size.
- **Slippage.** Stop slippage of 0.08% is assumed adequate at 5m resolution.
  Fast moves on small caps could be worse; the 2× case covers part of this.
- **Exchange hiccups.** There is no model of order rejections, API latency or
  exchange outages beyond missing bars.
- **Survivorship.** The universe was chosen in 2026, although the 4 delisted
  pairs are kept up to their delisting.
- **Futures-style playbook.** The playbook was designed around the NY equity
  open on futures. Whether crypto liquidity behaves the same way at 09:30 ET
  is exactly what is being tested.

## 14. What a pass means

**A CANDIDATE is a candidate only.** Before it can be added to the bot's
`VALIDATED_STRATEGIES`, three things are required:

1. **An independent re-run.** It is done by someone other than the first
   runner, from a fresh `git worktree` at the run commit, with
   `RESEARCH_OFFLINE=1`.
   - It must reproduce the data fingerprint, every verdict, and every reported
     number to 1e-9 relative.
   - The re-runner must also **check `research/smc.py` against this document**,
     rule by rule. Any mismatch voids the run until it is explained.
2. **A 4–8 week forward paper test.** Its pass rule, minimum trade count and
   duration are pre-registered and committed before it starts, as
   `research/carry_forward.py` did. It runs on live data with the same rules,
   and it records news overlap (§8.3).
3. **Its own executor, with its own review.** The bot's current entry logic
   cannot place CE limit orders with TP1/TP2/breakeven management, a
   killzone and per-symbol day/week limits. Bringing the strategy in needs a
   reviewed executor and tests. Going live behind the evidence gate is the
   owner's call, reviewed by the PM.

If nothing passes, RESULTS.md records that the playbook fails as
pre-registered, and there is no re-tuning.

## 15. What may not change after results are seen

None of the following may change after results are seen:
- the grid, the sides, and which side decides (short);
- every constant in `research/smc.py`, §4: swing N, the ATR length,
  displacement, the FVG minimum, the equal-level tolerance, the reclaim and
  MSS windows, the stop buffer, TP1, the TP2 minimum, the order TTL, the time
  stop, the killzone and sessions;
- the bias algorithm, the scoring and the grades;
- the fill model and the costs;
- the limits and their scope;
- the eligibility and completeness rules, and the tick rule;
- the periods, END, and every decision threshold.

**Bug fixes** are allowed only under all of these conditions:
- a test demonstrates the bug;
- the fix is documented with before and after numbers for all 8
  configurations × 3 sides;
- it is labelled as a post-hoc correction.

**Any further analysis** is labelled exploratory and cannot produce a
CANDIDATE.

## 16. Running (after approval only)

1. Write `research/smc_study.py` and a synthetic end-to-end test. Commit both
   **before** the first real-data run. The runner must:
   - load bars with `smc.load_symbol` (offline, END-capped);
   - build each symbol with `smc.prepare` over `smc.trading_days()`;
   - compute SMT reference times with `smc.ref_taken_times` (BTC and ETH);
   - generate setups with `smc.find_candidates`;
   - simulate and price them with `smc.attach_outcomes`;
   - apply gates and limits with `smc.walk`;
   - summarise with `smc.split` and `smc.summarize`;
   - decide with `smc.verdict(base, stress_2x_oos, portfolio_oos)`;
   - count the §10 sanity counters with `walk(stats=)` and
     `smc.spot_mismatch_days`.

   It should keep only setups and day contexts per symbol, not the bars
   (memory on the Pi).

   It must also:
   - refuse to run on a dirty tree or without `RESEARCH_OFFLINE=1`;
   - record the commit, the data fingerprint and END;
   - compute everything in one invocation;
   - print no statistic until all output is written.

   The synthetic end-to-end test monkeypatches `smc.load_symbol`. It uses at
   least 3 symbols, including BTC and ETH, and covers the full grid × all
   sides, the 2× run, the portfolio variant, the verdicts and the output
   files.
2. **No smoke or partial real-data runs.**
3. **The first real-data run is the run of record.** Its output is committed
   unchanged, and the write-up (`research/RESULTS_SMC.md`) follows.

```
TRADING_ENV_FILE=/nonexistent DISCORD_WEBHOOK_URL= RESEARCH_OFFLINE=1 \
  .venv/bin/python -m research.smc_study
```

## Appendix A. Data coverage (data-only pass, no strategy statistics)

Produced by `research/smc_data_check.py` on 2026-10-20 (download
04:48–05:32 UTC). These are the only real-data quantities computed so far:
counts, timestamps and gaps. No returns, signals, setups or trades.

**Download**
- **Command:** `research.smc_data_check --download --workers 3`. Each worker
  runs through `load_klines` at about 4 requests a second. That is about
  20,000 requests in total, at roughly 500–1,000 request weight a minute
  against Binance's 6,000 a minute; the live bot's use at the start was
  12 a minute.
- **Runtime:** 2,635 s for the download, about 165 s per full-history symbol
  per worker.
- **Disk:** 50 files, **1,095 MB**, in
  `/home/admin/trading_data/research/klines_5m/`.
- **Window:** the cache covers 2022-10-01 00:00 → 2026-10-06 23:55 UTC (bar
  opens), so its last bar closes exactly at END. Nothing later was
  downloaded.

**Coverage**
- **Total:** 19,557,592 5m bars. A full-history symbol has 422,480 of the
  422,496 grid slots.
- **The only gap anywhere** is a Binance-wide outage on **2023-03-24 12:40 →
  13:55 UTC** (16 bars, 08:40–09:55 NY). It hit all 44 symbols listed at the
  time. No symbol has any other missing bar.
  - Under §3.6 this makes the trading day 2023-03-24 ineligible for every
    symbol. The 1h and 4h bars it touches are incomplete.
  - The daily bar (94% present) and the weekly bar stay valid.

**Listed after the data start**

| symbol | first 5m bar (UTC) |
|---|---|
| APT | 2022-10-19 01:00 |
| ARB | 2023-03-23 15:00 |
| SUI | 2023-05-03 12:00 |
| WLD | 2023-07-24 09:00 |
| SEI | 2023-08-15 12:00 |
| TIA | 2023-10-31 16:00 |
| BLUR | 2023-11-24 09:00 |
| RENDER | 2024-07-26 08:00 |
| AGIX | 2023-02-17 08:00 |

RENDER's history starts at the RNDR → RENDER spot rename. The older RNDRUSDT
pair is a different symbol and is not included. Each of these symbols becomes
eligible 30 days after its first bar (§3.6).

**Delisted (last 5m bar opens at 02:55 UTC; delisted at 03:00 UTC)**

| symbol | last 5m bar (UTC) |
|---|---|
| OCEAN | 2024-07-01 02:55 |
| AGIX | 2024-07-01 02:55 |
| MATIC | 2024-09-10 02:55 |
| MKR | 2025-09-15 02:55 |

None of them has a bar after delisting, so none can trade after it (§3.5).

**Bars with zero volume** (Binance prints o = h = l = c with no trades).
These are bars that exist, so they are not gaps.
- **Most:** ILV 13,933 (3.3% of its bars), FLOW 12,391, SNX 11,081, EGLD
  9,154, BLUR 7,668 and RUNE 7,120. Then COMP 4,656, STX 4,358, MANA 3,106,
  ENJ 2,656, QNT 2,589 and MKR 2,231.
- **Fewest:** most large caps have 14. For BTC and ETH they all fall on
  2023-03-24 between 11:00 and 12:40 UTC: trading had halted just before
  the outage gap.
- **Kept as they are.** A flat bar can be neither a strict swing nor a
  displacement. The liquidity filter (§3.6, ≥ $2M a day) removes the thinnest
  days.
