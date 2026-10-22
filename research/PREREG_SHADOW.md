# Shadow learner — pre-registration

**Status:** written 2026-10-21, before T0. Reviewed by the PM before T0.
Code: `shadow.py`, hooks in `bot.py` (`_shadow_capture`, `_shadow_housekeeping`,
`_shadow_job`), tests in `tests/test_shadow.py` and
`tests/test_exit_helpers_parity.py`.

## Question

Does any exact strategy name among the short signals the evidence gate refuses
have a positive expectancy after costs, measured forward on data nobody has
seen, under the bot's own exit rules?

The shadow check on 2026-10-21 (`research/SHADOW_CHECK_2026-10-21.md`) found
the refused signals no better than random. **Expect zero candidates.** A
well-measured "no" is a valid result.

## T0 and the data

- **T0** is the `opened_ts` of the first shadow trade the merged code records.
  It is written once to the portfolio key `shadow_t0` and never changes.
- **Evaluation dates:** T0 + 56 days and T0 + 112 days. These are the only two
  times a name can become a candidate.
- **Data at an evaluation date D:** shadow trades opened in [T0, D) and closed by D.
- **Eligible names:** the exact strategy names of short entry rows journaled in
  `trades` in the 14 days before T0. Names first seen after T0 are collected
  and reported but can never become candidates.
- Running numbers before an evaluation date are shown on the dashboard marked
  "not evidence before <date>". They decide nothing.

## Population

A shadow trade is opened for every short entry the evidence gate refuses
(`reason = no_validated_edge`), at the moment of refusal, at most one open
shadow trade per (symbol, exact strategy name). The signal has already passed
the circuit-breaker gate; it has **not** been through the later gates (short
slots, duplicate-position guard, margin, correlation, exposure, SAC sizing
veto, minimum notional, edge-too-small). A validated name would still face all
of those gates in real trading, so real trades of a promoted name are a subset
of its shadow population.

## Entry levels and costs

- Entry price: the signal price × (1 − `SLIPPAGE_PCT`), as a real short fills.
- Initial stop and target: `brain.get_stop_take` then
  `risk_engine.apply_dynamic_rr` on that price, with the same confidence
  multiplier a real entry uses (`position_size_mult` ×
  `dynamic_conviction_size_mult(ml_prob, "short")`).
- Notional: a fixed `SHADOW_NOTIONAL_USD` = $150 (about the median real short).
- Exit fill: the exit price × (1 + `SLIPPAGE_PCT`), on every exit including
  take-profits, as the bot books it.
- Fees: `accounting_v2.net_realized_pnl` (the bot's own round-trip fee).
- Borrow: 0.03% per day of short notional (the bot books none; the research
  studies did).
- **R** = initial stop − entry, times shares. The metric is **net R per closed
  shadow trade**.

## Exit model (per closed 1-minute candle)

**The entry minute** (the minute containing the entry time) is tested for the
stop only, on its high: part of its range printed before the entry, so the
order of events is unknown. It never books a target and never moves a level.

Every later minute, in this order, using the bot's own pure helpers:
1. Survival stop: open P&L at the close below `HARD_STOP_LOSS_USD` → exit at the close.
2. Trailing stop: `bot.trail_stop_move`.
3. Break-even: `bot.breakeven_stop_move`.
4. Designed time decay: `bot.designed_decay_moves` (age = wall-clock minutes
   since entry at the candle's close).
5. Stop in place at the minute's start, tested on the high → exit at
   max(stop, open). Ambiguity goes against the trade: this is tested before any target.
6. Target in place at the minute's start, tested on the low, only if
   `bot._profit_clears_fees` allows it → exit at the target.
7. A stop moved this minute, tested on the close → exit at the close.
8. A target moved this minute, tested on the close, fee-gated → exit at the target.
9. Losing trade open ≥ `MAX_HOLD_OPEN_SECONDS` (4 h) → exit at the close.
10. Shadow cap: open ≥ 48 h → exit at the close.

**Missing data:** a shadow trade is never resolved across a missing candle. If
the next candle is missing but later ones exist, it waits 5 minutes for the
websocket backfill (requires `fix/ws-backfill` to be merged first); after that
it closes at its current stop (`gap_stop`). With no candles at all for 30
minutes it also closes at its stop. Gap stops are counted and reported.

## Every known difference from the live bot

1. Closed 1-minute candles instead of ~2-second websocket ticks (rules 5–8
   above approximate intra-minute order; the live bot exits a touched target on
   the touching tick).
2. A fixed $150 notional, so the survival stop triggers at a fixed % move; real
   sizes vary.
3. The later entry gates listed under *Population* are not applied.
4. One open shadow trade per (symbol, name); the live bot allows one short per symbol.
5. Capture happens on every refusal; the skip journal is throttled to one row
   per symbol per 5 minutes.
6. A 48-hour cap; the live bot has no cap for winning trades.
7. Gap stops as described; the live bot keeps holding through a data gap.
8. ATR from the 299 closed candles before the current one plus the current
   candle (300 = `CANDLE_LIMIT`, the same make-up as the bot's cache). At a
   live decision the current candle is still forming; in the shadow it is complete.
9. 0.03%/day borrow is charged; the live bot books none.
10. MFE/MAE, the RL experience log and the adaptive edge learner are not updated.

A candidate is evidence only for this exit model and this population.

## Decision rule (binding, fixed now)

At an evaluation date, an eligible name is a **candidate** only if all hold:

| check | threshold |
|---|---|
| closed shadow trades | ≥ 300 |
| distinct UTC days with a trade | ≥ 20 |
| mean net R per trade | ≥ +0.10 |
| day-clustered t-stat of the trade mean | ≥ the bar below (at least 3.6) |
| mean net R in each half of its days | > 0 in both |
| largest single-symbol share of total net R | ≤ 25% |

- **Day-clustered t** = mean net R per trade ÷ its standard error clustered by
  UTC day of `opened_ts`: SE = √(G/(G−1) · Σ_g (Σ_{i∈g} (r_i − mean))²) / N, the
  same formula as `research.smc.cluster_t` (a test pins the two together).
- **Halves** split the name's days at the median day; each half's figure is
  the mean over its trades.
- **The bar (multiple testing):** one-sided z for 0.05 / (2 looks × N), where N
  is the number of names frozen at T0, and never below 3.6. Each evaluation
  records N and the bar. (153 names were seen in 8 days before this was
  written, so N will likely be above 150; z for 150 names ≈ 3.59.)

Every eligible name's numbers are reported at each evaluation, not only
candidates.

## What a candidate leads to (promotion is manual)

1. A different agent checks the code against this document.
2. PM review.
3. The owner adds the exact name to `config.VALIDATED_STRATEGIES`.

Nothing promotes automatically.

## Demotion (automatic; it only removes permission to trade)

A name in `VALIDATED_STRATEGIES` is added to the runtime blocklist (portfolio
key `evidence_blocklist`) when its last 50 closed real trades have a mean net
P&L ≤ $0. The gate refuses blocklisted names. Only the owner clears the
blocklist.

## Switches and data integrity

`SHADOW_LEARNER=false` (env) turns the shadow learner off. It never affects a
real decision: capture is in memory only and never raises; all database work
runs in a worker thread with its own transactions.

- The capture queue is drained only by the worker after it holds its lock, so
  a run that is still busy (or timed out) never loses records.
- If the queue overflows its bound (20,000), the evicted records are counted
  (`dropped` in the portfolio key `shadow_stats`) and reported at each evaluation.
- The demotion blocklist is loaded at boot, before any entry is evaluated, and
  refreshed every minute.
- Before deploy, the live database must have no `shadow_t0` key (T0 is set by
  the first shadow trade the merged code records). Sandbox runs use a scratch
  database only.
