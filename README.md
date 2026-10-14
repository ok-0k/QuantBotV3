# ⚡ Quant Bot v3: Pi crypto trading bot (paper)

A 24/7 crypto trading bot for a Raspberry Pi 5. It runs on Binance market data
and executes **paper trades only**: there is no live-money mode. It has a short-only
signal engine, an XGBoost confidence gate, SAC reinforcement-learning sizing,
adaptive edge profiles, a full trade journal, symbol-split experiments, and an
offline research toolkit with a cost-aware backtester.

## Current status (2026-10-11)

- **It does not make money, and research says tuning won't fix that.** The bot's
  entries perform the same as shorting at random times. No simple price-based
  signal tested (17 strategies, held-out data, realistic costs) beats trading
  costs. Details are in [`research/RESULTS.md`](research/RESULTS.md).
- **An October 2026 audit fixed the money path and the plumbing** (live since
  2026-10-11, rollback tag `pre-audit-2026-10`):
  - Each trade is booked in **one database transaction**, so a crash can no
    longer leave cash, positions and the journal out of step.
  - A **ledger check runs every minute** and shows any drift on the dashboard.
  - The **circuit breaker now blocks shorts**. Before, it only blocked buys,
    which this short-only bot never makes.
  - **Crash safety:** a dead core loop restarts the bot instead of leaving it
    half-running, a second instance can't start, repeated crashes pause instead
    of looping, and one bad position can't stop exit checks for the others.
  - **Backups are integrity-checked**, and a restore script refuses to run while
    the services are up.
  - The **dashboard** gained health tiles (heartbeats, ledger, breaker headroom,
    model ages) and shows times in Pacific time.
- **The deferred exit fixes are on.** They sit behind `V3_FIXES` (see
  [Deferred exit fixes](#deferred-exit-fixes-v3_fixes)), which turned on when
  the experiment ended on 2026-10-14.
- **The designed time decay won its experiment and is now the default.**
  `decay_v2` compared the legacy time decay with the decay as designed. Group B
  lost about $1.63 less per symbol per day (95% CI $1.30 to $1.97), and the gap
  held before and after the audit deploy. It loses less mostly by trading less;
  it is still a net loss, not an edge.
- **The ML models don't add an edge.** The audit found the XGBoost gate is
  anti-predictive for shorts (AUC ≈ 0.36), and the SAC actor vetoes about 96% of
  trend-down shorts. Both models are static: incremental retraining never
  triggers, and the XGBoost model is about 6 months old. Retiring them is on
  the backlog.
- **Funding carry fails out of sample too, so no strategy passes.** Its earlier
  out-of-sample "confirmation" came from phantom hedges in pairs that had
  stopped trading on spot (MATIC, AGIX, BLUR), which the audit found. With
  hedges limited to hours when the spot pair trades, carry earned about 18% APR
  on deployed capital in sample (2023–24 bull-market funding) but lost money out
  of sample (OOS Sharpe −2.4). The frozen forward test in
  `research/carry_forward.py` keeps tracking it.

---

## Architecture

```
Binance WS (1m klines, 50 symbols)
   │  every update (~2 s) ──► _tick_exit_check: stops / targets / trailing / decay
   │  closed candle ────────► brain.get_ensemble_signal (short-only)
   │                              ├─ gates: circuit breaker / slots / correlation / exposure (blocks are journalled)
   │                              ├─ filters: squeeze / bull-DI / ADX / BTC-ETH macro
   │                              ├─ XGBoost gate: P(down) ≥ MIN_ML_CONFIDENCE
   │                              └─ _prepare_trade: SAC size × conviction × edge size_mult
   │                                     ├─ order journal (idempotent client ids)
   │                                     └─ paper fill ──► _commit_trade (one DB transaction) ──► trade journal
   ├─ exit_monitor (30 s): each position checked in isolation: stops / targets / survival stop / 4h time exit,
   │                       then equity snapshot + circuit breaker
   └─ housekeeping (60 s): ledger check, edge profiles
Supervisor: if any of the three loops dies, the process exits and systemd restarts it.

SQLite WAL (/home/admin/trading_data/trading.db) ◄── dashboard.py (FastAPI + SSE, port 8000)
Nightly cron 04:00 (Pi time): stop bot → prune_db.sh (verified backup, then prune) → start bot → restart dashboard
```

## File map

| Path | Role |
|------|------|
| `bot.py` | Async orchestrator: websocket, entry pipeline (`_prepare_trade` / `_commit_trade`), exit paths, experiment arms, supervisor, instance lock |
| `brain.py` | Short-only signal engine, regime classifier, circuit breaker, adaptive edge profiles |
| `risk_engine.py` | Conviction sizing, volatility R:R, correlation and exposure caps |
| `accounting_v2.py` | Fee-deducted PnL, equity components |
| `execution/` | Order backends: `paper` (default) and Binance Spot **testnet**. No mainnet mode |
| `db.py` | SQLite schema and migrations, `transaction()`, trade and order journals, ledger check |
| `config.py` | All tunables, env overrides, `~/.config/quant-bot/env` loader |
| `dashboard.py` | FastAPI + SSE dashboard (optional `DASHBOARD_AUTH_TOKEN`) |
| `ml_engine.py`, `features.py`, `rl_agent.py` | XGBoost gate, features, SAC actor inference |
| `insights.py` | Read-only journal analytics behind the dashboard's Insights / Experiment / "How Trades End" sections |
| `prune_db.sh` | Nightly maintenance: integrity-checked compressed backup first, then trims candles and old equity detail, **never trade data** |
| `scripts/restore_db.sh` | Restore a nightly backup (refuses while either service runs; dry run by default) |
| `scripts/set_ledger_anchor.py` | Record a reviewed out-of-band cash change so the ledger check reads ~0 (dry run by default) |
| `scripts/reset_circuit_breaker.py` | Re-base the breaker's peak equity after a reviewed drawdown (dry run by default) |
| `scripts/experiment_report.py` | Read-only A/B report for an experiment (pass its name, e.g. `decay_v2`) |
| `scripts/entry_edge.py` | Read-only: the bot's entries against random entry, using Binance history |
| `set_cash_balance.py`, `heal_equity.py` | Manual cash tools. The first records the change in `capital_adjustments`; the second refuses while positions are open |
| `research/` | Offline backtester, data cache, pre-registered study, carry analysis ([results](research/RESULTS.md)) |
| `tests/` | pytest suite (335 tests, sandboxed DB, no network) |
| `offline_trainer.py`, `deploy.sh` | Legacy laptop-side training and deploy tooling |
| `trader.py`, `live_trader.py/` | Legacy V1 scripts, not used (V1 cron disabled 2026-10-06) |

## Trading logic

**Entries** are shorts only, evaluated on each closed 1m candle:
1. Entry gates come first. No new position opens while the circuit breaker is
   tripped, all slots are full, or the correlation or exposure caps say no.
2. The brain blocks squeeze setups (high relative volume in an uptrend, bullish
   DI dominance, strong ADX uptrend, BTC/ETH macro risk-on).
3. A composite score combines ML bearish tilt and market structure. The XGBoost
   gate requires `1 − P(up) ≥ 0.80`.
4. Size is the SAC fraction × ML conviction × the adaptive edge `size_mult`,
   capped at `MAX_ORDER_EQUITY_FRAC` (15%) of equity. Shorts are capped at 5 open
   at once, split 3 + 3 by experiment group while one runs.

**Exits:**
- **Initial levels:** an ATR stop above entry and an ATR target below it.
- **Trailing stop:** once price passes the target.
- **Break-even stop:** after a confirmed favourable move.
- **Time decay:** pulls the target toward entry and tightens the stop as the
  trade ages, computed from the initial levels (see `decay_v2` below).
- **Survival stop:** closes a position at −$15.
- **4-hour exit:** closes losing positions after 4 hours.

Exit rules that matter:
- The **fee gate applies to take-profits only**. A take-profit fires only if it
  books a positive net after fees and exit slippage. Stops always fire.
- **Stops fill at market** when price is already beyond them (stop-market
  semantics). Take-profits fill at the target (resting limit).

**Circuit breaker:** when equity falls 20% below its recorded peak, new entries
stop. Open positions keep their exits. Entries resume once the drawdown recovers
to 10%, or after a reviewed `scripts/reset_circuit_breaker.py --confirm` and a
restart.

### Deferred exit fixes (`V3_FIXES`)

These change trading decisions, so they stayed off while `decay_v2` ran. They
are on by default since it ended. With env `V3_FIXES=false`, decisions and
fills are identical to the pre-audit code.

- **Stops and targets only tighten.** Without it, the time decay can overwrite a
  break-even or trailing stop with a looser level.
- **Tick checks run in a fixed order:** survival stop, then stop, then target.
- **Slots are reserved while an order is in flight**, so two entries can't
  overshoot the slot cap.
- **Exit learning uses the position's own regime**, not the regime of whichever
  symbol was evaluated last.

### Evidence gate (`ENTRY_EVIDENCE_GATE`, off by default)

The entry signal has shown no edge (see Research), so every trade it opens is
a guess that pays fees. With the gate on, a new entry is allowed only if its
exact strategy name is in `VALIDATED_STRATEGIES`: signals that passed a
pre-registered out-of-sample test. Every other signal is journaled as a skipped
row with reason `no_validated_edge`, at most one per symbol every 5 minutes, so
its features and forward returns can still be studied at no cost. Exits are
never gated. The list is empty today, so turning the gate on means the bot
opens no new positions until a signal passes. Env `ENTRY_EVIDENCE_GATE=true`
turns it on. The dashboard's System section shows an Evidence Gate tile, and
the bot logs one warning at startup while the gate is on with nothing validated.

## Trade journal

Every `trades` row carries:
- `entry_features`: a JSON snapshot of the decision. It includes ML probability
  and tier, score components, ADX/DI, relative volume, returns, ATR, stop and
  target distances, sizing, edge multipliers, the initial stop and target, and the
  experiment group.
- **Skipped entries are journalled too**, with the reason in `reason`:
  `sac_veto`, `size_below_min_notional`, `insufficient_margin`,
  `circuit_breaker`, `slots_full`, `corr_block`, `exposure_block`,
  `margin_halt` and `no_validated_edge`. Gate blocks are rate-limited to one
  row per symbol and reason every 5 minutes.
- `exit_reason`: one of `TAKE_PROFIT`, `STOP_LOSS`, `TIGHTENED_STOP`,
  `BREAKEVEN_STOP`, `TRAILING_STOP`, `HARD_STOP`, `TIME_HOLD_EXIT`, `MAX_HOLD`,
  `CIRCUIT_BREAKER` or `OTHER`. `exit_detail` holds the raw trigger text plus
  the mark price and the stored stop and target at exit.
- `regime` on exit rows is the regime the position was entered in.
- `peak_profit_pct` / `max_drawdown_pct`: the best and worst unrealised move
  while open.
- `hold_time_seconds`: how long the position was open.

Trade data is **kept forever**. The nightly prune never deletes `trades` or
`rl_experience`. Equity history keeps 30 s detail for 2 days, then one point
every 5 minutes.

## Experiment `decay_v2` (2026-10-06 to 2026-10-14, B adopted)

The legacy time decay ran on every websocket update (~2 s) and shrank the
*current* stop and target each time, so the effect compounded. It collapsed both
onto the entry price within about 5 minutes once a trade passed about 15 minutes.
That is the cause of the old ~20-minute trade lifecycle.

- **How the split works:** symbols alternate between group A and group B in
  `SYMBOLS` order.
- **Group A** keeps the legacy behaviour.
- **Group B** uses the decay as designed: λ comes from the trade's wall-clock
  age and is applied to the *initial* levels.
- **Headline metric:** net P&L per symbol per day for each group, with a 95%
  confidence interval from resampling days. Group B makes far fewer trades, so
  per-trade averages would mislead.
- **Pre-registered decision rule:** adopt B unless the interval shows B
  significantly worse.
- **Caveat:** group B ran without persistent break-even and trailing stops,
  which `V3_FIXES` fixes.
- **Result:** B adopted. Every short with recorded initial levels now uses the
  designed decay. The legacy path remains for group A, so the experiment can be
  re-run with env `EXPERIMENT_NAME=decay_v2`, and for longs, which record no
  initial levels and kept the legacy decay in both groups.

```bash
.venv/bin/python scripts/experiment_report.py decay_v2   # per-group stats, per-symbol-day CI, decision rule
```

## Research

The pre-registered study (strategy grid, periods, costs and pass/fail rules
committed before any result was seen) is in `research/`:

```bash
.venv/bin/python -m research.data_check           # refresh the cached Binance data + quality report
.venv/bin/python -m research.study --band 0.5     # 19-strategy grid, IS 2023-01..2025-04, OOS 2025-04..now
.venv/bin/python -m research.carry_basis          # carry economics incl. basis risk (exploratory)
.venv/bin/python -m research.carry_forward        # frozen forward test of funding carry
.venv/bin/python scripts/entry_edge.py            # the live bot's entries vs random entry
```

Data is cached in `/home/admin/trading_data/research/` (outside the repo). The
backtester charges the same costs as the paper model: 0.10% fee + 0.08% slippage
per fill, plus 0.03%/day short borrow. Hedge legs are only allowed in hours when
the spot pair actually trades. Unit tests prove there is no lookahead.

---

## Operations

| What | Where / how |
|------|-------------|
| Services | `quant-bot`, `quant-dashboard` (systemd, `Restart=always`; a drop-in at `/etc/systemd/system/quant-*.service.d/10-paths.conf` points them at this directory) |
| Deploy | Commit, then `sudo systemctl restart quant-bot quant-dashboard`. Otherwise the 04:00 cron picks it up |
| Rollback | `git reset --hard <tag>` in this directory, then restart both services. Deploys are tagged before they go live (e.g. `pre-audit-2026-10`) |
| Logs | `/dev/shm/trading_logs/bot.log` (RAM, rotates, lost on reboot). Only warnings and above go to the systemd journal, to spare the SD card. The journal in the DB is the durable record |
| Health | The dashboard's System tiles show the bot, exit-monitor, websocket and housekeeping heartbeats, the ledger check, model ages and `V3_FIXES`. A banner appears when a heartbeat is more than 90 s old |
| Crashes | If a core loop dies, the bot exits and systemd restarts it. A second unexpected exit within 15 minutes pauses 120 s first, so a persistent fault can't become a tight restart loop. The marker file is `/home/admin/trading_data/last_unexpected_exit`. A lock on `/home/admin/trading_data/bot.lock` stops a second instance |
| Ledger check | Every minute: drift = books − (`STARTING_CASH` + `capital_adjustments` + realised net P&L). A drift that stays constant is an unrecorded out-of-band cash change: review it, then record it with `scripts/set_ledger_anchor.py --confirm --amount <drift>`. A drift that moves means a booking leak |
| Backups | Nightly, before the prune: `/home/admin/trading_data/backups/db/trading-<UTC>.db.gz`, newest 14 kept. Each must pass `PRAGMA quick_check` first; a snapshot that fails is kept as `*.CHECK-FAILED.db` and nothing is rotated or pruned. They live on the same SD card, so copy one off the Pi now and then |
| Restore | Stop **both** services (`sudo systemctl stop quant-bot quant-dashboard`), run `scripts/restore_db.sh <file>` (the dry run verifies the backup), then `scripts/restore_db.sh <file> --yes`, then start both. The old files are moved aside, never deleted. Never overwrite `trading.db` while either service runs: they hold the WAL open and the result can be corrupt |
| Prune log | `/home/admin/trading_data/prune.log` (backup and prune results) |
| Alerts | Discord alerts are off until `DISCORD_WEBHOOK_URL` is set in `~/.config/quant-bot/env` |
| Dashboard | `http://<pi-ip>:8000`. Times are shown in America/Vancouver |
| Pre-deploy smoke test | Run `bot.py` against a copy of the DB with `TRADING_DATA_DIR` / `TRADING_LOG_DIR` pointed at a scratch dir, `TRADING_ENV_FILE=/nonexistent`, `EXECUTION_MODE=paper`, under `timeout` (15 minutes covers the ledger check and heartbeats) |

```bash
sqlite3 /home/admin/trading_data/trading.db \
  "SELECT exit_reason, COUNT(*), ROUND(SUM(pnl),2) FROM trades
   WHERE action='cover' AND status='filled' AND exit_reason IS NOT NULL GROUP BY 1;"
vcgencmd measure_temp; free -h
```

## Key configuration (`config.py`)

| Constant | Value | Notes |
|----------|-------|-------|
| `STARTING_CASH` | 10,000 | Paper balance |
| `SYMBOLS` | 50 pairs | 4 are delisted on Binance spot (MATIC, MKR, OCEAN, AGIX). Removing them shifts the list positions the `decay_v2` report uses to rebuild its groups |
| `EXECUTION_MODE` | paper | `paper` or `testnet` (env). **No live mode** |
| `SHORT_MAX_OPEN` | 5 | 3 per group while an experiment runs |
| `MAX_ORDER_EQUITY_FRAC` | 15% | Hard per-order clamp |
| `_SAC_SIZE_CEILING` (bot.py) | 35% | SAC sizing ceiling before the edge multiplier |
| `MIN_ML_CONFIDENCE` | 0.80 | Entry ML gate |
| `SHORT_STOP_LOSS_ATR_MULT` / `SHORT_TAKE_PROFIT_ATR_MULT` | 3.5× / 10× | Initial short levels |
| `DECAY_HALFLIFE_CANDLES` / `DECAY_MIN_CANDLES` | 240 / 30 | Designed decay: τ = 240 min, starts after 30 min |
| `FEE_GATE_ROUND_TRIP` | 0.20% | 0.12% with the opt-in BNB discount |
| `SLIPPAGE_PCT` | 0.08% | Per paper fill |
| `HARD_STOP_LOSS_USD` | −15 | Survival stop per position |
| `MAX_HOLD_OPEN_SECONDS` | 14,400 | Closes losing positions after 4h |
| `MAX_DRAWDOWN_PCT` / `CB_RECOVERY_PCT` | 20% / 10% | Circuit breaker trips at 20% drawdown, clears at 10% |
| `STALE_ENTRY_MAX_SECS` / `WS_BACKFILL_AFTER_SECS` | 180 / 90 | Stale-data entry gate, REST backfill after a websocket outage |
| `EXPERIMENT_NAME` | empty | No experiment. Env `EXPERIMENT_NAME=decay_v2` re-runs the decay split |
| `V3_FIXES` | on | Deferred exit fixes. Env `V3_FIXES=false` turns them off |
| `ENTRY_EVIDENCE_GATE` | off | Entries only from `VALIDATED_STRATEGIES` (empty). Env `ENTRY_EVIDENCE_GATE=true` turns it on |

## Execution modes and order protections

- `EXECUTION_MODE=paper` (default): simulated fills, pinned by
  `tests/test_executor.py`.
- `EXECUTION_MODE=testnet`: Binance Spot testnet. It needs
  `BINANCE_TESTNET_API_KEY` / `_SECRET` in `~/.config/quant-bot/env` (chmod
  600). There is deliberately no mainnet mode.
- Protections that apply in every mode:
  - **Duplicate orders:** deterministic client-order ids, claimed in the
    `orders` journal before money moves.
  - **Partial bookings:** cash, position and journal rows are written in one
    transaction or not at all.
  - **Stale data:** stale-candle entry gate.
  - **Oversized orders:** per-order equity clamp.
  - **Restarts:** boot-time reconciliation, a cash-invariant check and the
    once-a-minute ledger check.
  - **Network loss:** websocket backoff plus REST backfill.
  - **Testnet:** partial fills and retries are handled idempotently.

## Secrets

Never hardcode credentials. `config.py` loads `~/.config/quant-bot/env`
(KEY=VALUE, chmod 600), and real environment variables win. Known keys:
`DISCORD_WEBHOOK_URL`, `DASHBOARD_AUTH_TOKEN`, `BINANCE_API_KEY`,
`BINANCE_API_SECRET`, `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`.
The pre-commit hook (`scripts/pre-commit`) blocks staged lines that look like
credentials. Tests remove exchange keys and secrets from their environment
and blank the webhook.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q                   # 349 tests; throwaway temp DB, never the live one
V3_FIXES=false .venv/bin/python -m pytest tests/ -q    # same suite with the deferred exit fixes off
```

## Pi setup (first time)

1. Flash Raspberry Pi OS Lite (64-bit) and enable SSH.
2. Clone the repo, then run `sudo bash setup.sh`. It creates the venv, the
   tmpfs log dir and the systemd units, and sets `gpu_mem=16`. Check that the
   unit paths match your checkout (see Operations).
3. Start the services with `sudo systemctl start quant-bot quant-dashboard`.
