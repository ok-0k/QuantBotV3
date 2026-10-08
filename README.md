# ⚡ Quant Bot v3: Pi crypto trading bot (paper)

A 24/7 crypto trading bot for a Raspberry Pi 5. It runs on Binance market data
and executes **paper trades only**: there is no live-money mode. It has a short-only
signal engine, an XGBoost confidence gate, SAC reinforcement-learning sizing,
adaptive edge profiles, a full trade journal, symbol-split experiments, and an
offline research toolkit with a cost-aware backtester.

## Current status (2026-10-07)

- **It does not make money, and research says tuning won't fix that.** The bot's
  entries perform the same as shorting at random times. No simple price-based
  signal tested (17 strategies, held-out data, realistic costs) beats trading
  costs. Details are in [`research/RESULTS.md`](research/RESULTS.md).
- **The exits were buggy and are now fixed.** The fee gate could disable stops.
  Take-profits could book net losses. Stops filled at stale prices. The time
  decay collapsed targets within minutes.
- **An experiment is running.** `decay_v2` splits the symbols into two groups to
  compare the legacy time decay with the decay as designed. Check it with
  `scripts/experiment_report.py`.
- **One real edge was found: perpetual-futures funding carry.** It is
  market-neutral and depends on the regime: about 18% APR on deployed capital in
  2023–25, but rarely active now. A forward test was frozen on 2026-10-07 in
  `research/carry_forward.py`.

---

## Architecture

```
Binance WS (1m klines, 50 symbols)
   │  every update (~2 s) ──► _tick_exit_check: stops / targets / trailing / decay
   │  closed candle ────────► brain.get_ensemble_signal (short-only)
   │                              ├─ filters: squeeze / bull-DI / ADX / BTC-ETH macro
   │                              ├─ XGBoost gate: P(down) ≥ MIN_ML_CONFIDENCE
   │                              └─ _prepare_trade: SAC size × conviction × edge size_mult
   │                                     ├─ order journal (idempotent client ids)
   │                                     └─ paper fill ──► _commit_trade ──► trade journal
   └─ exit_monitor (30 s): stops / targets / survival stop / 4h time exit / equity / circuit breaker

SQLite WAL (/home/admin/trading_data/trading.db) ◄── dashboard.py (FastAPI + SSE, port 8000)
Nightly cron 04:00 (Pi time): stop bot → prune_db.sh → start bot → restart dashboard
```

## File map

| Path | Role |
|------|------|
| `bot.py` | Async orchestrator: websocket, entry pipeline (`_prepare_trade` / `_commit_trade`), exit paths, experiment arms |
| `brain.py` | Short-only signal engine, regime classifier, circuit breaker, adaptive edge profiles |
| `risk_engine.py` | Conviction sizing, volatility R:R, correlation and exposure caps |
| `accounting_v2.py` | Fee-deducted PnL, equity components |
| `execution/` | Order backends: `paper` (default) and Binance Spot **testnet**. No mainnet mode |
| `db.py` | SQLite schema and migrations, trade journal, orders journal |
| `config.py` | All tunables, env overrides, `~/.config/quant-bot/env` loader |
| `dashboard.py` | FastAPI + SSE dashboard (optional `DASHBOARD_AUTH_TOKEN`) |
| `ml_engine.py`, `features.py`, `rl_agent.py` | XGBoost gate, features, SAC actor inference |
| `prune_db.sh` | Nightly maintenance: compressed DB backup first, then trims candles and old equity detail, **never trade data** |
| `insights.py` | Read-only journal analytics behind the dashboard's Insights / Experiment / "How Trades End" sections |
| `scripts/experiment_report.py` | Read-only A/B report for the running experiment |
| `scripts/entry_edge.py` | Read-only: the bot's entries against random entry, using Binance history |
| `research/` | Offline backtester, data cache, pre-registered study, carry analysis ([results](research/RESULTS.md)) |
| `tests/` | pytest suite (224 tests, sandboxed DB, no network) |
| `offline_trainer.py`, `deploy.sh` | Legacy laptop-side training and deploy tooling |
| `trader.py`, `live_trader.py/` | Legacy V1 scripts, not used (V1 cron disabled 2026-10-06) |

## Trading logic

**Entries** are shorts only, evaluated on each closed 1m candle:
1. The brain blocks squeeze setups (high relative volume in an uptrend, bullish
   DI dominance, strong ADX uptrend, BTC/ETH macro risk-on).
2. A composite score combines ML bearish tilt and market structure. The XGBoost
   gate requires `1 − P(up) ≥ 0.80`.
3. Size is the SAC fraction × ML conviction × the adaptive edge `size_mult`,
   capped at `MAX_ORDER_EQUITY_FRAC` (15%) of equity. Shorts are capped at 5 open
   at once, split 3 + 3 by experiment group while one runs.

**Exits:**
- **Initial levels:** an ATR stop above entry and an ATR target below it.
- **Trailing stop:** once price passes the target.
- **Break-even stop:** after a confirmed favourable move.
- **Time decay:** depends on the experiment group (see below).
- **Survival stop:** closes a position at −$15.
- **4-hour exit:** closes losing positions after 4 hours.

Exit rules that matter:
- The **fee gate applies to take-profits only**. A take-profit fires only if it
  books a positive net after fees and exit slippage. Stops always fire.
- **Stops fill at market** when price is already beyond them (stop-market
  semantics). Take-profits fill at the target (resting limit).

## Trade journal

Every `trades` row carries:
- `entry_features`: a JSON snapshot of the decision. It includes ML probability
  and tier, score components, ADX/DI, relative volume, returns, ATR, stop and
  target distances, sizing, edge multipliers, the initial stop and target, and the
  experiment group. **Skipped entries get the snapshot too**, labelled `sac_veto`,
  `size_below_min_notional`, `insufficient_margin`, and so on.
- `exit_reason`: one of `TAKE_PROFIT`, `STOP_LOSS`, `TIGHTENED_STOP`,
  `BREAKEVEN_STOP`, `TRAILING_STOP`, `HARD_STOP`, `TIME_HOLD_EXIT`, `MAX_HOLD`,
  `CIRCUIT_BREAKER` or `OTHER`. The raw trigger text goes in `exit_detail`.
- `peak_profit_pct` / `max_drawdown_pct`: the best and worst unrealised move
  while open.
- `hold_time_seconds`: how long the position was open.

Trade data is **kept forever**. The nightly prune never deletes `trades` or
`rl_experience`. Equity history keeps 30 s detail for 2 days, then one point
every 5 minutes.

## Running experiment: `decay_v2`

The legacy time decay ran on every websocket update (~2 s) and shrank the
*current* stop and target each time, so the effect compounded. It collapsed both
onto the entry price within about 5 minutes once a trade passed about 15 minutes.
That is the cause of the old ~20-minute trade lifecycle.

- **How the split works:** symbols alternate between group A and group B in
  `SYMBOLS` order.
- **Group A** keeps the legacy behaviour.
- **Group B** uses the decay as designed: λ comes from the trade's wall-clock
  age and is applied to the *initial* levels.
- **Turning it off:** set `EXPERIMENT_NAME=""`.

```bash
.venv/bin/python scripts/experiment_report.py     # per-group stats + 95% CI on the difference
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
per fill, plus 0.03%/day short borrow. Unit tests prove there is no lookahead.

---

## Operations

| What | Where / how |
|------|-------------|
| Services | `quant-bot`, `quant-dashboard` (systemd; a drop-in at `/etc/systemd/system/quant-*.service.d/10-paths.conf` points them at this directory) |
| Deploy | Commit, then `sudo systemctl restart quant-bot quant-dashboard`. Otherwise the 04:00 cron picks it up |
| Logs | `/dev/shm/trading_logs/bot.log` (RAM, rotates, lost on reboot). The journal in the DB is the durable record |
| Backups | Nightly, before the prune: `/home/admin/trading_data/backups/db/trading-<UTC>.db.gz`, newest 14 kept. To restore, stop the bot, then `gunzip -c <file> > /home/admin/trading_data/trading.db`, then start it. They live on the same SD card, so copy one off the Pi now and then |
| Prune log | `/home/admin/trading_data/prune.log` (backup and prune results) |
| Alerts | Discord alerts are off until `DISCORD_WEBHOOK_URL` is set in `~/.config/quant-bot/env` |
| Dashboard | `http://<pi-ip>:8000` |
| Pre-deploy smoke test | Run `bot.py` against a copy of the DB with `TRADING_DATA_DIR` / `TRADING_LOG_DIR` pointed at a scratch dir, `TRADING_ENV_FILE=/nonexistent`, `EXECUTION_MODE=paper`, under `timeout -s KILL 75` |

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
| `SYMBOLS` | 50 pairs | 4 are delisted on Binance spot (MATIC, MKR, OCEAN, AGIX). Remove them only after the experiment ends, because groups are assigned by list position |
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
| `MAX_DRAWDOWN_PCT` / `CB_RECOVERY_PCT` | 20% / 10% | Circuit breaker trip / re-arm |
| `STALE_ENTRY_MAX_SECS` / `WS_BACKFILL_AFTER_SECS` | 180 / 90 | Stale-data entry gate, REST backfill after a websocket outage |
| `EXPERIMENT_NAME` | `decay_v2` | Empty string disables experiments |

## Execution modes and order protections

- `EXECUTION_MODE=paper` (default): simulated fills, pinned by
  `tests/test_executor.py`.
- `EXECUTION_MODE=testnet`: Binance Spot testnet. It needs
  `BINANCE_TESTNET_API_KEY` / `_SECRET` in `~/.config/quant-bot/env` (chmod
  600). There is deliberately no mainnet mode.
- Protections that apply in every mode:
  - **Duplicate orders:** deterministic client-order ids, claimed in the
    `orders` journal before money moves.
  - **Stale data:** stale-candle entry gate.
  - **Oversized orders:** per-order equity clamp.
  - **Restarts:** boot-time reconciliation and a cash-invariant check.
  - **Network loss:** websocket backoff plus REST backfill.
  - **Testnet:** partial fills and retries are handled idempotently.

## Secrets

Never hardcode credentials. `config.py` loads `~/.config/quant-bot/env`
(KEY=VALUE, chmod 600), and real environment variables win. Known keys:
`DISCORD_WEBHOOK_URL`, `DASHBOARD_AUTH_TOKEN`, `BINANCE_API_KEY`,
`BINANCE_API_SECRET`, `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`.
The pre-commit hook (`scripts/pre-commit`) blocks staged lines that look like
credentials.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # 224 tests; throwaway temp DB, never the live one
```

## Pi setup (first time)

1. Flash Raspberry Pi OS Lite (64-bit) and enable SSH.
2. Clone the repo, then run `sudo bash setup.sh`. It creates the venv, the
   tmpfs log dir and the systemd units, and sets `gpu_mem=16`. Check that the
   unit paths match your checkout (see Operations).
3. Start the services with `sudo systemctl start quant-bot quant-dashboard`.
