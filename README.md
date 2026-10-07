# ⚡ Quant Bot v3 — Pi  Edge Trading System

Self-improving algorithmic trading bot: XGBoost signal filtering + SAC reinforcement learning position sizing + genetic strategy evolution. Runs 24/7 on a headless Raspberry Pi .

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  asyncio event loop  (bot.py — main process)                    │
│                                                                 │
│  WS Listener ──► closed candle ──► ProcessPoolExecutor          │
│                                      Worker 1: XGBoost infer   │
│                                      Worker 2: SAC forward pass │
│                                                                 │
│  Exit Monitor — ATR stop-loss / take-profit / max hold time     │
│  ML Cron      — incremental XGBoost update every 100 candles   │
└─────────────────────────────────────────────────────────────────┘
         │                              ▲
         ▼ SQLite WAL                   │ reads (non-blocking)
┌─────────────────────┐       ┌─────────────────────────┐
│   trading.db        │       │  dashboard.py (FastAPI)  │
│  • candles          │       │  SSE → browser Chart.js  │
│  • positions        │       │  http://pi-ip:8000       │
│  • trades           │       └─────────────────────────┘
│  • equity_curve     │
│  • rl_experience    │
└─────────────────────┘
```

## File Map

| File | Role | Runs On |
|------|------|---------|
| `bot.py` | Async WebSocket orchestrator + `_execute_trade` executor | Pi (main) |
| `brain.py` | Short-only signal engine + regime + circuit breaker + adaptive edge | Pi |
| `execution/` | Order backends: `paper` (default) and Binance Spot **testnet** — no live mode exists | Pi |
| `risk_engine.py` | Conviction sizing, volatility R:R, correlation + exposure caps | Pi |
| `accounting_v2.py` | Fee-deducted PnL / equity components | Pi |
| `binance_margin.py` | Optional read-only margin telemetry (signed SAPI GET) | Pi |
| `strategies.py` | Legacy strategy library (not wired into the current short-only Brain) | Pi |
| `features.py` | Vectorized feature engineering (pandas/numpy) | Pi (worker) |
| `ml_engine.py` | XGBoost incremental training + inference | Pi (worker) |
| `rl_agent.py` | SAC actor NumPy inference (40 µs forward pass) | Pi (worker) |
| `db.py` | WAL-mode SQLite: candles/positions/trades/equity/RL + **orders journal** | Pi |
| `config.py` | All tunable constants + env overrides + `~/.config/quant-bot/env` loader | Pi |
| `dashboard.py` | FastAPI + SSE UI (optional `DASHBOARD_AUTH_TOKEN`) | Pi (systemd unit) |
| `offline_trainer.py` | Optuna + full XGBoost + SAC training | **Laptop** |
| `tests/` | pytest suite — characterization + protection tests (sandboxed DB) | dev |

---

## Pi 5 First-Time Setup

### 1. Flash OS
- Flash **Raspberry Pi OS Lite (64-bit)** — no desktop, no X11
- Enable SSH in raspi-config before first boot

### 2. Mount USB SSD
```bash
sudo mkdir -p /home/pi/trading_data
# Find your drive: lsblk
sudo mount /dev/sda1 /home/pi/trading_data
# Add to /etc/fstab for persistence
echo '/dev/sda1 /home/pi/trading_data ext4 defaults,noatime 0 2' | sudo tee -a /etc/fstab
```

### 3. Run Setup Script
```bash
git clone <your-repo> /home/pi/quant_bot
cd /home/pi/quant_bot
sudo bash setup.sh
```

This will:
- Install Python dependencies in a venv
- Configure `tmpfs` at `/dev/shm/trading_logs` (zero SD card wear)
- Install and enable `quant-bot` and `quant-dashboard` systemd services
- Set GPU memory to 16 MB (headless)
- Configure journald log rotation

### 4. Start Services
```bash
sudo systemctl start quant-bot
sudo systemctl start quant-dashboard

# Check they're running
sudo systemctl status quant-bot
sudo systemctl status quant-dashboard

# Follow logs
journalctl -u quant-bot -f
```

### 5. Access Dashboard
Open `http://<pi-ip>:8000` in your browser.

---

## Offline Training (Run on Laptop)

The Pi only does **inference**. Heavy training runs on your laptop.

### Install offline dependencies
```bash
pip install -r requirements-offline.txt
```

### Run the full pipeline
```bash
# Fetches Binance data, runs Optuna (80 trials), trains XGBoost + SAC
python offline_trainer.py

# With RL experience from the Pi's live trading database
python offline_trainer.py --db /path/to/trading.db

# Limit Optuna trials for a quick test
python offline_trainer.py --trials 20
```

Output in `./offline_output/`:
- `xgb_model.json` — optimised XGBoost model
- `sac_actor.npz` — trained SAC actor weights
- `best_params.json` — winning hyperparameters

### Deploy artefacts to Pi
```bash
# Code + models + restart services
./deploy.sh <pi-ip> --models --restart

# Code only
./deploy.sh <pi-ip>
```

The SAC actor hot-reloads automatically — no bot restart needed.

---

## Key Configuration (`config.py`) — synced 2026-07-11

| Constant | Live value | Notes |
|----------|---------|-------|
| `STARTING_CASH` | 10,000 | Paper trading balance |
| `SYMBOLS` | 47 pairs | Large caps → gaming/NFT alts |
| `MAX_OPEN_POSITIONS` | 15 | Concurrent longs |
| `SHORT_MAX_OPEN` | 5 | Concurrent shorts |
| `_SAC_SIZE_CEILING` (bot.py) | 35% | Per-trade equity ceiling (SAC-sized) |
| `MAX_ORDER_EQUITY_FRAC` | 50% | V4 hard per-order clamp (defense in depth) |
| `STOP_LOSS_ATR_MULT` / `TAKE_PROFIT_ATR_MULT` | 6× / 12× | Long ATR distances |
| `SHORT_STOP_LOSS_ATR_MULT` / `SHORT_TAKE_PROFIT_ATR_MULT` | 3.5× / 10× | Short ATR distances |
| `MAX_DRAWDOWN_PCT` / `CB_RECOVERY_PCT` | 20% / 10% | Circuit breaker trip / re-arm |
| `MIN_ML_CONFIDENCE` | 0.80 | Entry ML gate (long: p, short: 1−p) |
| `HARD_STOP_LOSS_USD` | −15 | Survival kill-switch per position |
| `FEE_GATE_ROUND_TRIP` | 0.12% | Round-trip taker fee model |
| `EXECUTION_MODE` | paper | `paper` \| `testnet` (env) — **no live mode** |
| `STALE_ENTRY_MAX_SECS` | 180 | V4: refuse entries on stale candles |
| `WS_BACKFILL_AFTER_SECS` | 90 | V4: REST backfill after WS outage |
| `PROCESS_POOL_WORKERS` | 2 | Worker processes (thermal limit) |
| `CHECK_EVERY_SECS` | 30 | Exit monitor scan interval |

Strategy-lifecycle note: `config.py`'s `TRIAL_*` / `KILL_THRESHOLD` /
`MUTATION_*` constants belong to the legacy `strategies.py` system, which the
current short-only `brain.py` does **not** use. `strategy_engine_gen2.py`
carries its own separate thresholds and is also not wired into `bot.py`.
The live signal path is: `brain.get_ensemble_signal` → ML gate → SAC sizing.

## Execution modes & V4 order protections

- `EXECUTION_MODE=paper` (default): simulated fills, identical maths to the
  historical behaviour — pinned by `tests/test_executor.py`.
- `EXECUTION_MODE=testnet`: Binance Spot testnet. Requires
  `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` (put them in
  `~/.config/quant-bot/env`, chmod 600). There is deliberately no mainnet mode.
- Protections active in every mode:
  - **Duplicate orders**: deterministic client-order-id per
    (symbol, action, candle) claimed in the `orders` journal before money
    moves; retries and duplicate signals collapse onto one order.
  - **Stale data**: entries refused when the newest candle is older than
    `STALE_ENTRY_MAX_SECS`; exits never gated.
  - **Oversized orders**: `MAX_ORDER_EQUITY_FRAC` clamp at the executor.
  - **Restarts**: boot-time journal reconciliation + cash-invariant report.
  - **Network loss**: WS exponential backoff + full REST backfill after
    outages > `WS_BACKFILL_AFTER_SECS`; entry freeze via the stale gate.
  - **Partial fills / retries / rate limits** (testnet): actual executedQty
    booked, query-before-retry idempotency, weight-based token bucket.

## Secrets

Never hardcode credentials. `config.py` loads `~/.config/quant-bot/env`
(KEY=VALUE, chmod 600) at import; real env vars win. Known keys:
`DISCORD_WEBHOOK_URL`, `DASHBOARD_AUTH_TOKEN`, `BINANCE_API_KEY`,
`BINANCE_API_SECRET`, `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`.
A pre-commit hook (`scripts/pre-commit`, installed in `.git/hooks/`) blocks
staged lines that look like credentials.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
Runs against a throwaway temp-dir database — never the live one.

---

## Hardware Optimisations

| Problem | Solution |
|---------|---------|
| SD card wear | Logs → `/dev/shm/trading_logs` (tmpfs RAM) |
| Python GIL | CPU-bound ML offloaded to `ProcessPoolExecutor` |
| Thermal throttling | `nthread=2` XGBoost, offline training on laptop |
| Memory pressure | SQLite `mmap_size=30GB`, WAL mode, 64 MB page cache |
| Display overhead | Headless OS, `gpu_mem=16` |

---

## Trading Logic Flow

```
Closed 1m candle
       │
       ├─► [Worker] XGBoost.predict_proba(features) → ml_prob
       │
       └─► Brain.get_ensemble_signal()
                │
                ├─► Regime detection (ADX + realised vol)
                ├─► Softmax allocation (Sharpe-weighted + regime weights)
                ├─► Run all strategy functions
                ├─► ML gate: buy only if ml_prob ≥ 0.60
                └─► Signal: buy / sell / none
                           │
                           └─► [Worker] SAC.forward(state) → position_fraction
                                          │
                                          └─► Execute trade (slippage, ATR stops)
                                                     │
                                                     └─► brain.reward() → update
                                                         scores / regime weights /
                                                         replay buffer / mutations
```

---

## Monitoring

```bash
# Bot logs (live)
journalctl -u quant-bot -f

# Dashboard logs
journalctl -u quant-dashboard -f

# Database quick check
sqlite3 ~/trading_data/trading.db "SELECT * FROM positions WHERE shares > 0;"
sqlite3 ~/trading_data/trading.db "SELECT COUNT(*), SUM(pnl) FROM trades WHERE status='filled';"

# CPU temperature
vcgencmd measure_temp

# Memory usage
free -h
```

---

## Strategy Lifecycle

1. **Core strategies** (RSI_EMA, BOLLINGER, MACD, MOMENTUM, MEAN_REVERSION, YOLO_FIRE) — start with `generation=0`, mutated if performance degrades
2. **Generated strategies** — randomly assembled from building blocks every 50 trades, run a 15-trade trial
3. **Graduation** — pass `win_rate ≥ 42%` AND `total_pnl ≥ -$5` → becomes permanent
4. **Retirement** — fail trial or drop below KILL_THRESHOLD → archived to graveyard
5. **Mutation** — noisy Gaussian perturbation of numeric params, 30% probability per param
