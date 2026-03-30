# ⚡ Quant Bot v3 — Raspberry Pi 5 Edge Trading System

Self-improving algorithmic trading bot: XGBoost signal filtering + SAC reinforcement learning position sizing + genetic strategy evolution. Runs 24/7 on a headless Raspberry Pi 5.

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
| `bot.py` | Async WebSocket orchestrator | Pi (main) |
| `brain.py` | Strategy ensemble + regime + circuit breaker | Pi |
| `strategies.py` | 6 core strategies + generated strategy engine | Pi |
| `features.py` | Vectorized feature engineering (pandas/numpy) | Pi (worker) |
| `ml_engine.py` | XGBoost incremental training + inference | Pi (worker) |
| `rl_agent.py` | SAC actor NumPy inference (40 µs forward pass) | Pi (worker) |
| `db.py` | WAL-mode SQLite with all PRAGMAs | Pi |
| `config.py` | All tunable constants | Pi |
| `dashboard.py` | FastAPI + HTMX + SSE real-time UI | Pi (separate process) |
| `offline_trainer.py` | Optuna + full XGBoost + SAC training | **Laptop** |

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

## Key Configuration (`config.py`)

| Constant | Default | Notes |
|----------|---------|-------|
| `STARTING_CASH` | 10,000 | Paper trading balance |
| `SYMBOLS` | 5 pairs | BTC ETH SOL BNB XRP |
| `TRADE_SIZE_PCT` | 8% | Base allocation per trade |
| `YOLO_TRADE_SIZE_PCT` | 12% | YOLO_FIRE starting size |
| `MAX_TRADE_SIZE_PCT` | 25% | Hard ceiling |
| `MAX_OPEN_POSITIONS` | 4 | Concurrent positions |
| `STOP_LOSS_ATR_MULT` | 4.5× | ATR-based stop distance |
| `TAKE_PROFIT_ATR_MULT` | 8.0× | ATR-based target distance |
| `MAX_DRAWDOWN_PCT` | 20% | Circuit breaker threshold |
| `ML_SIGNAL_THRESHOLD` | 0.60 | XGBoost buy-gate threshold |
| `PROCESS_POOL_WORKERS` | 2 | Worker processes (thermal limit) |
| `CHECK_EVERY_SECS` | 30 | Exit monitor scan interval |

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
