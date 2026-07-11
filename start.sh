Regime Distribution
#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Launch all bot processes on the Raspberry Pi 5
#
# PI 5 EDGE DEPLOYMENT STEPS
# ──────────────────────────
# 1. OS preparation (headless — no X11/Wayland):
#      sudo raspi-config → System → Boot → Console (disable desktop)
#
# 2. Move data directory to USB SSD (protects SD card from wear):
#      sudo mount /dev/sda1 /mnt/ssd
#      mkdir -p /mnt/ssd/trading_data
#      export TRADING_DATA_DIR=/mnt/ssd/trading_data
#      # Add to /etc/environment so it persists across reboots
#
# 3. Create tmpfs for logs (volatile RAM — SD card writes = 0):
#      # Add to /etc/fstab:
#      tmpfs /home/pi/trading_logs tmpfs defaults,noatime,size=64m 0 0
#
# 4. Install Python dependencies:
#      pip install -r requirements.txt --break-system-packages
#
# 5. Run this script (or use systemd service below):
#      chmod +x start.sh && ./start.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# SYSTEMD SERVICE  (auto-restart on crash, auto-start on boot)
# ─────────────────────────────────────────────────────────────────────────────
# Create /etc/systemd/system/quant-bot.service:
#
# [Unit]
# Description=Quant Bot v3 Trading Engine
# After=network.target
# Wants=network-online.target
#
# [Service]
# User=pi
# WorkingDirectory=/home/pi/trading_bot
# Environment=TRADING_DATA_DIR=/mnt/ssd/trading_data
# ExecStart=/home/pi/trading_bot/start.sh
# Restart=always
# RestartSec=10
# StandardOutput=journal
# StandardError=journal
#
# [Install]
# WantedBy=multi-user.target
#
# Then: sudo systemctl enable quant-bot && sudo systemctl start quant-bot
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TRADING_DATA_DIR="${TRADING_DATA_DIR:-/home/pi/trading_data}"

mkdir -p "$TRADING_DATA_DIR"
mkdir -p /dev/shm/trading_logs

echo "═══════════════════════════════════════════════════"
echo "  ⚡ Quant Bot v3 — Raspberry Pi 5 Edge Deployment"
echo "  Data dir: $TRADING_DATA_DIR"
echo "  Logs:     /dev/shm/trading_logs (tmpfs)"
echo "═══════════════════════════════════════════════════"

cd "$SCRIPT_DIR"

# Start bot (background)
python bot.py &
BOT_PID=$!
echo "🤖 Bot started (PID $BOT_PID)"

# Wait 3s for database init
sleep 3

# Start dashboard (foreground)
echo "📊 Dashboard → http://$(hostname -I | awk '{print $1}'):8000"
/home/admin/trading_bot/.venv/bin/python dashboard.py &
DASH_PID=$!

wait $BOT_PID

# On dashboard exit, kill bot
kill $DASH_PID 2>/dev/null || true
