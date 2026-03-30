#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-shot Raspberry Pi 5 environment bootstrap
#
# Run ONCE after a fresh Raspberry Pi OS Lite install.
# Assumes: headless (no X11/Wayland), USB SSD mounted at /home/pi/trading_data
#
# Usage:
#   chmod +x setup.sh && sudo bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Ensure running as root ─────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || error "Run as root: sudo bash setup.sh"

TRADING_USER="${SUDO_USER:-pi}"
HOME_DIR="/home/${TRADING_USER}"
DATA_DIR="${HOME_DIR}/trading_data"
BOT_DIR="${HOME_DIR}/quant_bot"

info "Setting up Quant Bot v3 for user: ${TRADING_USER}"

# ── 2. System update & dependencies ──────────────────────────────────────────
info "Updating apt packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential git curl wget \
    libatlas-base-dev libhdf5-dev libssl-dev \
    sqlite3 \
    tmpfs                          2>/dev/null || true

ok "System packages installed"

# ── 3. Disable unnecessary services (thermal relief) ─────────────────────────
info "Disabling bluetooth and WiFi power management..."
systemctl disable bluetooth.service  2>/dev/null || true
systemctl stop    bluetooth.service  2>/dev/null || true
# Disable WiFi power save (increases latency)
if command -v iwconfig &>/dev/null; then
    iwconfig wlan0 power off 2>/dev/null || true
fi

ok "Power management configured"

# ── 4. tmpfs for logs (zero SD card wear) ────────────────────────────────────
info "Configuring tmpfs for logs at /dev/shm/trading_logs..."
mkdir -p /dev/shm/trading_logs
chown "${TRADING_USER}:${TRADING_USER}" /dev/shm/trading_logs

# Persist across reboots via fstab (tmpfs auto-created at boot)
FSTAB_LINE="tmpfs /dev/shm tmpfs defaults,noatime,nosuid,size=256m 0 0"
grep -qF "noatime,nosuid,size=256m" /etc/fstab || echo "${FSTAB_LINE}" >> /etc/fstab

ok "tmpfs configured"

# ── 5. USB SSD data directory ─────────────────────────────────────────────────
info "Creating data directory at ${DATA_DIR}..."
mkdir -p "${DATA_DIR}"
chown -R "${TRADING_USER}:${TRADING_USER}" "${DATA_DIR}"
ok "Data directory ready"

# ── 6. Python virtual environment ────────────────────────────────────────────
info "Creating Python venv at ${BOT_DIR}/.venv ..."
mkdir -p "${BOT_DIR}"
chown -R "${TRADING_USER}:${TRADING_USER}" "${BOT_DIR}"

sudo -u "${TRADING_USER}" python3 -m venv "${BOT_DIR}/.venv"

info "Installing Python dependencies (this may take a few minutes on Pi)..."
sudo -u "${TRADING_USER}" "${BOT_DIR}/.venv/bin/pip" install --upgrade pip wheel -q
sudo -u "${TRADING_USER}" "${BOT_DIR}/.venv/bin/pip" install \
    aiohttp \
    websockets \
    fastapi \
    uvicorn[standard] \
    sse-starlette \
    numpy \
    pandas \
    xgboost \
    pandas-ta \
    requests \
    -q

ok "Python dependencies installed"

# ── 7. systemd services ───────────────────────────────────────────────────────
info "Installing systemd services..."

# Bot service
cat > /etc/systemd/system/quant-bot.service << EOF
[Unit]
Description=Quant Bot v3 — Trading Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${TRADING_USER}
WorkingDirectory=${BOT_DIR}
Environment="TRADING_DATA_DIR=${DATA_DIR}"
ExecStart=${BOT_DIR}/.venv/bin/python ${BOT_DIR}/bot.py
Restart=on-failure
RestartSec=15s
StandardOutput=journal
StandardError=journal
# Thermal: limit CPU priority so the OS stays responsive
Nice=5

[Install]
WantedBy=multi-user.target
EOF

# Dashboard service
cat > /etc/systemd/system/quant-dashboard.service << EOF
[Unit]
Description=Quant Bot v3 — Dashboard
After=quant-bot.service
Wants=quant-bot.service

[Service]
Type=simple
User=${TRADING_USER}
WorkingDirectory=${BOT_DIR}
Environment="TRADING_DATA_DIR=${DATA_DIR}"
ExecStart=${BOT_DIR}/.venv/bin/python ${BOT_DIR}/dashboard.py
Restart=on-failure
RestartSec=10s
StandardOutput=journal
StandardError=journal
Nice=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable quant-bot.service
systemctl enable quant-dashboard.service

ok "systemd services installed and enabled"

# ── 8. Log rotation (journal) ─────────────────────────────────────────────────
info "Configuring journald log rotation (max 50 MB)..."
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/quant-bot.conf << EOF
[Journal]
SystemMaxUse=50M
RuntimeMaxUse=50M
EOF
systemctl restart systemd-journald 2>/dev/null || true
ok "Log rotation configured"

# ── 9. GPU memory split (free RAM for ML) ────────────────────────────────────
info "Setting GPU memory split to 16 MB (headless — no display needed)..."
GPU_MEM_LINE="gpu_mem=16"
CONFIG_FILE="/boot/firmware/config.txt"
[[ -f /boot/config.txt ]] && CONFIG_FILE="/boot/config.txt"
grep -qF "gpu_mem=" "${CONFIG_FILE}" || echo "${GPU_MEM_LINE}" >> "${CONFIG_FILE}"
ok "GPU memory set to 16 MB"

# ── 10. Print summary ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Quant Bot v3 — Setup Complete! ⚡          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Bot directory : ${BOT_DIR}"
echo "  Data directory: ${DATA_DIR}"
echo "  Python venv   : ${BOT_DIR}/.venv"
echo "  Logs          : journalctl -u quant-bot -f"
echo ""
echo "  Next steps:"
echo "  1. Copy your .py files into ${BOT_DIR}"
echo "  2. Start the bot:       sudo systemctl start quant-bot"
echo "  3. Start the dashboard: sudo systemctl start quant-dashboard"
echo "  4. Dashboard URL:       http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "  To deploy from your laptop after offline training:"
echo "  rsync -av offline_output/ pi@<pi-ip>:${DATA_DIR}/"
echo "  rsync -av *.py pi@<pi-ip>:${BOT_DIR}/"
echo ""
