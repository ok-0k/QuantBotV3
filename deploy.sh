#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Push code and/or trained models to the Raspberry Pi
#
# Usage:
#   ./deploy.sh <pi-ip>             # push code only
#   ./deploy.sh <pi-ip> --models    # push code + trained model artefacts
#   ./deploy.sh <pi-ip> --restart   # push code + restart services
#
# Prerequisites: SSH key installed on the Pi (ssh-copy-id pi@<ip>)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PI_IP="${1:-}"
PUSH_MODELS=false
RESTART_SERVICES=false

for arg in "${@:2}"; do
    [[ "$arg" == "--models"  ]] && PUSH_MODELS=true
    [[ "$arg" == "--restart" ]] && RESTART_SERVICES=true
done

[[ -z "$PI_IP" ]] && { echo "Usage: ./deploy.sh <pi-ip> [--models] [--restart]"; exit 1; }

PI_USER="${PI_USER:-pi}"
BOT_DIR="${BOT_DIR:-/home/${PI_USER}/quant_bot}"
DATA_DIR="${DATA_DIR:-/home/${PI_USER}/trading_data}"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${BLUE}[deploy]${NC} $*"; }
ok()    { echo -e "${GREEN}[done]${NC}  $*"; }

# ── Push source code ──────────────────────────────────────────────────────────
info "Pushing source code → ${PI_USER}@${PI_IP}:${BOT_DIR}/"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'offline_output' \
    bot.py brain.py config.py dashboard.py db.py \
    features.py ml_engine.py rl_agent.py strategies.py \
    requirements.txt \
    "${PI_USER}@${PI_IP}:${BOT_DIR}/"

ok "Source code deployed"

# ── Push trained model artefacts ──────────────────────────────────────────────
if [[ "$PUSH_MODELS" == true ]]; then
    ARTEFACTS=("offline_output/xgb_model.json" "offline_output/sac_actor.npz" "offline_output/best_params.json")
    FOUND=false
    for f in "${ARTEFACTS[@]}"; do
        [[ -f "$f" ]] && FOUND=true && break
    done

    if [[ "$FOUND" == true ]]; then
        info "Pushing trained model artefacts → ${PI_USER}@${PI_IP}:${DATA_DIR}/"
        rsync -avz --progress \
            offline_output/xgb_model.json \
            offline_output/sac_actor.npz \
            offline_output/best_params.json \
            "${PI_USER}@${PI_IP}:${DATA_DIR}/" 2>/dev/null || true
        ok "Model artefacts deployed"
    else
        echo "  [warn] No artefacts found in ./offline_output/ — run offline_trainer.py first"
    fi
fi

# ── Restart services ──────────────────────────────────────────────────────────
if [[ "$RESTART_SERVICES" == true ]]; then
    info "Restarting services on Pi..."
    ssh "${PI_USER}@${PI_IP}" "sudo systemctl restart quant-bot quant-dashboard"
    ok "Services restarted"
fi

echo ""
echo -e "${GREEN}Deploy complete.${NC}"
echo "  Logs    : ssh ${PI_USER}@${PI_IP} 'journalctl -u quant-bot -f'"
echo "  Dashboard: http://${PI_IP}:8000"
