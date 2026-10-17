#!/bin/bash
# ==============================================================================
# 📦 VPS Migration Bundle Exporter for Gold & Bitcoin Autonomous Trading System
# Packages the entire Wine MT5 environment, state databases, project code,
# and systemd user services into a single transferable archive.
# ==============================================================================
set -e

BACKUP_DIR="/home/mcpuser"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="gold_trading_system_backup_${TIMESTAMP}.tar.gz"
TARGET_ARCHIVE="${BACKUP_DIR}/${ARCHIVE_NAME}"

echo "============================================================"
echo "📦 Creating Complete Trading System Migration Archive"
echo "============================================================"
echo "Target archive: $TARGET_ARCHIVE"
echo "Packaging:"
echo "  1. ~/.wine              (Wine prefix, MT5 terminal, Python 64-bit)"
echo "  2. ~/.wine-bin          (Portable Wine binary distribution)"
echo "  3. ~/bin                (Wine wrapper scripts)"
echo "  4. ~/.gold_trading_data (SQLite databases & compounding ladder states)"
echo "  5. ~/MCP/gold-trading-system (Repository code, configs & .env)"
echo "  6. ~/.config/systemd/user (Systemd services: bridge, gold, btc)"
echo "------------------------------------------------------------"

cd /home/mcpuser

tar --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    -czvf "$TARGET_ARCHIVE" \
    .wine \
    .wine-bin \
    bin \
    .gold_trading_data \
    MCP/gold-trading-system \
    .config/systemd/user/mt5-bridge.service \
    .config/systemd/user/gold-auto-trader.service \
    .config/systemd/user/btc-auto-trader.service

echo "------------------------------------------------------------"
echo "✅ Backup successfully created at: $TARGET_ARCHIVE"
echo "📦 Archive size: $(ls -lh "$TARGET_ARCHIVE" | awk '{print $5}')"
echo "============================================================"
echo "To transfer to your new VPS, run from this machine:"
echo "  scp $TARGET_ARCHIVE <new_user>@<NEW_VPS_IP>:/home/<new_user>/"
echo "============================================================"
