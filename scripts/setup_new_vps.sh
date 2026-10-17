#!/bin/bash
# ==============================================================================
# 🚀 New VPS One-Click Restoration Script for Gold & Bitcoin Trading System
# Run this script on the destination VPS after extracting the backup archive.
# ==============================================================================
set -e

echo "============================================================"
echo "🚀 Restoring Gold & Bitcoin Autonomous Trading System"
echo "============================================================"

# 1. Check user and directory paths
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# 2. Install required system packages
echo "📦 Installing Linux dependencies (xvfb, python3, tmux, fonts)..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y xvfb python3 python3-pip tmux curl libglib2.0-0 libx11-6 libfreetype6 libfontconfig1 rsync
fi

# 3. Enable linger so user services run 24/7 in background without an active SSH session
echo "🛡️ Enabling systemd user linger for $CURRENT_USER..."
sudo loginctl enable-linger "$CURRENT_USER" || true

# 4. Install host Python dependencies
echo "🐍 Installing Python package dependencies..."
pip3 install --upgrade pip --break-system-packages 2>/dev/null || true
pip3 install -r /home/mcpuser/MCP/gold-trading-system/requirements.txt --break-system-packages 2>/dev/null || pip3 install -r /home/mcpuser/MCP/gold-trading-system/requirements.txt

# 5. Fix permissions on executables
chmod +x /home/mcpuser/bin/* 2>/dev/null || true
chmod +x /home/mcpuser/MCP/gold-trading-system/scripts/*.sh 2>/dev/null || true

# 6. Reload and start systemd user units
echo "⚙️ Reloading systemd user daemon and starting services..."
systemctl --user daemon-reload
systemctl --user enable mt5-bridge.service gold-auto-trader.service btc-auto-trader.service
systemctl --user restart mt5-bridge.service
sleep 5
systemctl --user restart gold-auto-trader.service btc-auto-trader.service

echo "============================================================"
echo "✅ Restoration Complete! Checking service health:"
echo "============================================================"
systemctl --user status mt5-bridge.service gold-auto-trader.service btc-auto-trader.service --no-pager
