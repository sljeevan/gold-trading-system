#!/bin/bash
# ==============================================================================
# Start Profile A Autonomous Crypto Yield Engines (BTCUSD & ETHUSD)
# Runs as independent systemd user services connected to Bridge 2 (Port 8889)
# ==============================================================================
PROJECT_ROOT=/home/mcpuser/MCP/gold-trading-system

# 1. Ensure Bridge 2 (Monthly_winsaw) is running
if ! python3 -c "import socket; s = socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1', 8889)); s.close()" 2>/dev/null; then
    echo "Starting Profile A MT5 Bridge on port 8889..."
    bash "$PROJECT_ROOT/scripts/start_profile_a_bridge.sh"
else
    echo "✅ Profile A MT5 Bridge (Port 8889) is already running."
fi

echo "🚀 Starting Profile A Autonomous Crypto Yield Engines..."
systemctl --user enable yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service
systemctl --user restart yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service

echo ""
echo "Current Service Status:"
systemctl --user status yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service --no-pager
