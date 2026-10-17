#!/bin/bash
# Stop Bitcoin (BTCUSD) 24/7 Auto-Trader via systemd user service
systemctl --user stop btc-auto-trader.service
pkill -f "run_auto_trader.*BTCUSD" 2>/dev/null || true

echo "🛑 Bitcoin (BTCUSD) 24/7 Auto-Trader stopped."
