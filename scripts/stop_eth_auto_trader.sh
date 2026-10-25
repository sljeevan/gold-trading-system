#!/bin/bash
# Stop Ethereum (ETHUSD) 24/7 Auto-Trader via systemd user service
systemctl --user stop eth-auto-trader.service
pkill -f "run_auto_trader.*ETHUSD" 2>/dev/null || true

echo "🛑 Ethereum (ETHUSD) 24/7 Auto-Trader stopped."
