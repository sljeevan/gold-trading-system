#!/bin/bash
systemctl --user stop gold-auto-trader.service 2>/dev/null || true
tmux kill-session -t gold-auto-trader 2>/dev/null || true
pkill -f "run_auto_trader.*(XAUUSD|DONCHIAN|GOLD)" 2>/dev/null || true
echo "🛑 Gold 24/7 Auto-Trader stopped."
