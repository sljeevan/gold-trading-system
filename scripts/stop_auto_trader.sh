#!/bin/bash
tmux kill-session -t gold-auto-trader 2>/dev/null || true
pkill -f "run_auto_trader" 2>/dev/null || true
echo "🛑 Gold 24/7 Auto-Trader stopped."
