#!/bin/bash
# Start 24/7 Gold Auto-Trader Daemon in tmux session
tmux kill-session -t gold-auto-trader 2>/dev/null || true
sleep 1

INTERVAL=${1:-60}
MIN_RR=${2:-1.25}
RISK_PCT=${3:-10.0}
TIMEFRAME=${4:-"1h"}
MODE=${5:-"COMPOUNDING_TOUCH"}

EXTRA_ARGS="${*:6}"

tmux new-session -d -s gold-auto-trader "PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src python3 -u -m gold_trader.scripts.run_auto_trader --loop --interval-seconds $INTERVAL --min-rr $MIN_RR --risk-pct $RISK_PCT --timeframe $TIMEFRAME --mode $MODE $EXTRA_ARGS 2>&1 | tee /home/mcpuser/MCP/gold-trading-system/auto_trader.log"

echo "✅ Gold 24/7 Auto-Trader launched in tmux session 'gold-auto-trader' (Mode: ${MODE}, Timeframe: ${TIMEFRAME}, Interval: ${INTERVAL}s, Min RR: ${MIN_RR}R)"
sleep 2
tmux capture-pane -t gold-auto-trader -p
