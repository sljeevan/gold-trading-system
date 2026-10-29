#!/bin/bash
export PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src
export HOME=/home/mcpuser
export USER=mcpuser
export PATH=/home/mcpuser/bin:/usr/local/bin:/usr/bin:/bin

PYTHON_BIN=/home/mcpuser/MCP/gold-trading-system/.venv/bin/python

echo "[AutoTrader] Checking MT5 Bridge status on 127.0.0.1:8888..."
for i in {1..30}; do
    if $PYTHON_BIN -c "import socket; s = socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1', 8888)); s.close()" 2>/dev/null; then
        echo "[AutoTrader] MT5 Bridge is online and accepting connections!"
        break
    fi
    echo "[AutoTrader] Waiting for MT5 Bridge to start (attempt $i/30)..."
    sleep 1
done

exec $PYTHON_BIN -u -m gold_trader.scripts.run_auto_trader \
    --loop \
    --interval-seconds 60 \
    --flat-risk \
    --fixed-risk-usd 75.44 \
    --min-rr 1.25 \
    --risk-pct 8.0 \
    --timeframe 1h \
    --mode FLAT_RISK_TOUCH \
    --runner-trail-buffer 4.50 \
    --runner-trail-buffer-r 0.50 \
    --max-positions 0 \
    --runner-target-2-multiplier 3.00 \
    --enable-ml-filter \
    --ml-min-prob 0.50 \
    --require-slope-alignment \
    --one-trade-per-bar \
    --loss-cooldown-minutes 60 \
    --max-daily-trades 6 \
    --session-filter 2>&1 | tee -a /home/mcpuser/MCP/gold-trading-system/auto_trader.log

