#!/bin/bash
export PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src
export HOME=/home/mcpuser
export USER=mcpuser
export PATH=/home/mcpuser/bin:/usr/local/bin:/usr/bin:/bin

PYTHON_BIN=/home/mcpuser/MCP/gold-trading-system/.venv/bin/python

echo "[ETHAutoTrader] Checking MT5 Bridge status on 127.0.0.1:8888..."
for i in {1..30}; do
    if $PYTHON_BIN -c "import socket; s = socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1', 8888)); s.close()" 2>/dev/null; then
        echo "[ETHAutoTrader] MT5 Bridge is online and accepting connections!"
        break
    fi
    echo "[ETHAutoTrader] Waiting for MT5 Bridge to start (attempt $i/30)..."
    sleep 1
done

exec $PYTHON_BIN -u -m gold_trader.scripts.run_auto_trader \
    --symbol ETHUSD \
    --mode FLAT_RISK_TOUCH \
    --flat-risk \
    --fixed-risk-usd 75.44 \
    --timeframe 1h \
    --min-rr 1.25 \
    --risk-pct 8.0 \
    --interval-seconds 60 \
    --one-trade-per-bar \
    --loss-cooldown-minutes 60 \
    --max-daily-trades 6 \
    --no-session-filter \
    --runner-trail-buffer 10.00 \
    --max-positions 2 \
    --max-unhedged-positions 1 \
    --allow-runner-compounding \
    --runner-target-2-multiplier 3.00 \
    --no-enable-ml-filter \
    --adx-min-threshold 22.0 \
    --require-slope-alignment \
    --db-path /home/mcpuser/.gold_trading_data/eth_auto_trader.db \
    --loop 2>&1 | tee -a /home/mcpuser/MCP/gold-trading-system/eth_auto_trader.log
