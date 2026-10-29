#!/bin/bash
# ==============================================================================
# Aegis Alpha — Profile A: Bitcoin (BTCUSD) Autonomous Crypto Yield Engine
# Setup: Fixed Risk $75 | Tight TSL $200.00 | ML >= 0.48 | Max 1 Position | Tag: YieldA
# Completely isolated from existing production daemons (independent DB & logs)
# ==============================================================================
export PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src
export HOME=/home/mcpuser
export USER=mcpuser
export PATH=/home/mcpuser/bin:/usr/local/bin:/usr/bin:/bin

PYTHON_BIN=/home/mcpuser/MCP/gold-trading-system/.venv/bin/python

echo "[YieldA-BTC] Checking MT5 Bridge status on 127.0.0.1:8889 (Monthly_winsaw)..."
for i in {1..30}; do
    if $PYTHON_BIN -c "import socket; s = socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1', 8889)); s.close()" 2>/dev/null; then
        echo "[YieldA-BTC] MT5 Bridge 2 (Monthly_winsaw) is online and accepting connections!"
        break
    fi
    echo "[YieldA-BTC] Waiting for MT5 Bridge 2 to start (attempt $i/30)..."
    sleep 1
done

exec $PYTHON_BIN -u -m gold_trader.scripts.run_auto_trader \
    --bridge-port 8889 \
    --symbol BTCUSD \
    --mode FLAT_RISK_TOUCH \
    --flat-risk \
    --fixed-risk-usd 75.00 \
    --timeframe 1h \
    --min-rr 1.25 \
    --risk-pct 15.0 \
    --interval-seconds 60 \
    --one-trade-per-bar \
    --loss-cooldown-minutes 60 \
    --max-daily-trades 6 \
    --no-session-filter \
    --runner-trail-buffer 200.00 \
    --runner-trail-buffer-r 0.50 \
    --max-positions 1 \
    --order-tag YieldA \
    --runner-target-2-multiplier 3.00 \
    --enable-ml-filter \
    --ml-min-prob 0.50 \
    --require-slope-alignment \
    --db-path /home/mcpuser/.gold_trading_data/yield_a_btc_auto_trader.db \
    --loop 2>&1 | tee -a /home/mcpuser/MCP/gold-trading-system/yield_a_btc_auto_trader.log
