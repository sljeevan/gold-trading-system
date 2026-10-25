#!/bin/bash
export PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src
export HOME=/home/mcpuser
export USER=mcpuser
export PATH=/home/mcpuser/bin:/usr/local/bin:/usr/bin:/bin
export TRADING_API_BASE_URL="http://127.0.0.1:8787"

STREAMLIT_BIN=/home/mcpuser/MCP/gold-trading-system/.venv/bin/streamlit

exec $STREAMLIT_BIN run /home/mcpuser/MCP/gold-trading-system/atlas_trading_dashboard/app.py \
    --server.port 5000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
