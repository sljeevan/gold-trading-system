#!/bin/bash
export PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src
export HOME=/home/mcpuser
export USER=mcpuser
export PATH=/home/mcpuser/bin:/usr/local/bin:/usr/bin:/bin

PYTHON_BIN=/home/mcpuser/MCP/gold-trading-system/.venv/bin/python

exec $PYTHON_BIN -m uvicorn gold_trader.api.dashboard_api:app --host 127.0.0.1 --port 8787
