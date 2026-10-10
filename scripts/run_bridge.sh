#!/bin/bash
export PYTHONIOENCODING=utf-8
export HOME=/home/mcpuser
export USER=mcpuser
export PATH=/home/mcpuser/bin:/usr/local/bin:/usr/bin:/bin

exec /usr/bin/xvfb-run -a /home/mcpuser/bin/wine C:\\python64\\python.exe -u Z:\\home\\mcpuser\\MCP\\gold-trading-system\\src\\gold_trader\\bridge_server.py 2>&1 | tee -a /home/mcpuser/MCP/gold-trading-system/bridge.log
