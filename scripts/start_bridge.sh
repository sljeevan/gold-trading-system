#!/bin/bash
tmux kill-session -t mt5-bridge 2>/dev/null || true
pkill -f "bridge_server.py" 2>/dev/null || true
pkill -f "terminal64.exe" 2>/dev/null || true
sleep 1

# Launch in persistent tmux session with UTF-8 encoding and unbuffered output
tmux new-session -d -s mt5-bridge 'PYTHONIOENCODING=utf-8 xvfb-run -a /home/mcpuser/bin/wine C:\\python64\\python.exe -u Z:\\home\\mcpuser\\MCP\\gold-trading-system\\src\\gold_trader\\bridge_server.py 2>&1 | tee /home/mcpuser/MCP/gold-trading-system/bridge.log'

echo "Starting XM MT5 Bridge in tmux session 'mt5-bridge'..."
for i in {1..35}; do
    if grep -q "listening on TCP port" /home/mcpuser/MCP/gold-trading-system/bridge.log 2>/dev/null; then
        echo "✅ Bridge is ready and connected to XM!"
        cat /home/mcpuser/MCP/gold-trading-system/bridge.log
        exit 0
    fi
    sleep 1
done

cat /home/mcpuser/MCP/gold-trading-system/bridge.log
