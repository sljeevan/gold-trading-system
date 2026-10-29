#!/bin/bash
# Start Profile A (Model 2) Dedicated MT5 Bridge Server on TCP port 8889
set -e

PROJECT_ROOT=/home/mcpuser/MCP/gold-trading-system
LOG_FILE="$PROJECT_ROOT/bridge_profile_a.log"

tmux kill-session -t mt5-bridge-profile-a 2>/dev/null || true
sleep 1

echo "Starting Profile A XM MT5 Bridge (Port 8889, 'Monthly_winsaw') in tmux session 'mt5-bridge-profile-a'..."

tmux new-session -d -s mt5-bridge-profile-a \
    "PYTHONIOENCODING=utf-8 xvfb-run -a /home/mcpuser/bin/wine C:\\\\python64\\\\python.exe -u Z:\\\\home\\\\mcpuser\\\\MCP\\\\gold-trading-system\\\\src\\\\gold_trader\\\\bridge_server.py --port 8889 --path 'C:\\\\Program Files\\\\XM Global MT5 Profile A\\\\terminal64.exe' --prefix XM_YIELDA 2>&1 | tee $LOG_FILE"

echo "Waiting for Profile A Bridge to bind port 8889 and initialize MT5..."
for i in {1..40}; do
    if grep -q "listening on TCP port 8889" "$LOG_FILE" 2>/dev/null; then
        echo "✅ Profile A Bridge is online and connected to XMGlobal-MT5 10!"
        cat "$LOG_FILE"
        exit 0
    fi
    sleep 1
done

echo "⚠️ Bridge startup status:"
cat "$LOG_FILE"
