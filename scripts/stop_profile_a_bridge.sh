#!/bin/bash
# Stop Profile A Dedicated MT5 Bridge Server
tmux kill-session -t mt5-bridge-profile-a 2>/dev/null || true
echo "Profile A MT5 Bridge (Port 8889) stopped."
