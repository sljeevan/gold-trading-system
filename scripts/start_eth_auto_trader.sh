#!/bin/bash
# Start Ethereum (ETHUSD) 24/7 Auto-Trader via systemd user service
systemctl --user daemon-reload
systemctl --user enable eth-auto-trader.service
systemctl --user restart eth-auto-trader.service

echo "✅ Ethereum (ETHUSD) 24/7 Auto-Trader started via systemd user unit 'eth-auto-trader.service'"
sleep 2
systemctl --user status eth-auto-trader.service --no-pager
