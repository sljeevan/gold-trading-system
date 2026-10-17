#!/bin/bash
# Start Bitcoin (BTCUSD) 24/7 Auto-Trader via systemd user service
systemctl --user daemon-reload
systemctl --user enable btc-auto-trader.service
systemctl --user restart btc-auto-trader.service

echo "✅ Bitcoin (BTCUSD) 24/7 Auto-Trader started via systemd user unit 'btc-auto-trader.service'"
sleep 2
systemctl --user status btc-auto-trader.service --no-pager
