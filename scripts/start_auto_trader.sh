#!/bin/bash
# Start Gold (XAUUSD) 24/7 Auto-Trader via systemd user service
systemctl --user daemon-reload
systemctl --user enable gold-auto-trader.service
systemctl --user restart gold-auto-trader.service

echo "✅ Gold (XAUUSD) 24/7 Auto-Trader started via systemd user unit 'gold-auto-trader.service'"
sleep 2
systemctl --user status gold-auto-trader.service --no-pager
