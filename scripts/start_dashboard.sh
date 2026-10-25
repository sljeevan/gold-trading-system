#!/bin/bash
# Start Atlas Trading Dashboard Backend & Frontend
systemctl --user daemon-reload
systemctl --user enable --now trading-dashboard-api.service trading-dashboard-ui.service

echo "✅ Trading Dashboard services started!"
sleep 2
systemctl --user status trading-dashboard-api.service trading-dashboard-ui.service --no-pager
