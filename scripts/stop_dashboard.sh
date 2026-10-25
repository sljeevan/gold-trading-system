#!/bin/bash
# Stop Atlas Trading Dashboard Backend & Frontend
systemctl --user stop trading-dashboard-ui.service trading-dashboard-api.service

echo "🛑 Trading Dashboard services stopped."
