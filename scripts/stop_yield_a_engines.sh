#!/bin/bash
# ==============================================================================
# Stop Profile A Autonomous Crypto Yield Engines (BTCUSD & ETHUSD)
# Does NOT touch existing production models (gold/btc/eth auto-trader)
# ==============================================================================
echo "🛑 Stopping Profile A Autonomous Crypto Yield Engines..."
systemctl --user stop yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service
systemctl --user disable yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service

echo ""
echo "Current Service Status:"
systemctl --user status yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service --no-pager
