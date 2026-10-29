#!/bin/bash
# ==============================================================================
# Multi-Model Production Fleet Monitor
# Displays status for Model 1 (Production Baseline) & Model 2 (Profile A Yield)
# Across Dual MT5 Bridge Terminals (Port 8888 & Port 8889)
# ==============================================================================
echo "========================================================================"
echo "🛡️  AEGIS ALPHA — PRODUCTION DUAL-ACCOUNT FLEET STATUS"
echo "========================================================================"

echo ""
echo "▶ MODEL 1: Production Baseline Services (Uncapped / Port 8888)"
systemctl --user status gold-auto-trader.service btc-auto-trader.service eth-auto-trader.service --no-pager | grep -E "Loaded:|Active:"

echo ""
echo "▶ MODEL 2: Profile A Yield Engines ('Monthly_winsaw' / Port 8889)"
systemctl --user status yield-a-btc-auto-trader.service yield-a-eth-auto-trader.service --no-pager | grep -E "Loaded:|Active:"

echo ""
echo "▶ MT5 Dual Broker Terminals & Positions:"
PYTHONPATH=/home/mcpuser/MCP/gold-trading-system/src python3 -c "
from gold_trader.brokers.socket_broker import SocketBroker

# Model 1 Broker (Port 8888)
b1 = SocketBroker(port=8888)
if b1.connect():
    acc1 = b1.get_account_info()
    pos1 = b1.get_open_positions()
    print(f'  • Account 1 (Model 1): {acc1.get(\"account_id\")} | {acc1.get(\"broker\")} | Balance: \${acc1.get(\"balance\"):,.2f} | Open Pos: {len(pos1)}')
else:
    print('  [!] Bridge 1 (Port 8888) Offline')

# Model 2 Broker (Port 8889)
b2 = SocketBroker(port=8889)
if b2.connect():
    acc2 = b2.get_account_info()
    pos2 = b2.get_open_positions()
    print(f'  • Account 2 (Monthly_winsaw): {acc2.get(\"account_id\")} | {acc2.get(\"broker\")} | Balance: \${acc2.get(\"balance\"):,.2f} | Open Pos: {len(pos2)}')
else:
    print('  [!] Bridge 2 (Port 8889) Offline')
"
echo "========================================================================"
