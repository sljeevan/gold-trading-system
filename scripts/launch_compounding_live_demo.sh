#!/bin/bash
# ==============================================================================
# Master Production Launcher: 30-Level Gold Compounding Challenge (Live Demo)
# Timeframe: 15-Minute (15m) | Policy: ONE_STEP_LADDER ($20 -> $52,384)
# ==============================================================================
set -e

PROJECT_DIR="/home/mcpuser/MCP/gold-trading-system"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "🚀 PREPARING 30-LEVEL GOLD COMPOUNDING LIVE DEMO SETUP (15M TIMEFRAME)"
echo "======================================================================"

# 1. Verify / Connect MT5 Socket Bridge
echo "[1/4] Checking XM MetaTrader 5 Bridge connectivity..."
BRIDGE_RUNNING=$(python3 -c "
from gold_trader.brokers.socket_broker import SocketBroker
b = SocketBroker()
print(b.connect())
" 2>/dev/null || echo "False")

if [ "$BRIDGE_RUNNING" != "True" ]; then
    echo "⚠️  MT5 Bridge is not running. Starting bridge via scripts/start_bridge.sh..."
    bash "$PROJECT_DIR/scripts/start_bridge.sh"
    sleep 3
else
    echo "✅ XM MT5 Bridge is ACTIVE and CONNECTED!"
fi

# 2. Verify Account Info & Balance
echo "[2/4] Fetching XM MT5 Account Details..."
python3 -c "
from gold_trader.brokers.socket_broker import SocketBroker
b = SocketBroker()
info = b.get_account_info()
print(f'   Broker: {info.get(\"broker\")} | Account: #{info.get(\"account_id\")}')
print(f'   Balance: \${float(info.get(\"balance\", 0)):,.2f} {info.get(\"currency\")} | Leverage: 1:{info.get(\"leverage\")}')
"

# 3. Confirm Compounding Ladder State (Level 1 / $20.00 / ONE_STEP_LADDER)
echo "[3/4] Initializing Compounding Ladder Matrix to Stage 1..."
python3 -c "
from gold_trader.core.compounding_ladder import CompoundingLadderManager
ladder = CompoundingLadderManager(loss_policy='ONE_STEP_LADDER')
# Ensure clean initial state if starting fresh
st = ladder.get_status()
print(f'   Active Level: Level {st[\"current_level\"]} (Target Lots: {st[\"next_trade_specs\"][\"target_lot_size\"]} Lots)')
print(f'   Allocated Capital: \${st[\"current_balance\"]:,.2f} | Policy: {st[\"loss_policy\"]}')
print(f'   Target TP: +{st[\"next_trade_specs\"][\"tp_pips\"]} Pips (+\${st[\"next_trade_specs\"][\"tp_points_usd\"]:,.2f}/oz) -> +\${st[\"next_trade_specs\"][\"expected_gain_usd\"]:,.2f}')
print(f'   Target SL: -{st[\"next_trade_specs\"][\"sl_pips\"]} Pips (-\${st[\"next_trade_specs\"][\"sl_points_usd\"]:,.2f}/oz) -> -\${st[\"next_trade_specs\"][\"expected_risk_usd\"]:,.2f}')
print(f'   Breakeven Lock: +{st[\"next_trade_specs\"][\"breakeven_lock_pips\"]} Pips (+\${st[\"next_trade_specs\"][\"breakeven_lock_usd\"]:,.2f}/oz)')
print(f'   Limit Pullback Discount: {st[\"next_trade_specs\"][\"pullback_limit_discount_pips\"]} Pips (\${st[\"next_trade_specs\"][\"pullback_limit_discount_usd\"]:,.2f}/oz)')
"

# 4. Launch Autonomous Auto-Trader in tmux
echo "[4/4] Launching 24/7 Autonomous Auto-Trader in tmux session 'gold-auto-trader'..."
bash "$PROJECT_DIR/scripts/start_auto_trader.sh" 60 1.3 1.0 15m COMPOUNDING_LADDER

echo ""
echo "======================================================================"
echo "🎯 LIVE SETUP ARMED & READY FOR TOMORROW'S MARKET SESSION!"
echo "======================================================================"
echo "  • Timeframe: 15-Minute (15m Candles)"
echo "  • Strategy: EMA 9/21 Value Zone Pullback + 1H Trend Confluence"
echo "  • Entry: 8-Pip Limit Wholesale Order (Market - \$0.80 BUY / +\$0.80 SELL)"
echo "  • Protection: +10-Pip (+\$1.00) Breakeven Lock to Entry + \$0.35"
echo "  • Active Level: Level 1 (0.03 Lots / \$20.00 Base Capital)"
echo "  • Monitoring Loop: Every 60 seconds (London & NY: 07:00–19:00 UTC)"
echo ""
echo "Helpful Commands:"
echo "  View live scan stream:   tail -f $PROJECT_DIR/auto_trader.log"
echo "  Check ladder status:     python3 -c \"from gold_trader.core.compounding_ladder import CompoundingLadderManager; print(CompoundingLadderManager().get_status())\""
echo "  Stop the auto trader:    bash $PROJECT_DIR/scripts/stop_auto_trader.sh"
echo "======================================================================"
