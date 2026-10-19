---
name: gold-compounding-system
description: >-
  Use this skill whenever working on the Gold (XAUUSD), Bitcoin (BTCUSD), or Ethereum (ETHUSD)
  Autonomous Trading System, Compounding Ladder, Flat Risk execution, XM MT5 socket bridge,
  ML meta-labeling, backtesting, or auto-trader daemon operations.
---

# Gold & Multi-Asset Autonomous Trading System Skill

Use this skill when managing, debugging, analyzing, or operating the 24/7 autonomous quantitative trading engines for **Spot Gold (XAU/USD)**, **Bitcoin (BTC/USD)**, and **Ethereum (ETH/USD)**.

---

## 1. Quick Context & System Architecture

- **Project Root**: `/home/mcpuser/MCP/gold-trading-system`
- **Data & Telemetry Storage**: `~/.gold_trading_data/`
- **MT5 Socket Bridge**: `127.0.0.1:8888` (Wine 64-bit + XM MT5 Terminal on Xvfb display `:99`)
- **Active Production Execution Mode**: **`FLAT_RISK_TOUCH`** (Fixed Risk: **$75.44** on $1,000 baseline, target gain: **+$113.16 / 1.5R**)
- **State & Databases**:
  - Gold: `~/.gold_trading_data/compounding_ladder_state.json` & `~/.gold_trading_data/auto_trader.db`
  - Bitcoin: `~/.gold_trading_data/btc_compounding_ladder_state.json` & `~/.gold_trading_data/btc_auto_trader.db`
  - Ethereum: `~/.gold_trading_data/eth_compounding_ladder_state.json` & `~/.gold_trading_data/eth_auto_trader.db`
- **ML Barrier Models**:
  - Gold: `~/.gold_trading_data/models/gold_ml_barrier.joblib`
  - Bitcoin: `~/.gold_trading_data/models/btc_ml_barrier.joblib`

---

## 2. Multi-Asset Microstructure & Execution Parameters

| Parameter | Gold (XAUUSD) | Bitcoin (BTCUSD) | Ethereum (ETHUSD) | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Contract Size** | `100.0` (1 lot = 100 oz) | `1.0` (1 lot = 1 BTC) | `1.0` (1 lot = 1 ETH) | Broker contract specifications on XM MT5 |
| **Minimum Stop Loss** | `$8.00` (80 pips) | `$500.00` | `$20.00` | Enforces dynamic ATR clearance above noise |
| **Break-Even Buffer** | `+$0.50` | `+$30.00` | `+$2.00` | Eliminates broker spread whip-outs on runners |
| **Runner Trailing Buffer**| `$4.50` | `$350.00` | `$10.00` | Trailing buffer behind peak excursion |
| **Trading Schedule** | Sun 22:05 – Fri 21:00 UTC | 24/7/365 (No Weekend Filter) | 24/7/365 (Crypto) | Gold closes on weekends; crypto trades 24/7 |
| **Session Filter** | 07:00 – 19:00 UTC | Disabled (`--no-session-filter`) | Enabled or Disabled | Avoids Asian session chop on Gold |

---

## 3. Core Operational Procedures

### 3.1 Verify MT5 Bridge & Account Health
```bash
python3 -c "
from gold_trader.brokers.socket_broker import SocketBroker
broker = SocketBroker()
print('Bridge Online:', broker.connect())
print('Account Info:', broker.get_account_info())
print('Open Positions:', broker.get_open_positions())
"
```

### 3.2 Inspect Auto-Trader Daemons & Services
```bash
# Check all running daemons
systemctl --user status gold-auto-trader.service btc-auto-trader.service --no-pager
ps aux | grep -E "python.*auto_trader" | grep -v grep

# Restart services
systemctl --user restart gold-auto-trader.service btc-auto-trader.service

# Tail live execution logs
tail -f /home/mcpuser/MCP/gold-trading-system/auto_trader.log
tail -f /home/mcpuser/MCP/gold-trading-system/btc_auto_trader.log
tail -f /home/mcpuser/MCP/gold-trading-system/eth_auto_trader.log
```

### 3.3 Test Strategy Setups & ML Predictions
```bash
PYTHONPATH=src python3 -c "
from gold_trader.core.strategy_engine import generate_compounding_ladder_setup
from gold_trader.core.compounding_ladder import CompoundingLadderManager

for sym in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    state_file = f'/home/mcpuser/.gold_trading_data/{sym[:3].lower()}_compounding_ladder_state.json'
    ladder = CompoundingLadderManager(state_file=state_file)
    setup = generate_compounding_ladder_setup(
        symbol=sym,
        timeframe='1h',
        ladder_manager=ladder,
        min_win_probability=0.44,
        strategy_mode='HIGH_VELOCITY_TOUCH_1H',
        donchian_period=10,
        rr_multiplier=1.25,
        trend_filter_ema=50,
        trigger_mode='TOUCH'
    )
    print(f'=== {sym} Setup ===')
    print('Status:', setup.get('status'))
    print('Action:', setup.get('action'))
    print('Market Price:', setup.get('market_price'))
    print('Reason:', setup.get('reason'))
    if 'win_probability' in setup:
        print('ML Win Probability:', setup.get('win_probability'))
    print()
"
```

### 3.4 Manage Compounding Ladder & Sizing Modes
- **Inspect Ladder State**:
  ```bash
  PYTHONPATH=src python3 -c "from gold_trader.core.compounding_ladder import CompoundingLadderManager; print(CompoundingLadderManager().get_status())"
  ```
- **Switch to Noise-Adaptive Compounding Ladder**:
  ```bash
  PYTHONPATH=src python3 -c "from gold_trader.core.compounding_ladder import CompoundingLadderManager; CompoundingLadderManager().set_archetype('NOISE_ADAPTIVE_LADDER', loss_policy='PROGRESSIVE_RATCHET')"
  ```
- **Set Flat Risk Baseline ($75.44 Risk)**:
  ```bash
  PYTHONPATH=src python3 -c "from gold_trader.core.compounding_ladder import CompoundingLadderManager; CompoundingLadderManager().set_archetype('FLAT_RISK')"
  ```

---

## 4. Key Architectural Safeguards & Rules

1. **Never use fixed tight stops ($1.50 or $2.50)**: Gold ATR is $3.50–$7.50. Fixed stops get liquidated by noise. Always enforce dynamic ATR stops ($\ge \$8.00$).
2. **ML Gatekeeper (`GoldMLPredictor`)**: Must always pass `symbol=symbol` to ensure asset-specific models are loaded. Minimum threshold is configured via `--ml-min-prob` (production baseline: **0.44**).
3. **Delayed Break-Even Triggers**: Never shift SL to break-even until trade is at least **+1.0R** in profit to prevent premature stop-outs.
4. **Order Comments & Telemetry**:
   - Flat Risk trades: `AI-FlatRisk-{symbol}`
   - Compounding trades: `AI-Compounding-L{level}`
   - Filtered/Pass setups: Must log real price, candle time, and explicit ML gate rationale (never default to $0.0).
5. **Live MT5 Broker Priority**: Always query `broker.get_rates()` first for `XAUUSD`, `BTCUSD`, and `ETHUSD` before falling back to Yahoo Finance.
6. **High-Impact News Catalyst Guard**: Enforce ±15m execution blackout around US CPI, NFP, and FOMC releases via OpenBB.
