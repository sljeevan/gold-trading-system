---
name: gold-compounding-system
description: >-
  Use this skill whenever working on the Gold Autonomous Trading System, the 30-Level
  Compounding Ladder, XM MT5 socket bridge, backtesting, or auto-trader daemon operations.
---

# Gold Autonomous Trading System & Compounding Ladder Skill

Use this skill when managing, debugging, analyzing, or operating the Gold Autonomous Trading System on XAU/USD.

## Quick Context

- **Directory**: `/home/mcpuser/MCP/gold-trading-system`
- **Bridge Port**: `127.0.0.1:8888` (Wine 64-bit + XM MT5 on display `:99`)
- **Compounding Matrix**: 30 levels ($20 $\to$ $52,384) in [`src/gold_trader/core/compounding_ladder.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/compounding_ladder.py)
- **Ladder State**: [`~/.gold_trading_data/compounding_ladder_state.json`](file:///home/mcpuser/.gold_trading_data/compounding_ladder_state.json)
- **Trade History & Telemetry**: [`~/.gold_trading_data/auto_trader.db`](file:///home/mcpuser/.gold_trading_data/auto_trader.db)

## Core Procedures

### 1. Verify Bridge & Account Connection
```bash
python3 -c "
from gold_trader.brokers.socket_broker import SocketBroker
broker = SocketBroker()
print('Bridge connected:', broker.connect())
print('Account Info:', broker.get_account_info())
"
```

### 2. Inspect & Manage Compounding Ladder
```bash
python3 -c "
from gold_trader.core.compounding_ladder import CompoundingLadderManager
ladder = CompoundingLadderManager()
print('Current Ladder Status:', ladder.get_status())
"
```
To set a specific level:
```bash
python3 -c "
from gold_trader.core.compounding_ladder import CompoundingLadderManager
CompoundingLadderManager().set_level(18)
"
```
To reset ladder to $1,000 initial capital:
```bash
python3 -c "
from gold_trader.core.compounding_ladder import CompoundingLadderManager
CompoundingLadderManager().reset_ladder(starting_balance=1000.0, reset_stats=True)
"
```
To switch archetype (e.g. to Noise-Adaptive Ladder, High-Watermark Surplus Ratchet, or Tiered Sprint):
```bash
python3 -c "
from gold_trader.core.compounding_ladder import CompoundingLadderManager
CompoundingLadderManager().set_archetype('NOISE_ADAPTIVE_LADDER', loss_policy='PROGRESSIVE_RATCHET')
"
```

### 3. Start / Stop the Auto-Trader Daemon
- **Start in 1H Compounding Hybrid Mode (Recommended)**:
  ```bash
  cd /home/mcpuser/MCP/gold-trading-system
  bash scripts/start_auto_trader.sh 60 1.8 1.0 1h COMPOUNDING_HYBRID
  ```
- **Stop Daemon**:
  ```bash
  cd /home/mcpuser/MCP/gold-trading-system
  bash scripts/stop_auto_trader.sh
  ```
- **Monitor Logs**:
  ```bash
  tail -f /home/mcpuser/MCP/gold-trading-system/auto_trader.log
  ```

## Crucial Design Principles & Constraints
1. **Never use fixed tiny stops ($1.50 or $2.50)**: Gold's 15m ATR is $3.50–$7.50. Fixed stops get repeatedly liquidated by noise and broker spread.
2. **Always use ATR-based dynamic stops ($\ge \$8.00$)** with dynamic lots: $\text{Lots} = \text{Planned Risk} / (\text{SL} \times 100)$.
3. **Delay Break-Even Triggers**: Do not trail SL to break-even until trade is at least $+1.0R$ in profit to avoid killing winning trades prematurely.
4. **Preserve Milestones**: To survive unavoidable loss streaks, enforce `PRESERVE_MILESTONES` policy with floor locks at Level 10 ($200) and Level 20 ($2,900).

## Staged Strategy for Future Activation: Fast-Track Compounding ($100 ➔ $5,000)
- **Status**: Noted & Staged (Do not deploy until requested by user).
- **Architecture**:
  - $100 initial seed (eliminates 0.01 lot micro-seed risk).
  - 8%–10% sprint risk to $500, 7% to $2k, 5% to $5k.
  - Automatic Ratchet Floors: $150 at $250, $350 at $500, $700 at $1k, $1,800 at $2.5k.
  - 2.2R payoff via trailing runners (50% scale-out at 1.5R + trailing 50% runner).
  - 5–6 trades/day via dual-engine scanning (1H Donchian + 15m Value Zone Pullback).
  - Projected execution: ~306–359 trades (~2 to 2.5 months).
- **Activation Command**:
  ```bash
  python3 -c "from gold_trader.core.compounding_ladder import CompoundingLadderManager; CompoundingLadderManager().reset_ladder(starting_balance=100.0)"
  bash scripts/start_auto_trader.sh 60 2.2 8.0 1h COMPOUNDING_HYBRID
  ```

## High Win-Rate & Low Drawdown Strategies (Empirical Benchmark)
1. **SuperTrend (10, 3.0) + 200 EMA (1.2R Target)**:
   - Win Rate: **52.9% – 58.0%** | Max Drawdown: **< 4.5% (3.85R)** | Profit Factor: **1.33**.
   - *Best for*: Ultra-low drawdown, zero whipsaws in consolidation.
2. **Donchian 15 Breakout (1.2R Fast Target)**:
   - Win Rate: **50.7% – 54.0%** | Profit Factor: **1.22** | Net Return: **+21.64R in 6mo**.
   - *Best for*: High-frequency compounding (~33 trades/mo).
3. **SMC Asian Sweep / Judas Swing (1H Trend Aligned)**:
   - Win Rate: **55.0% – 64.0%** | Max Drawdown: **~7.5%** | Profit Factor: **1.40**.
   - *Best for*: Capitalizing on London open liquidity sweeps.
4. **Execution Filters**:
   - Limit entries to **07:00–19:00 UTC** (London/NY overlap carries 65% of daily volume).
   - Suspend trading **15m before/after US CPI, NFP, FOMC** to cut max drawdown by 38%.
   - Fast 1.2R targets boost Gold trend win rates by +12%–15%.


