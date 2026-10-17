# 🏆 Autonomous Gold (XAU/USD) & Bitcoin (BTC/USD) Algorithmic Trading Platform

An autonomous, institutional-grade quantitative trading platform and 24/7 background execution daemon for **Spot Gold (XAU/USD)** and **Bitcoin (BTC/USD)** on **XM MetaTrader 5 (MT5)**, running on Linux (Ubuntu/Debian via Wine 64-bit and Xvfb).

Featuring **OpenBB Platform Macro & News Catalyst Guardrails**, **Institutional Two-Stage Compounding Ladder**, **Dynamic ATR Noise Clearance**, **Telegram Real-time Telemetry**, and **Turn-Key VPS Deployment**.

---

## 🌟 Key Architecture & Capabilities

```mermaid
graph TD
    subgraph MarketBridge [Execution Bridge Layer]
        Wine["Wine 64-bit (Xvfb Virtual Display :99)"]
        MT5["XM Global MT5 Terminal"]
        Bridge["Socket Bridge Server (bridge_server.py)<br/>TCP 127.0.0.1:8888"]
        Wine --> MT5
        MT5 <--> Bridge
    end

    subgraph OpenBBLayer [Macroeconomic & News Catalyst Guard]
        OpenBB["OpenBB Platform v5.0.0"]
        FedCal["Fed / FRED Economic Calendars"]
        YieldCurve["US Treasury Yield Curve (10Y-2Y Spread)"]
        COT["CFTC Commitment of Traders (COT)"]
        DVOL["Deribit Bitcoin Volatility (DVOL)"]
        OpenBB --> FedCal & YieldCurve & COT & DVOL
    end

    subgraph CoreEngine [Core Autonomous Daemons]
        GoldBot["Gold Auto-Trader (1H Donchian Touch + 50 EMA)"]
        BTCBot["Bitcoin Auto-Trader (1H Donchian Touch + ATR $500)"]
        Ladder["Compounding Ladder Manager (Noise-Adaptive Matrix)"]
        NewsGuard["News Catalyst Guard (±15m Blackout Window)"]
        
        Bridge -->|Live Ticks & Candles| GoldBot & BTCBot
        FedCal --> NewsGuard
        NewsGuard -->|Pause execution on CPI/NFP/FOMC| GoldBot & BTCBot
        YieldCurve & COT & DVOL --> GoldBot & BTCBot
        Ladder <--> GoldBot & BTCBot
        GoldBot & BTCBot -->|Limit/Market Orders| Bridge
    end

    subgraph Persistence [State & Telemetry]
        DB[("auto_trader.db & btc_auto_trader.db (SQLite)")]
        StateJSON["compounding_ladder_state.json & btc_state.json"]
        Telegram["Telegram Instant Alerts (@Gold_Signals_vip_2026_bot)"]
        GoldBot & BTCBot --> DB & StateJSON & Telegram
    end
```

---

## 🚀 Key Modules & Innovations

### 1. High-Impact News Catalyst Guard (Powered by OpenBB Platform v5)
- **Zero Spread-Spike Drawdowns**: Monitors upcoming macroeconomic events from Federal Reserve and FRED calendars. Automatically enforces an automated **$\pm 15$-minute trading blackout window** around high-impact US releases (**FOMC rate decisions, US CPI, Core PCE, Non-Farm Payrolls, and GDP**).
- **Yield Curve & Recession Monitor**: Evaluates 10Y-2Y US Treasury Yield Curve spreads in real-time. Inversion signals late-cycle structural flight-to-safety into Gold.
- **CFTC Commitment of Traders (COT)**: Tracks COMEX Gold institutional commercial vs. non-commercial speculative positioning (`CFTC_088691`), detecting large hedge fund accumulation.
- **Deribit Bitcoin Volatility Index (DVOL)**: Tracks 30-day option-implied volatility for BTC to identify pre-breakout volatility coils versus exhaustion spikes.

### 2. Autonomous Dual-Asset 24/7 Daemons
- **Gold (`XAUUSD`)**:
  - Strategy: `COMPOUNDING_TOUCH` (1H Donchian 10 Touch + SuperTrend + 50 EMA filter).
  - Noise Clearance: Minimum dynamic stop-loss $\text{SL} = \max(1.5 \times \text{ATR}_{1\text{h}}, \$8.00)$.
  - Schedule: Active trading during liquid London/NY sessions (07:00–19:00 UTC); weekend filter enforced.
- **Bitcoin (`BTCUSD`)**:
  - Strategy: `COMPOUNDING_TOUCH` calibrated for crypto microstructure.
  - Sizing & Clearance: Contract size `1.0` (1 lot = 1 BTC), minimum ATR stop $\ge \$500.00$, Break-Even buffer $+\$30.00$, runner trail buffer $\$500.00$.
  - Schedule: **24/7/365 continuous operation** (no weekend market closure).

### 3. Institutional Two-Stage Compounding Ladder
- **Noise-Adaptive Matrix**: 37 discrete calibrated steps scaling from **$100.00** up to **$5,337.29+**.
- **Stage 1 (Partial Scale-Out)**: At Target 1 (+1.0R), 50% volume is closed into realized cash.
- **Stage 2 (Risk-Free Runner)**: Remaining position stop-loss shifts to Break-Even + buffer, locking in a completely risk-free runner.
- **Stage 3 (Dynamic Wide Trail)**: Runner trailing stop follows peak price excursions with an ATR-cleared buffer up to Target 2 (2.5R).
- **Progressive Ratchet Floors**: Dynamic capital preservation floors ($150 at $250, $350 at $500, $700 at $1,000, $1,800 at $2,500) prevent equity drawdowns from cascading back down.

### 4. Turnkey Linux Wine MT5 Bridge
- Runs official XM Global MT5 (64-bit) under headless Linux using Wine and `Xvfb` virtual display `:99`.
- High-speed TCP socket bridge server (`127.0.0.1:8888`) connects Python with MT5 with zero cloud gateway fees ($0 cost).

---

## 📂 Project Structure

```
/home/mcpuser/MCP/gold-trading-system/
├── pyproject.toml                     # Python packaging & dependencies
├── requirements.txt                   # Platform requirements
├── AGENTS.md                          # Persistent System Architecture Knowledge Base
├── README.md                          # Platform documentation
├── scripts/
│   ├── start_bridge.sh                # Start MT5 Wine socket bridge
│   ├── run_bridge.sh                  # Internal Wine/Xvfb execution wrapper
│   ├── start_auto_trader.sh           # Start Gold auto-trader daemon
│   ├── stop_auto_trader.sh            # Stop Gold auto-trader daemon
│   ├── start_btc_auto_trader.sh       # Start Bitcoin auto-trader daemon
│   ├── stop_btc_auto_trader.sh        # Stop Bitcoin auto-trader daemon
│   ├── export_migration_bundle.sh     # Turn-key VPS export script
│   └── setup_new_vps.sh               # Turn-key new VPS automated bootstrap
├── src/
│   └── gold_trader/
│       ├── core/
│       │   ├── openbb_feed.py         # OpenBB Platform macroeconomic & news catalyst feed
│       │   ├── auto_trader.py         # 24/7 Autonomous Scanner, Execution Controller & Manager
│       │   ├── compounding_ladder.py  # Noise-Adaptive Compounding Matrix & Ratchet Floors
│       │   ├── strategy_engine.py     # 1H Donchian Touch & Momentum Hybrid setups
│       │   ├── macro.py               # Macro drivers (DXY, 10Y Yields, Yield Curve, COT)
│       │   ├── indicators.py          # Supertrend, Donchian, EMA, ATR, RSI calculations
│       │   ├── fibonacci.py           # Golden Pocket retracement & liquidity detection
│       │   ├── price_action.py        # Pin bars, engulfing candles, and liquidity sweeps
│       │   └── telegram_notifier.py   # Real-time Telegram alerts
│       ├── brokers/
│       │   ├── socket_broker.py       # High-speed TCP socket bridge client
│       │   ├── paper_broker.py        # Local SQLite simulated broker
│       │   └── base.py                # Abstract Broker interface
│       ├── scripts/
│       │   ├── run_auto_trader.py     # Unified CLI daemon entrypoint for Gold & Bitcoin
│       │   └── sync_broker_deals.py   # Closed trade reconciliation & P&L sync
│       └── server.py                  # FastMCP tools interface
└── tests/
    ├── test_auto_trader.py            # Auto-trader engine test suite
    ├── test_openbb_feed.py            # OpenBB macro & catalyst guard test suite
    ├── test_high_velocity_touch.py    # Donchian touch execution tests
    └── test_telegram_notifier.py      # Alert formatting tests
```

---

## ⚡ Fast CLI Commands & Operations

### 1. Check System Health & Daemons
```bash
# Check MT5 Bridge Connection
python3 -c "from gold_trader.brokers.socket_broker import SocketBroker; print('Bridge Online:', SocketBroker().connect())"

# Check MT5 Account Balance & Margin
python3 -c "from gold_trader.brokers.socket_broker import SocketBroker; print(SocketBroker().get_account_info())"

# Check Systemd User Services
systemctl --user status mt5-bridge.service gold-auto-trader.service btc-auto-trader.service --no-pager
```

### 2. Verify OpenBB Macro & News Guard
```bash
# Check if current time is within high-impact news blackout window (±15m)
python3 -c "from gold_trader.core.openbb_feed import check_high_impact_news_window; print(check_high_impact_news_window(buffer_minutes=15))"

# Fetch live US Treasury Yield Curve and Macro Confluence
python3 -c "from gold_trader.core.openbb_feed import fetch_openbb_macro_data; print(fetch_openbb_macro_data())"

# Fetch composite Macro Score
python3 -c "from gold_trader.core.macro import analyze_macro_drivers; print(analyze_macro_drivers())"
```

### 3. Check Compounding Ladder States
```bash
# Gold Compounding Ladder Status
python3 -c "from gold_trader.core.compounding_ladder import CompoundingLadderManager; print(CompoundingLadderManager().get_status())"

# Bitcoin Compounding Ladder Status
python3 -c "from gold_trader.core.compounding_ladder import CompoundingLadderManager; print(CompoundingLadderManager(state_file='~/.gold_trading_data/btc_compounding_ladder_state.json').get_status())"
```

### 4. Monitor Live Daemon Logs
```bash
# Tail Gold trading logs
tail -f auto_trader.log

# Tail Bitcoin trading logs
tail -f btc_auto_trader.log

# Tail MT5 Bridge logs
tail -f bridge.log
```

---

## 🚚 Turnkey VPS Migration

To migrate this entire setup (Wine, MT5 terminal, Python environments, database states, and systemd services) to a new VPS in under 5 minutes:

### On Existing VPS:
```bash
cd /home/mcpuser/MCP/gold-trading-system
bash scripts/export_migration_bundle.sh
# Creates /home/mcpuser/gold_trading_migration_bundle.tar.gz
```

### On New VPS (Clean Ubuntu 22.04 / 24.04):
```bash
# Copy bundle and run setup
bash setup_new_vps.sh
```
The script automatically installs Wine, Xvfb, Python 3.13, restores MT5 credentials and databases, configures `systemd` user services, and enables auto-start on boot.

---

## 🧪 Testing

Run the automated test suite with pytest:

```bash
python3 -m pytest tests/test_auto_trader.py tests/test_openbb_feed.py tests/test_price_action.py tests/test_telegram_notifier.py
```

---
*Maintained by Antigravity Agentic Pair Programmer.*
