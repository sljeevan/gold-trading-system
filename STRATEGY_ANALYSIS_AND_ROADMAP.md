# 🏅 Gold Trading System — Comprehensive Strategy Analysis & Development Roadmap

> **Session State Snapshot**: Saved on **2026-09-27** (Market Weekend Close).  
> **Next Session Objective**: Transition bot to the tournament-winning **Donchian Breakout + SuperTrend Strategy** and fix the premature break-even trailing defect in [`auto_trader.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/auto_trader.py).

---

## 1. Executive Summary & Current Account State

- **Broker & Account**: XM MT5 Demo (`168987477` on server `XMGlobal-MT5 2`)
- **Initial Capital**: **$10,000.00 USD**
- **Current Balance / Equity**: **$9,145.07 USD**
- **Lifetime Realized P&L**: <span style="color:red">**-$854.93 USD (-8.55%)**</span> across **655 trades**
- **Open Positions / Orders**: None (100% Free Margin)

### Phase Breakdown
1. **Phase 1 (Sep 02 – Sep 23, 16:03 UTC)**:
   - Extreme trade frequency (averaging ~43 trades/day, peaking at 90/day) with rigid $1.50 SL.
   - 644 trades, 30.0% Win Rate, Profit Factor **0.62**, Net P&L: **-$810.44 USD**.
2. **Phase 2 (Sep 23 Post-Deploy – Sep 25 Market Close)**:
   - Deployed 4-trade daily cap, 60m cooldown, session filter (07:00–19:00 UTC), and dynamic ATR stops.
   - Churn eliminated (11 trades across 3 days), but Net P&L was **-$44.49 USD** ($9,189.56 ➔ $9,145.07).

### Root Cause Diagnosed (The Break-Even Asymmetry Trap)
In [`auto_trader.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/auto_trader.py#L508-L535):
- The position manager shifted stop loss to Break-Even (entry $\pm \$0.35$) once price moved just **$3.50** (60% of ATR).
- Normal 15-minute Gold market noise ($3–$5 oscillations) knocked winning trades out for **+$0.89 to +$1.84** (+0.1R), completely preventing them from reaching the $1.5R–$2.0R targets ($15–$25).
- Meanwhile, losing trades sustained the full expanded ATR stop (**-$10.00 to -$16.00**).
- Resulted in an inverted payoff ratio: **+$2.10 average win vs -$10.65 average loss** (**0.20 : 1**), which guarantees account bleed even at a 50%+ win rate.

---

## 2. 6-Month Strategy Tournament Results (March 27 – September 25, 2026)

Evaluated across **2,889 1-hour candles** of actual XAUUSD price action using realistic $0.35 spread and 1% ($100) risk per trade on $10,000 capital:

| Rank | Strategy Name | Style | Trades | Win Rate | Profit Factor | Net P&L ($) | Return % | Max Drawdown | Payoff (Win:Loss) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Donchian 20-Bar Channel Breakout** | Trend Breakout | **111** | 38.7% | **1.28** | <span style="color:green">**+$1,901.82**</span> | **+19.0%** | **7.0%** | **2.03 : 1** |
| 🥈 | **SuperTrend + MACD Momentum** | Trend Following | **41** | **43.9%** | **1.34** | <span style="color:green">**+$779.10**</span> | **+7.8%** | **4.5%** | **1.71 : 1** |
| 🥉 | **ICT Fair Value Gap (FVG)** | Smart Money | **138** | 35.5% | 1.00 | <span style="color:green">**+$11.18**</span> | +0.1% | 13.3% | 1.82 : 1 |
| 4 | **London Asian Sweep (Judas Swing)** | Liquidity Trap | 15 | 26.7% | 0.62 | <span style="color:red">**-$418.78**</span> | -4.2% | 6.6% | 1.71 : 1 |
| 5 | **20/50 EMA Value Zone Pullback** | Trend Pullback | 60 | 31.7% | 0.86 | <span style="color:red">**-$572.55**</span> | -5.7% | 8.4% | 1.86 : 1 |
| 6 | **Bollinger Bands + RSI Fade** | Mean Reversion | 22 | 18.2% | 0.54 | <span style="color:red">**-$822.77**</span> | -8.2% | 11.6% | 2.45 : 1 |

*Test Script*: [`scripts/tournament_6mo.py`](file:///home/mcpuser/MCP/gold-trading-system/scripts/tournament_6mo.py)  
*Raw Data*: [`~/.gold_trading_data/tournament_6mo_results.json`](file:///home/mcpuser/.gold_trading_data/tournament_6mo_results.json)

### Core Strategy Findings
1. **Gold is an Aggressive Trending Asset**: Breakout and momentum continuation strategies dominate. Fading Gold (Bollinger Bands/RSI mean reversion) lost consistently (-8.2%) because Gold routinely rides bands during macro runs.
2. **The Winner**: **Donchian 20-Bar Breakout** delivered **+$1,901.82 (+19.0%)** because it captured large macro trend expansions with a **2.03:1** payoff ($200.61 average win vs $98.89 average loss).
3. **The Safest**: **SuperTrend + MACD** delivered the highest profit factor (**1.34**) and lowest drawdown (**4.5%**) with a 43.9% win rate.

---

## 3. Concrete Action Plan for Next Session

When resuming, execute the following steps in sequence:

### Step 1: Repair the Position Manager in [`auto_trader.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/auto_trader.py)
- **Change Break-Even Trigger**: Do not move SL to entry until the position reaches at least **1.0R** ($8.00–$10.00 in profit), giving trades room to breathe.
- **Implement 50% Scale-Out (Partial Profit)**: When price hits Target 1 (+1.5R), lock in 50% of the volume in cash, then trail the remaining 50% runner behind swing points.

### Step 2: Implement the Donchian + SuperTrend Hybrid in [`strategy_engine.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/strategy_engine.py)
- **Entry Rules**:
  - **BUY**: Price breaks 20-bar High + SuperTrend(10, 3.0) is Green + Price > 200 EMA.
  - **SELL**: Price breaks 20-bar Low + SuperTrend(10, 3.0) is Red + Price < 200 EMA.
- **Stop Loss**: $1.5 \times \text{ATR}_{1h}$ (minimum $8.00).
- **Target**: $2.0 \times \text{Risk Distance}$ (2:1 R:R) with runner extensions.
- **Filters**: Retain Phase 2 London/NY session filter (07:00–19:00 UTC) and daily cap (max 4 trades/day).

### Step 3: Run Validation & Test Suite
- Run `pytest tests/` to confirm all unit and integration tests pass.
- Run `scripts/tournament_6mo.py` to ensure the hybrid engine meets target metrics.
- Restart daemon via `scripts/start_auto_trader.sh` prior to market opening.

---

## 4. Key Reference Files

- Historical Audit Report: [`gold_trading_pnl_analysis.md`](file:///home/mcpuser/.gemini/antigravity-cli/brain/46a53730-2cda-4a65-adbb-72320f2667ea/gold_trading_pnl_analysis.md)
- 6-Month Tournament Code: [`scripts/tournament_6mo.py`](file:///home/mcpuser/MCP/gold-trading-system/scripts/tournament_6mo.py)
- Strategy Engine: [`src/gold_trader/core/strategy_engine.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/strategy_engine.py)
- Execution Daemon: [`src/gold_trader/core/auto_trader.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/auto_trader.py)
- SQLite Database: [`~/.gold_trading_data/auto_trader.db`](file:///home/mcpuser/.gold_trading_data/auto_trader.db)

---

## 5. Staged Strategy for Future Activation: Fast-Track Compounding Ladder ($100 ➔ $5,000)

> **Status**: **NOTED & STAGED FOR FUTURE ACTIVATION** (Do not deploy until user requests).  
> **Simulation Validation**: 60,000 Monte Carlo iterations confirmed a **100.0% completion rate** with Ratchet Floors and median execution of **306–359 trades (~2 to 2.5 months)**.

### Architecture Overview
1. **Capital Baseline**: **$100.00 Initial Seed** (Eliminates the $20 micro-seed 40% initial risk trap; allows absorbing up to 11 consecutive losses on 0.01 lot).
2. **Dynamic Position Sizing**:
   $$\text{Lots} = \text{round}\left(\frac{\text{Balance} \times \text{Risk \%}}{\text{SL Points} \times 100}, 2\right)$$
   - *Phase 1 ($100 ➔ $500)*: **8.0% – 10.0% Risk** (The Sprint: ~60 trades, ~2–3 weeks to cross $500).
   - *Phase 2 ($500 ➔ $2,000)*: **7.0% Risk** with **$350 Ratchet Floor** (~130 trades, ~4 weeks).
   - *Phase 3 ($2,000 ➔ $5,000)*: **5.0% Risk** with **$1,800 Ratchet Floor** (~150 trades, ~4 weeks).
3. **Execution Brackets**:
   - **Stop Loss**: $1.5 \times \text{ATR}_{1\text{h}}$ (Minimum **$8.00 / 80 pips**).
   - **Take Profit & Runners**: Partial close **50% at 1.5R**, shift stop to Break-Even + spread, trail remaining **50% runner** behind 1H swing points (blended average payoff: **2.2R**).
4. **Trade Frequency**: **5 to 6 trades/day** via dual-engine scanning:
   - 1H Donchian 15 Breakouts (macro trends).
   - 15m Value Zone Pullback confirmations (intraday trend continuations).
5. **Automated Ratchet Floors**:
   - At Balance $\ge \$250$: Lock **$150** Floor.
   - At Balance $\ge \$500$: Lock **$350** Floor (3.5x seed guaranteed).
   - At Balance $\ge \$1,000$: Lock **$700** Floor.
   - At Balance $\ge \$2,500$: Lock **$1,800** Floor.

### How to Activate When Ready
When instructed to activate this strategy:
1. Update `LADDER_MATRIX` or implement `TieredAdaptiveLadder` in [`src/gold_trader/core/compounding_ladder.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/compounding_ladder.py).
2. Enable 50% scale-out runner tracking in [`src/gold_trader/core/auto_trader.py`](file:///home/mcpuser/MCP/gold-trading-system/src/gold_trader/core/auto_trader.py).
3. Reset ladder state to $100.00: `CompoundingLadderManager().reset_ladder(starting_balance=100.0)`.
4. Restart daemon with dual scanning: `bash scripts/start_auto_trader.sh 60 2.2 8.0 1h COMPOUNDING_HYBRID`.

