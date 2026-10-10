"""
Comprehensive Quantitative Backtest of the "Silent Flip" Strategy on Spot Gold (XAU/USD).
Strategy Source: ProRealAlgos ("How Trading Like an Idiot Makes Me $20,000/Month")

Dataset: 12,000 15-Minute Candles directly from XM MetaTrader 5 Broker (April 2026 - October 2026, 6 Months).
Asset: Spot Gold (XAU/USD)
Account Capital: $10,000.00
Risk per Trade: 1.0% ($100.00 planned dollar risk with dynamic lot sizing)
Execution Friction: 4.0 pips ($0.40 / oz) broker spread + slippage

Tested Variables:
1. Session Timing:
   - London Open Only (First 60 mins: 10:00 - 11:00 MT5 / 07:00 - 08:00 UTC)
   - New York Open Only (First 60 mins: 16:30 - 17:30 MT5 / 13:30 - 14:30 UTC)
   - Dual Session Opens (London & NY Opens combined)
   - Full Active London/NY Trading Session (10:00 - 20:00 MT5)
   - 24-Hour Scan (Any time setup triggers)
2. Exit & Take-Profit Mechanics:
   - Original Video Rule: Target Opposite Range Level (Range Low -> Range High)
   - Fast Mean-Reversion: Fixed 1.2R Target
   - Standard Asymmetric: Fixed 2.0R Target
   - Dual-Target: 50% Scale-Out @ 1.5R, Move SL to BE, 50% Runner to Opposite Level
3. Stop-Loss Mechanics:
   - Tight Structural Stop: Tested Level + $1.50 buffer (Retail approach)
   - Volatility-Adaptive Stop: max(Level, 1.5x 15m ATR) (Institutional approach)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.indicators import calc_atr


def run_comprehensive_study():
    print("=" * 85)
    print("      SILENT FLIP (PROREALALGOS) - 6-MONTH GOLD (XAU/USD) QUANTITATIVE BENCHMARK")
    print("=" * 85)

    # 1. Fetch 12,000 candles from XM MT5 Broker
    broker = SocketBroker()
    raw_candles = broker.get_rates("GOLD", "15m", 12000)
    if not raw_candles or len(raw_candles) < 2000:
        print(f"Error: only received {len(raw_candles) if raw_candles else 0} candles.")
        return

    n = len(raw_candles)
    start_dt = raw_candles[0]["date"]
    end_dt = raw_candles[-1]["date"]
    print(f"Candles Loaded: {n:,} 15m bars | Span: {start_dt} to {end_dt} (6 Full Months)")

    # 2. Build Daily Map for Prior Day Range & Flip Levels
    daily_candles: Dict[str, Dict[str, float]] = {}
    for c in raw_candles:
        day_str = c["date"][:10]
        h = float(c["high"])
        l = float(c["low"])
        if day_str not in daily_candles:
            daily_candles[day_str] = {"high": h, "low": l}
        else:
            daily_candles[day_str]["high"] = max(daily_candles[day_str]["high"], h)
            daily_candles[day_str]["low"] = min(daily_candles[day_str]["low"], l)

    sorted_days = sorted(daily_candles.keys())

    levels_by_day: Dict[str, Dict[str, float]] = {}
    for i in range(4, len(sorted_days)):
        today = sorted_days[i]
        yesterday = sorted_days[i - 1]

        rh = daily_candles[yesterday]["high"]
        rl = daily_candles[yesterday]["low"]

        prior_highs = [daily_candles[sorted_days[k]]["high"] for k in range(i - 4, i - 1)]
        prior_lows = [daily_candles[sorted_days[k]]["low"] for k in range(i - 4, i - 1)]

        fh = max(prior_highs)
        fl = min(prior_lows)

        if fh <= rh:
            fh = rh + max(8.0, (rh - rl) * 0.5)
        if fl >= rl:
            fl = rl - max(8.0, (rh - rl) * 0.5)

        levels_by_day[today] = {
            "RH": rh,
            "RL": rl,
            "FH": fh,
            "FL": fl,
        }

    # Precalculate ATR (14-period on 15m)
    highs = [float(c["high"]) for c in raw_candles]
    lows = [float(c["low"]) for c in raw_candles]
    closes = [float(c["close"]) for c in raw_candles]
    atr_series = calc_atr(highs, lows, closes, 14)

    # Define Simulation Scenarios to Test
    scenarios = [
        # --- Group 1: Session Opening Windows (First 60 mins of Market Open) ---
        {
            "name": "Scenario 1: London Open (First 60m) - Original Video (Opposite Level)",
            "session_filter": lambda t: "10:00" <= t <= "11:00",
            "tp_mode": "OPPOSITE_LEVEL",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 2: London Open (First 60m) - Fixed 1.5R Fast Target",
            "session_filter": lambda t: "10:00" <= t <= "11:00",
            "tp_mode": "FIXED_1.5R",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 3: NY Open (First 60m: 16:30-17:30) - Original Video (Opposite Level)",
            "session_filter": lambda t: "16:30" <= t <= "17:30",
            "tp_mode": "OPPOSITE_LEVEL",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 4: Dual Open (London + NY Opens) - Original Video (Opposite Level)",
            "session_filter": lambda t: ("10:00" <= t <= "11:00") or ("16:30" <= t <= "17:30"),
            "tp_mode": "OPPOSITE_LEVEL",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 5: Dual Open (London + NY Opens) - Fixed 2.0R Target",
            "session_filter": lambda t: ("10:00" <= t <= "11:00") or ("16:30" <= t <= "17:30"),
            "tp_mode": "FIXED_2.0R",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 6: Dual Open - ATR Volatility Stop ($1.5x ATR) + 2.0R Target",
            "session_filter": lambda t: ("10:00" <= t <= "11:00") or ("16:30" <= t <= "17:30"),
            "tp_mode": "FIXED_2.0R",
            "sl_mode": "ATR",
            "buffer": 2.00,
        },
        {
            "name": "Scenario 7: Dual Open - Partial Scale-Out (50% @ 1.5R + Runner to Opp Level)",
            "session_filter": lambda t: ("10:00" <= t <= "11:00") or ("16:30" <= t <= "17:30"),
            "tp_mode": "PARTIAL_SCALE",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        # --- Group 2: Full Liquid Session (10:00 to 20:00 MT5) ---
        {
            "name": "Scenario 8: Full London/NY Session (10:00-20:00) - Original Video (Opposite Level)",
            "session_filter": lambda t: "10:00" <= t <= "20:00",
            "tp_mode": "OPPOSITE_LEVEL",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 9: Full London/NY Session - Fixed 1.5R Fast Target",
            "session_filter": lambda t: "10:00" <= t <= "20:00",
            "tp_mode": "FIXED_1.5R",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 10: Full London/NY Session - ATR Volatility Stop + 2.0R Target",
            "session_filter": lambda t: "10:00" <= t <= "20:00",
            "tp_mode": "FIXED_2.0R",
            "sl_mode": "ATR",
            "buffer": 2.00,
        },
        {
            "name": "Scenario 11: Full 24-Hour Scan - Original Video (Opposite Level)",
            "session_filter": lambda t: True,
            "tp_mode": "OPPOSITE_LEVEL",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
        {
            "name": "Scenario 12: Full 24-Hour Scan - Fixed 1.5R Target",
            "session_filter": lambda t: True,
            "tp_mode": "FIXED_1.5R",
            "sl_mode": "STRUCTURAL",
            "buffer": 1.50,
        },
    ]

    initial_balance = 10000.0
    risk_per_trade_usd = 100.0
    spread_usd = 0.40 # 4.0 pips

    all_scenario_results = []

    for sc in scenarios:
        s_name = sc["name"]
        filter_fn = sc["session_filter"]
        tp_mode = sc["tp_mode"]
        sl_mode = sc["sl_mode"]
        buffer_usd = sc["buffer"]

        equity = initial_balance
        peak_equity = initial_balance
        max_dd_usd = 0.0
        max_dd_pct = 0.0

        closed_trades: List[Dict[str, Any]] = []
        monthly_stats: Dict[str, Dict[str, Any]] = {}

        # Track active trade so we don't open duplicate simultaneous trades
        in_trade_until_idx = -1

        for i in range(50, n - 40):
            if i <= in_trade_until_idx:
                continue

            c1 = raw_candles[i]
            dt_str = c1["date"]
            day_str = dt_str[:10]
            time_str = dt_str[11:16]

            if day_str not in levels_by_day:
                continue

            if not filter_fn(time_str):
                continue

            levels = levels_by_day[day_str]
            rh = levels["RH"]
            rl = levels["RL"]
            fh = levels["FH"]
            fl = levels["FL"]

            c2 = raw_candles[i + 1]
            c3 = raw_candles[i + 2]

            c1_h, c1_l, c1_o, c1_c = float(c1["high"]), float(c1["low"]), float(c1["open"]), float(c1["close"])
            c2_h, c2_l, c2_o, c2_c = float(c2["high"]), float(c2["low"]), float(c2["open"]), float(c2["close"])
            c3_h, c3_l = float(c3["high"]), float(c3["low"])

            current_atr = atr_series[i] or 4.50

            signal = None
            leaned_lvl = None
            opp_lvl = None

            # 1. SELL SETUP (At Range High or Flip High)
            for test_lvl, target_lvl in [(fh, fl), (rh, rl)]:
                # C1 tests level (high reaches level) and respects it (close <= level)
                if c1_h >= test_lvl and c1_c <= test_lvl:
                    # C1 was bullish impulse into level
                    # C2 (Silent Candle) must close bearish (against C1) and respect level
                    if c2_c < c2_o and c2_c <= test_lvl:
                        signal = "SELL"
                        leaned_lvl = test_lvl
                        opp_lvl = target_lvl
                        break

            # 2. BUY SETUP (At Range Low or Flip Low)
            if not signal:
                for test_lvl, target_lvl in [(fl, fh), (rl, rh)]:
                    # C1 tests level (low reaches level) and respects it (close >= level)
                    if c1_l <= test_lvl and c1_c >= test_lvl:
                        # C2 (Silent Candle) must close bullish (against C1) and respect level
                        if c2_c > c2_o and c2_c >= test_lvl:
                            signal = "BUY"
                            leaned_lvl = test_lvl
                            opp_lvl = target_lvl
                            break

            if not signal:
                continue

            # Candle 3: Check entry trigger
            entry_price = None
            sl_price = None
            tp_price = None

            if signal == "BUY":
                # Must break above High of Candle 2
                if c3_h > c2_h:
                    entry_price = c2_h + (spread_usd / 2.0)
                    if sl_mode == "ATR":
                        sl_price = entry_price - max(1.5 * current_atr, 6.0)
                    else:
                        sl_price = min(c1_l, c2_l, leaned_lvl) - buffer_usd

                    sl_distance = entry_price - sl_price
                    if sl_distance <= 1.0:
                        sl_distance = 2.50
                        sl_price = entry_price - sl_distance

                    if tp_mode == "OPPOSITE_LEVEL":
                        tp_price = opp_lvl
                    elif tp_mode == "FIXED_1.5R":
                        tp_price = entry_price + (1.5 * sl_distance)
                    elif tp_mode in ("FIXED_2.0R", "PARTIAL_SCALE"):
                        tp_price = entry_price + (2.0 * sl_distance)

            elif signal == "SELL":
                # Must break below Low of Candle 2
                if c3_l < c2_l:
                    entry_price = c2_l - (spread_usd / 2.0)
                    if sl_mode == "ATR":
                        sl_price = entry_price + max(1.5 * current_atr, 6.0)
                    else:
                        sl_price = max(c1_h, c2_h, leaned_lvl) + buffer_usd

                    sl_distance = sl_price - entry_price
                    if sl_distance <= 1.0:
                        sl_distance = 2.50
                        sl_price = entry_price + sl_distance

                    if tp_mode == "OPPOSITE_LEVEL":
                        tp_price = opp_lvl
                    elif tp_mode == "FIXED_1.5R":
                        tp_price = entry_price - (1.5 * sl_distance)
                    elif tp_mode in ("FIXED_2.0R", "PARTIAL_SCALE"):
                        tp_price = entry_price - (2.0 * sl_distance)

            if entry_price is None or sl_price is None or tp_price is None:
                continue

            # Position sizing: 1% risk ($100)
            sl_dist = abs(entry_price - sl_price)
            lots = round(risk_per_trade_usd / (sl_dist * 100.0), 2)
            lots = max(0.01, min(lots, 5.0))

            # Simulate Forward Execution (walk candle by candle, max 32 bars = 8 hours)
            exit_price = None
            exit_reason = None
            half_closed = False
            half_pnl = 0.0
            tp1_target = entry_price + (1.5 * sl_dist) if signal == "BUY" else entry_price - (1.5 * sl_dist)

            last_step_idx = i + 2

            for step in range(i + 2, min(i + 36, n)):
                last_step_idx = step
                bar = raw_candles[step]
                b_high = float(bar["high"])
                b_low = float(bar["low"])

                # Partial scale-out logic
                if tp_mode == "PARTIAL_SCALE" and not half_closed:
                    if signal == "BUY" and b_high >= tp1_target:
                        half_closed = True
                        half_pnl = 0.5 * lots * (tp1_target - entry_price - spread_usd) * 100.0
                        sl_price = entry_price + 0.20
                        tp_price = opp_lvl
                    elif signal == "SELL" and b_low <= tp1_target:
                        half_closed = True
                        half_pnl = 0.5 * lots * (entry_price - tp1_target - spread_usd) * 100.0
                        sl_price = entry_price - 0.20
                        tp_price = opp_lvl

                # Check SL & TP
                if signal == "BUY":
                    if b_low <= sl_price:
                        exit_price = sl_price
                        exit_reason = "STOP_LOSS" if not half_closed else "BE_STOP"
                        break
                    elif b_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TAKE_PROFIT"
                        break
                elif signal == "SELL":
                    if b_high >= sl_price:
                        exit_price = sl_price
                        exit_reason = "STOP_LOSS" if not half_closed else "BE_STOP"
                        break
                    elif b_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = "TAKE_PROFIT"
                        break

            if exit_price is None:
                final_bar = raw_candles[min(i + 35, n - 1)]
                exit_price = float(final_bar["close"])
                exit_reason = "TIME_EXPIRY"

            in_trade_until_idx = last_step_idx

            # Calculate PnL
            if tp_mode == "PARTIAL_SCALE":
                if half_closed:
                    rem_lots = 0.5 * lots
                    if signal == "BUY":
                        rem_pnl = rem_lots * (exit_price - entry_price - spread_usd) * 100.0
                    else:
                        rem_pnl = rem_lots * (entry_price - exit_price - spread_usd) * 100.0
                    pnl_usd = round(half_pnl + rem_pnl, 2)
                else:
                    if signal == "BUY":
                        pnl_usd = round(lots * (exit_price - entry_price - spread_usd) * 100.0, 2)
                    else:
                        pnl_usd = round(lots * (entry_price - exit_price - spread_usd) * 100.0, 2)
            else:
                if signal == "BUY":
                    pnl_usd = round(lots * (exit_price - entry_price - spread_usd) * 100.0, 2)
                else:
                    pnl_usd = round(lots * (entry_price - exit_price - spread_usd) * 100.0, 2)

            equity += pnl_usd
            if equity > peak_equity:
                peak_equity = equity
            dd_usd = peak_equity - equity
            dd_pct = (dd_usd / peak_equity) * 100.0
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            month_str = dt_str[:7]
            if month_str not in monthly_stats:
                monthly_stats[month_str] = {"trades": 0, "wins": 0, "pnl": 0.0}
            monthly_stats[month_str]["trades"] += 1
            if pnl_usd > 0:
                monthly_stats[month_str]["wins"] += 1
            monthly_stats[month_str]["pnl"] = round(monthly_stats[month_str]["pnl"] + pnl_usd, 2)

            closed_trades.append({
                "date": dt_str,
                "signal": signal,
                "entry": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "exit": exit_price,
                "reason": exit_reason,
                "pnl": pnl_usd,
            })

        # Calculate Statistics
        total_t = len(closed_trades)
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] <= 0]
        win_rate = (len(wins) / total_t * 100.0) if total_t > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        net_pnl = equity - initial_balance
        return_pct = (net_pnl / initial_balance) * 100.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        res_entry = {
            "scenario": s_name,
            "trades": total_t,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(return_pct, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(payoff, 2),
            "max_dd_pct": round(max_dd_pct, 1),
            "final_equity": round(equity, 2),
            "monthly": monthly_stats,
        }
        all_scenario_results.append(res_entry)

        print(f"\n{s_name}")
        print(f"  Trades: {total_t:3d} | Win Rate: {win_rate:5.1f}% | Profit Factor: {profit_factor:5.2f}")
        print(f"  Net P&L: ${net_pnl:+9.2f} ({return_pct:+6.1f}%) | Max Drawdown: {max_dd_pct:5.1f}%")
        print(f"  Avg Win: ${avg_win:6.2f} | Avg Loss: ${avg_loss:6.2f} | Payoff Ratio: {payoff:4.2f}:1")

    # Output JSON summary
    out_file = os.path.expanduser("~/.gold_trading_data/silent_flip_comprehensive_results.json")
    with open(out_file, "w") as f:
        json.dump(all_scenario_results, f, indent=2)
    print(f"\nSaved all results to: {out_file}")

    return all_scenario_results


if __name__ == "__main__":
    run_comprehensive_study()
