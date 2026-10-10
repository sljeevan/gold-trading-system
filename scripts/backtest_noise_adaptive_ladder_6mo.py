#!/usr/bin/env python3
"""
6-Month Historical Backtest for the Gold Noise-Adaptive Compounding Ladder.
Evaluates the exact strategy currently running live on XM MT5:
- Timeframe: 1-Hour candles (March - October 2026, ~4,500 hours / 6 months)
- Strategy: Donchian 15 Breakout + SuperTrend (10, 3.0) + 200 EMA Filter
- Session Filter: 07:00 - 19:00 UTC (London/NY overlap)
- Seed: $100.00 starting capital
- Noise Clearance: SL = max(1.5 * ATR_1h, $8.00)
- Target: 1.5R (and 1.8R comparison)
- Lots: Dynamic dollar risk lot calculation
- Capital Protection: Progressive Ratchet Floors + 2-Loss Circuit Breaker
"""

import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.core.compounding_ladder import (
    NOISE_ADAPTIVE_MATRIX,
    find_closest_level_for_balance,
    get_noise_adaptive_level_info,
)
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_supertrend,
)

SPREAD_USD = 0.35  # Realistic XM live broker spread ($0.35/oz)


def run_noise_adaptive_backtest(candles_1h: List[Dict[str, Any]], target_rr: float = 1.5):
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h])
    lows = np.array([float(c["low"]) for c in candles_1h])
    closes = np.array([float(c["close"]) for c in candles_1h])
    opens = np.array([float(c["open"]) for c in candles_1h])
    dates = [c["date"] for c in candles_1h]

    # Calculate indicators
    donch15 = calc_donchian(highs.tolist(), lows.tolist(), 15)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    # State tracking
    balance = 100.0
    peak_balance = balance
    current_level = 1
    ratchet_floor = 0.0
    consecutive_losses = 0
    consecutive_wins = 0
    max_consec_wins = 0
    max_consec_losses = 0

    wins = 0
    losses = 0
    scratches = 0
    gross_profit = 0.0
    gross_loss = 0.0

    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    active_until = -1
    trades = []
    milestones = {}

    for i in range(200, n - 48):
        if i <= active_until:
            continue

        dt = dates[i]
        # Parse hour for session filter (07:00 - 19:00 UTC)
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19):
            continue

        d_up = donch15["upper"][i - 1]
        d_dn = donch15["lower"][i - 1]
        st_dir = st["direction"][i]
        e200 = ema200[i]
        atr_val = atr14[i]

        if None in (d_up, d_dn, st_dir, e200, atr_val):
            continue

        c = closes[i]
        direction = None
        if c > d_up and st_dir == 1 and c > e200:
            direction = "BUY"
        elif c < d_dn and st_dir == -1 and c < e200:
            direction = "SELL"

        if not direction:
            continue

        # Progressive Ratchet Floor update
        if balance >= 4000.0:
            ratchet_floor = max(ratchet_floor, 3000.0)
        elif balance >= 2500.0:
            ratchet_floor = max(ratchet_floor, 1800.0)
        elif balance >= 1000.0:
            ratchet_floor = max(ratchet_floor, 700.0)
        elif balance >= 500.0:
            ratchet_floor = max(ratchet_floor, 350.0)
        elif balance >= 250.0:
            ratchet_floor = max(ratchet_floor, 150.0)

        # Get level specs from NOISE_ADAPTIVE_MATRIX
        lvl_info = get_noise_adaptive_level_info(current_level)
        planned_risk_usd = lvl_info["risk_dollars"]

        # Circuit breaker: halve risk after 2 consecutive losses
        cb_active = consecutive_losses >= 2
        if cb_active:
            planned_risk_usd *= 0.50

        # Enforce minimum ATR stop clearance ($8.00)
        sl_pts = max(round(1.5 * atr_val, 2), 8.00)
        tp_pts = round(sl_pts * target_rr, 2)
        be_trigger_pts = round(sl_pts * 1.0, 2)  # Delay BE to 1.0R

        # Dynamic lot sizing: Lots = planned_risk / (sl_pts * 100)
        lots = max(0.01, min(50.0, math.floor((planned_risk_usd / (sl_pts * 100.0)) * 100.0 + 0.0001) / 100.0))
        dollar_per_pt = lots * 100.0

        eff_entry = c + SPREAD_USD if direction == "BUY" else c
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + SPREAD_USD
        tp_price = eff_entry + tp_pts if direction == "BUY" else eff_entry - tp_pts - SPREAD_USD
        be_price = eff_entry + be_trigger_pts if direction == "BUY" else eff_entry - be_trigger_pts

        # Intrabar pathing (worst-case adverse check first)
        outcome = "TIME_EXIT"
        is_be_engaged = False
        bars_held = 0

        for k in range(i + 1, min(i + 49, n)):
            bars_held = k - i
            h_k = highs[k]
            l_k = lows[k]
            curr_sl = eff_entry if is_be_engaged else sl_price

            if direction == "BUY":
                if l_k <= curr_sl:
                    outcome = "SCRATCH" if is_be_engaged else "LOSS"
                    break
                if h_k >= be_price:
                    is_be_engaged = True
                if h_k >= tp_price:
                    outcome = "WIN"
                    break
            else:
                if h_k >= curr_sl:
                    outcome = "SCRATCH" if is_be_engaged else "LOSS"
                    break
                if l_k <= be_price:
                    is_be_engaged = True
                if l_k <= tp_price:
                    outcome = "WIN"
                    break

        active_until = i + bars_held
        trade_pnl = 0.0

        if outcome == "WIN":
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)

            trade_pnl = tp_pts * dollar_per_pt
            gross_profit += trade_pnl
            balance += trade_pnl

            # Advance level on win
            if current_level < len(NOISE_ADAPTIVE_MATRIX):
                current_level += 1

        elif outcome == "SCRATCH":
            scratches += 1
            trade_pnl = 0.0  # Risk-free scratch

        else:  # LOSS
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)

            trade_pnl = -sl_pts * dollar_per_pt
            gross_loss += abs(trade_pnl)
            balance = max(ratchet_floor, balance - abs(trade_pnl))

            # Step down level per progressive ratchet policy
            current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)

        peak_balance = max(peak_balance, balance)
        dd_dollars = peak_balance - balance
        dd_pct = (dd_dollars / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_dd_dollars = max(max_dd_dollars, dd_dollars)
        max_dd_pct = max(max_dd_pct, dd_pct)

        trade_record = {
            "trade_no": wins + losses + scratches,
            "date": dt,
            "direction": direction,
            "entry_price": round(c, 2),
            "level": lvl_info["level"],
            "phase": lvl_info["phase"],
            "lots": lots,
            "sl_pts": sl_pts,
            "tp_pts": tp_pts,
            "outcome": outcome,
            "pnl": round(trade_pnl, 2),
            "balance": round(balance, 2),
            "ratchet_floor": ratchet_floor,
            "cb_active": cb_active,
        }
        trades.append(trade_record)

        # Record milestones
        for m in [200, 350, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]:
            if balance >= m and m not in milestones:
                milestones[m] = {"date": dt, "trade_no": len(trades), "balance": round(balance, 2)}

    # Month-by-month performance
    monthly = {}
    for t in trades:
        m_key = t["date"][:7]
        if m_key not in monthly:
            monthly[m_key] = {"trades": 0, "wins": 0, "losses": 0, "scratches": 0, "pnl": 0.0, "end_bal": 0.0}
        monthly[m_key]["trades"] += 1
        if t["outcome"] == "WIN":
            monthly[m_key]["wins"] += 1
        elif t["outcome"] == "LOSS":
            monthly[m_key]["losses"] += 1
        else:
            monthly[m_key]["scratches"] += 1
        monthly[m_key]["pnl"] += t["pnl"]
        monthly[m_key]["end_bal"] = t["balance"]

    decided = wins + losses
    win_rate_decided = (wins / decided * 100.0) if decided > 0 else 0.0
    win_rate_total = (wins / len(trades) * 100.0) if trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    return {
        "target_rr": target_rr,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_decided": round(win_rate_decided, 1),
        "win_rate_total": round(win_rate_total, 1),
        "profit_factor": round(profit_factor, 2),
        "starting_balance": 100.0,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - 100.0, 2),
        "return_pct": round(((balance - 100.0) / 100.0) * 100.0, 1),
        "peak_balance": round(peak_balance, 2),
        "ratchet_floor": ratchet_floor,
        "peak_level": max([t["level"] for t in trades]) if trades else 1,
        "ending_level": current_level,
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "milestones": milestones,
        "monthly": monthly,
        "trades": trades,
    }


def main():
    print("=" * 75)
    print("📊 RUNNING 6-MONTH HISTORICAL BACKTEST ON GOLD (XAU/USD 1H)")
    print("=" * 75)

    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    if not candles or len(candles) < 500:
        print(f"Error fetching candles: {len(candles) if candles else 0} received.")
        return

    print(f"Loaded {len(candles)} historical 1-Hour candles from {candles[0]['date']} to {candles[-1]['date']}.")

    # Run for 1.5R (Noise-Adaptive standard)
    res_15 = run_noise_adaptive_backtest(candles, target_rr=1.5)

    # Run for 1.8R (Hybrid extended target)
    res_18 = run_noise_adaptive_backtest(candles, target_rr=1.8)

    print("\n" + "=" * 75)
    print(" 🏆 6-MONTH BACKTEST RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<36} {'1.5R Target (Standard)':<20} {'1.8R Target (Extended)':<20}")
    print("-" * 75)
    print(f"{'Starting Balance':<36} ${res_15['starting_balance']:<19.2f} ${res_18['starting_balance']:<19.2f}")
    print(f"{'Ending Balance':<36} ${res_15['ending_balance']:<19,.2f} ${res_18['ending_balance']:<19,.2f}")
    print(f"{'Net Profit ($)':<36} +${res_15['net_profit']:<18,.2f} +${res_18['net_profit']:<18,.2f}")
    print(f"{'Total Return (%)':<36} +{res_15['return_pct']:<18.1f}% +{res_18['return_pct']:<18.1f}%")
    print(f"{'Total Trades':<36} {res_15['total_trades']:<20} {res_18['total_trades']:<20}")
    print(f"{'Wins / Losses / Scratches':<36} {res_15['wins']}W / {res_15['losses']}L / {res_15['scratches']}S{'':<6} {res_18['wins']}W / {res_18['losses']}L / {res_18['scratches']}S")
    print(f"{'Win Rate (Decided Trades)':<36} {res_15['win_rate_decided']:<19}% {res_18['win_rate_decided']:<19}%")
    print(f"{'Profit Factor':<36} {res_15['profit_factor']:<20} {res_18['profit_factor']:<20}")
    print(f"{'Peak Balance':<36} ${res_15['peak_balance']:<19,.2f} ${res_18['peak_balance']:<19,.2f}")
    print(f"{'Locked Ratchet Floor':<36} ${res_15['ratchet_floor']:<19,.2f} ${res_18['ratchet_floor']:<19,.2f}")
    print(f"{'Peak Ladder Level':<36} Level {res_15['peak_level']:<13} Level {res_18['peak_level']:<13}")
    print(f"{'Max Drawdown ($)':<36} ${res_15['max_dd_dollars']:<19,.2f} ${res_18['max_dd_dollars']:<19,.2f}")
    print(f"{'Max Drawdown (%)':<36} {res_15['max_dd_pct']:<19.1f}% {res_18['max_dd_pct']:<19.1f}%")
    print(f"{'Max Win Streak':<36} {res_15['max_consec_wins']} trades{'':<11} {res_18['max_consec_wins']} trades")
    print(f"{'Max Loss Streak':<36} {res_15['max_consec_losses']} trades{'':<11} {res_18['max_consec_losses']} trades")

    print("\n" + "=" * 75)
    print(" 📅 MONTH-BY-MONTH BREAKDOWN (1.5R Standard Target)")
    print("=" * 75)
    print(f"{'Month':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'BE':<5} {'Win Rate':<10} {'Net P&L ($)':<14} {'Ending Equity ($)'}")
    print("-" * 75)
    for m, d in sorted(res_15["monthly"].items()):
        dec = d["wins"] + d["losses"]
        wr = (d["wins"] / dec * 100.0) if dec > 0 else 0.0
        print(f"{m:<10} {d['trades']:<8} {d['wins']:<6} {d['losses']:<8} {d['scratches']:<5} {wr:<9.1f}% +${d['pnl']:<12,.2f} ${d['end_bal']:<12,.2f}")

    print("\n" + "=" * 75)
    print(" 🎯 CAPITAL MILESTONES ACHIEVED (1.5R)")
    print("=" * 75)
    for m, info in sorted(res_15["milestones"].items()):
        print(f"   • ${m:>4,}: Reached on {info['date'][:10]} (Trade #{info['trade_no']}, Equity: ${info['balance']:,.2f})")

    print("=" * 75)


if __name__ == "__main__":
    main()
