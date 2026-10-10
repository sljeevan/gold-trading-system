#!/usr/bin/env python3
"""
Detailed Simulation of Option 1 (Donchian 10 TOUCH, 50 EMA, 1.25R Target)
with $200.00 Starting Seed on the Noise-Adaptive Compounding Ladder.
"""

import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.core.compounding_ladder import (
    NOISE_ADAPTIVE_MATRIX,
    find_closest_level_for_balance,
    get_noise_adaptive_level_info,
)
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_atr, calc_donchian, calc_ema, calc_rsi, calc_supertrend


def simulate(candles: List[Dict[str, Any]], start_bal: float = 200.0) -> Dict[str, Any]:
    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles])
    lows = np.array([float(c["low"]) for c in candles])
    closes = np.array([float(c["close"]) for c in candles])
    opens = np.array([float(c["open"]) for c in candles])
    dates = [c["date"] for c in candles]

    dc = calc_donchian(highs.tolist(), lows.tolist(), 10)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema50 = calc_ema(closes.tolist(), 50)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi14 = calc_rsi(closes.tolist(), 14)

    balance = start_bal
    peak_balance = balance
    current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)
    start_level = current_level
    max_level_reached = current_level
    ratchet_floor = 0.0
    consecutive_losses = 0
    consecutive_wins = 0
    max_consec_wins = 0
    max_consec_losses = 0

    wins, losses, scratches = 0, 0, 0
    gross_p, gross_l = 0.0, 0.0
    max_dd_dollars = 0.0
    max_dd_pct = 0.0
    active_until = -1
    trades = []
    milestones = {}
    monthly_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "scratches": 0, "pnl": 0.0})

    for i in range(200, n - 48):
        if i <= active_until:
            continue
        dt = dates[i]
        month_key = dt[:7]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19):
            continue

        d_up = dc["upper"][i - 1]
        d_dn = dc["lower"][i - 1]
        st_dir = st["direction"][i]
        e50 = ema50[i]
        atr_val = atr14[i]
        r = rsi14[i]

        if None in (d_up, d_dn, st_dir, atr_val, r, e50):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        bull_trend = (st_dir == 1) and (c > e50) and (r <= 75.0)
        bear_trend = (st_dir == -1) and (c < e50) and (r >= 25.0)

        direction = None
        entry_price = c

        if (h >= d_up or c >= d_up) and bull_trend:
            direction = "BUY"
            entry_price = max(o, d_up) if o <= d_up else o
        elif (l <= d_dn or c <= d_dn) and bear_trend:
            direction = "SELL"
            entry_price = min(o, d_dn) if o >= d_dn else o

        if not direction:
            continue

        # Progressive Ratchet Floor
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

        lvl_info = get_noise_adaptive_level_info(current_level)
        planned_risk_usd = lvl_info["risk_dollars"]
        if consecutive_losses >= 2:
            planned_risk_usd *= 0.50

        sl_pts = max(round(1.5 * atr_val, 2), 8.00)
        tp_pts = round(sl_pts * 1.25, 2)
        be_trigger_pts = round(sl_pts * 1.0, 2)

        lots = max(0.01, min(50.0, math.floor((planned_risk_usd / (sl_pts * 100.0)) * 100.0 + 0.0001) / 100.0))
        dollar_per_pt = lots * 100.0

        eff_entry = entry_price + 0.35 if direction == "BUY" else entry_price
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + 0.35
        tp_price = eff_entry + tp_pts if direction == "BUY" else eff_entry - tp_pts - 0.35
        be_price = eff_entry + be_trigger_pts if direction == "BUY" else eff_entry - be_trigger_pts

        outcome = "TIME_EXIT"
        is_be = False
        bars_held = 0

        for k in range(i + 1, min(i + 49, n)):
            bars_held = k - i
            h_k, l_k = highs[k], lows[k]
            curr_sl = eff_entry if is_be else sl_price
            if direction == "BUY":
                if l_k <= curr_sl:
                    outcome = "SCRATCH" if is_be else "LOSS"
                    break
                if h_k >= be_price:
                    is_be = True
                if h_k >= tp_price:
                    outcome = "WIN"
                    break
            else:
                if h_k >= curr_sl:
                    outcome = "SCRATCH" if is_be else "LOSS"
                    break
                if l_k <= be_price:
                    is_be = True
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
            gross_p += trade_pnl
            balance += trade_pnl
            if current_level < len(NOISE_ADAPTIVE_MATRIX):
                current_level += 1
            max_level_reached = max(max_level_reached, current_level)
        elif outcome == "SCRATCH":
            scratches += 1
            trade_pnl = 0.0
        else:
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)
            trade_pnl = -sl_pts * dollar_per_pt
            gross_l += abs(trade_pnl)
            balance = max(ratchet_floor, balance - abs(trade_pnl))
            current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)

        peak_balance = max(peak_balance, balance)
        dd_dollars = peak_balance - balance
        dd_pct = (dd_dollars / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_dd_dollars = max(max_dd_dollars, dd_dollars)
        max_dd_pct = max(max_dd_pct, dd_pct)

        monthly_stats[month_key]["trades"] += 1
        if outcome == "WIN":
            monthly_stats[month_key]["wins"] += 1
        elif outcome == "LOSS":
            monthly_stats[month_key]["losses"] += 1
        else:
            monthly_stats[month_key]["scratches"] += 1
        monthly_stats[month_key]["pnl"] += trade_pnl

        for m in [250, 350, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
            if balance >= m and m not in milestones:
                milestones[m] = {
                    "date": dt,
                    "trade_no": len(trades) + 1,
                    "balance": round(balance, 2),
                    "level": current_level,
                }

        trades.append({
            "trade_no": len(trades) + 1,
            "dt": dt,
            "outcome": outcome,
            "lots": lots,
            "pnl": round(trade_pnl, 2),
            "balance": round(balance, 2),
            "level": current_level,
        })

    dec = wins + losses
    return {
        "start_bal": start_bal,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - start_bal, 2),
        "return_pct": round(((balance - start_bal) / start_bal) * 100.0, 1),
        "peak_balance": round(peak_balance, 2),
        "start_level": start_level,
        "max_level_reached": max_level_reached,
        "ending_level": current_level,
        "ratchet_floor": ratchet_floor,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(wins / dec * 100.0, 1) if dec > 0 else 0,
        "profit_factor": round(gross_p / gross_l, 2) if gross_l > 0 else 999.0,
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "milestones": milestones,
        "monthly_stats": dict(monthly_stats),
        "trades": trades,
    }


def main():
    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    r100 = simulate(candles, 100.0)
    r200 = simulate(candles, 200.0)

    print("=" * 80)
    print("🚀 OPTION 1 (TOUCH 1.25R) COMPOUNDING COMPARISON: $100 vs $200 SEED")
    print("=" * 80)
    print(f"{'Metric':<30} {'$100 Starting Seed':<25} {'$200 Starting Seed':<25}")
    print("-" * 80)
    print(f"{'Starting Balance':<30} ${r100['start_bal']:<24,.2f} ${r200['start_bal']:<24,.2f}")
    print(f"{'Starting Ladder Level':<30} Level {r100['start_level']:<20} Level {r200['start_level']:<20}")
    print(f"{'Initial Planned Risk':<30} $10.00 (10.0%)           $17.49 (8.7%) - Level 5")
    print(f"{'Ending Balance':<30} ${r100['ending_balance']:<24,.2f} ${r200['ending_balance']:<24,.2f}")
    print(f"{'Net Profit ($)':<30} +${r100['net_profit']:<23,.2f} +${r200['net_profit']:<23,.2f}")
    print(f"{'Return On Investment':<30} +{r100['return_pct']:<23.1f}% +{r200['return_pct']:<23.1f}%")
    print(f"{'Peak Account Balance':<30} ${r100['peak_balance']:<24,.2f} ${r200['peak_balance']:<24,.2f}")
    print(f"{'Peak Ladder Level':<30} Level {r100['max_level_reached']:<20} Level {r200['max_level_reached']:<20}")
    print(f"{'Ending Ladder Level':<30} Level {r100['ending_level']:<20} Level {r200['ending_level']:<20}")
    print(f"{'Permanently Locked Floor':<30} ${r100['ratchet_floor']:<24,.2f} ${r200['ratchet_floor']:<24,.2f}")
    print(f"{'Total Executions':<30} {r100['total_trades']:<25} {r200['total_trades']:<25}")
    print(f"{'Win Rate (Decided)':<30} {r100['win_rate']:<24.1f}% {r200['win_rate']:<24.1f}%")
    print(f"{'Profit Factor':<30} {r100['profit_factor']:<25.2f} {r200['profit_factor']:<25.2f}")
    print(f"{'Max Loss Streak':<30} {r100['max_consec_losses']:<25} {r200['max_consec_losses']:<25}")
    print(f"{'Max Drawdown ($)':<30} ${r100['max_dd_dollars']:<24,.2f} ${r200['max_dd_dollars']:<24,.2f}")
    print(f"{'Max Drawdown (%)':<30} {r100['max_dd_pct']:<24.1f}% {r200['max_dd_pct']:<24.1f}%")

    print("\n" + "=" * 80)
    print("🏆 MILESTONES ACHIEVED WITH $200 SEED")
    print("=" * 80)
    for m, v in sorted(r200["milestones"].items()):
        print(f"  - ${m:>4} Milestone: Reached on {v['date']} (Trade #{v['trade_no']}, Balance: ${v['balance']:,.2f}, Level {v['level']})")

    print("\n" + "=" * 80)
    print("📅 MONTH-BY-MONTH CAPITAL PROGRESSION ($200 SEED)")
    print("=" * 80)
    print(f"{'Month':<10} {'Trades':<8} {'W - L - BE':<14} {'Win Rate':<10} {'Monthly P&L':<15} {'Ending Balance'}")
    print("-" * 80)
    cum_bal = 200.0
    for k, v in sorted(r200["monthly_stats"].items()):
        dec = v["wins"] + v["losses"]
        wr = (v["wins"] / dec * 100.0) if dec > 0 else 0
        cum_bal += v["pnl"]
        print(f"{k:<10} {v['trades']:<8} {v['wins']:>2}W-{v['losses']:>2}L-{v['scratches']:>2}BE   {wr:>5.1f}%     ${v['pnl']:>+10.2f}     ${cum_bal:>10.2f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
