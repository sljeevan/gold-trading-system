#!/usr/bin/env python3
"""
1-Year Backtest of High-Velocity TOUCH Strategy on the Compounding Ladder Architecture.
Evaluates 12 Full Months (October 2025 – October 2026, 5,701 1-Hour Candles).
Compounding Ladder: Noise-Adaptive 37-Level Matrix + Progressive Ratchet Floors + 2-Loss Circuit Breaker.

Tests:
1. Starting Seed: $1,000.00 (Level 19 Initialization)
2. Starting Seed: $200.00   (Level 5 Initialization - Live Daemon Setting)
3. Starting Seed: $100.00   (Level 1 Initialization - Base Matrix)
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
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)

SPREAD_USD = 0.35  # $0.35/oz XM MT5 live spread


def run_compounding_ladder_1year(
    candles: List[Dict[str, Any]],
    start_balance: float = 1000.0,
    donch_period: int = 10,
    target_rr: float = 1.25,
    session_hours: tuple = (7, 19),
) -> Dict[str, Any]:
    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles])
    lows = np.array([float(c["low"]) for c in candles])
    closes = np.array([float(c["close"]) for c in candles])
    opens = np.array([float(c["open"]) for c in candles])
    dates = [c["date"] for c in candles]

    dc = calc_donchian(highs.tolist(), lows.tolist(), donch_period)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema50 = calc_ema(closes.tolist(), 50)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi14 = calc_rsi(closes.tolist(), 14)

    balance = start_balance
    peak_balance = balance
    current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)
    start_level = current_level
    max_level_reached = current_level
    ratchet_floor = 0.0

    # Initial floor check based on starting balance
    if balance >= 4000.0:
        ratchet_floor = 3000.0
    elif balance >= 2500.0:
        ratchet_floor = 1800.0
    elif balance >= 1000.0:
        ratchet_floor = 700.0
    elif balance >= 500.0:
        ratchet_floor = 350.0
    elif balance >= 250.0:
        ratchet_floor = 150.0

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
        if not (session_hours[0] <= hour <= session_hours[1]):
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

        lvl_info = get_noise_adaptive_level_info(current_level)
        planned_risk_usd = lvl_info["risk_dollars"]

        # Circuit breaker: cut risk by 50% after 2 consecutive losses
        cb_active = (consecutive_losses >= 2)
        if cb_active:
            planned_risk_usd *= 0.50

        # Dynamic ATR stop loss ($8.00 minimum noise clearance)
        sl_pts = max(round(1.5 * atr_val, 2), 8.00)
        tp_pts = round(sl_pts * target_rr, 2)
        be_trigger_pts = round(sl_pts * 1.0, 2)  # Delayed break-even at +1.0R

        lots = max(0.01, min(50.0, math.floor((planned_risk_usd / (sl_pts * 100.0)) * 100.0 + 0.0001) / 100.0))
        dollar_per_pt = lots * 100.0

        eff_entry = entry_price + SPREAD_USD if direction == "BUY" else entry_price
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + SPREAD_USD
        tp_price = eff_entry + tp_pts if direction == "BUY" else eff_entry - tp_pts - SPREAD_USD
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

        else:  # LOSS
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

        # Track milestone hits
        for m in [250, 350, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
            if balance >= m and m not in milestones:
                milestones[m] = {"date": dt, "trade_no": len(trades) + 1, "balance": round(balance, 2), "level": current_level}

        trades.append({
            "trade_no": len(trades) + 1,
            "date": dt,
            "direction": direction,
            "entry": round(entry_price, 2),
            "level": lvl_info["level"],
            "phase": lvl_info["phase"],
            "lots": lots,
            "outcome": outcome,
            "pnl": round(trade_pnl, 2),
            "balance": round(balance, 2),
            "ratchet_floor": ratchet_floor,
            "circuit_breaker": cb_active,
        })

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    profit_factor = (gross_p / gross_l) if gross_l > 0 else 999.0

    return {
        "start_balance": start_balance,
        "start_level": start_level,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - start_balance, 2),
        "roi_pct": round(((balance - start_balance) / start_balance) * 100.0, 2),
        "peak_balance": round(peak_balance, 2),
        "max_level_reached": max_level_reached,
        "ending_level": current_level,
        "ratchet_floor": ratchet_floor,
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "total_trades": len(trades),
        "trades_per_month": round(len(trades) / 12.0, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit": round(gross_p, 2),
        "gross_loss": round(gross_l, 2),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "milestones": milestones,
        "monthly_stats": dict(monthly_stats),
        "trades": trades,
    }


def main():
    print("=" * 115)
    print("🪜 1-YEAR COMPOUNDING LADDER BACKTEST: HIGH-VELOCITY TOUCH STRATEGY")
    print("   Dataset: 5,701 1-Hour Candles (October 06, 2025 to October 06, 2026)")
    print("   Architecture: 37-Level Noise-Adaptive Matrix + Progressive Ratchet Floors + 2-Loss Circuit Breaker")
    print("=" * 115)

    candles = fetch_gold_candles("XAUUSD", period="1y", interval="1h")
    print(f"Candles Loaded: {len(candles)} 1H bars ({candles[0]['date']} to {candles[-1]['date']})\n")

    seeds = [
        {"name": "$1,000 Seed (Level 19 - Expansion Phase)", "bal": 1000.0},
        {"name": "$200 Seed   (Level 5 - Live Daemon Sprint)", "bal": 200.0},
        {"name": "$100 Seed   (Level 1 - Base Matrix Sprint)", "bal": 100.0},
    ]

    results = []
    for s in seeds:
        res = run_compounding_ladder_1year(candles, start_balance=s["bal"])
        res["label"] = s["name"]
        results.append(res)

    print(f"{'#':<3} {'Seed & Ladder Setup':<42} {'Trades':<8} {'WinRate':<9} {'PF':<6} {'Peak Lvl':<9} {'End Bal':<12} {'Net P&L':<11} {'ROI %':<9} {'Floor':<10} {'Max DD%':<8}")
    print("-" * 115)
    for idx, r in enumerate(results, 1):
        net_str = f"${r['net_profit']:+,.2f}"
        print(f"{idx:<3} {r['label']:<42} {r['total_trades']:<8} {r['win_rate']:<8.1f}% {r['profit_factor']:<6.2f} L{r['max_level_reached']:<8} ${r['ending_balance']:<11,.2f} {net_str:<11} {r['roi_pct']:<8.1f}% ${r['ratchet_floor']:<9,.0f} {r['max_dd_pct']:<7.1f}%")
    print("=" * 115)

    # Detailed 12-Month Table for $1,000 Seed and $200 Seed
    print("\n" + "=" * 105)
    print("📅 12-MONTH PROGRESSION: $1,000 COMPOUNDING LADDER VS $200 COMPOUNDING LADDER")
    print("=" * 105)
    print(f"{'Month':<10} {'Trades':<8} {'W - L - BE':<14} {'Win Rate':<10} {'$1000 Seed P&L':<16} {'$1000 Balance':<16} {'$200 Seed P&L':<15} {'$200 Balance':<14}")
    print("-" * 105)

    m_keys = sorted(results[0]["monthly_stats"].keys())
    m_1000 = results[0]["monthly_stats"]
    m_200 = results[1]["monthly_stats"]

    bal_1000 = 1000.0
    bal_200 = 200.0

    for mk in m_keys:
        t_stat = results[0]["monthly_stats"][mk]
        dec = t_stat["wins"] + t_stat["losses"]
        wr = (t_stat["wins"] / dec * 100.0) if dec > 0 else 0.0

        pnl_1k = m_1000[mk]["pnl"]
        pnl_200 = m_200[mk]["pnl"]
        bal_1000 += pnl_1k
        bal_200 += pnl_200

        pnl_1k_str = f"${pnl_1k:+,.2f}"
        pnl_200_str = f"${pnl_200:+,.2f}"
        rec_str = f"{t_stat['wins']}W-{t_stat['losses']}L-{t_stat['scratches']}BE"

        print(f"{mk:<10} {t_stat['trades']:<8} {rec_str:<14} {wr:<9.1f}% {pnl_1k_str:<16} ${bal_1000:<15,.2f} {pnl_200_str:<15} ${bal_200:<13,.2f}")

    print("-" * 105)
    tot_1k_str = f"${results[0]['net_profit']:+,.2f}"
    tot_200_str = f"${results[1]['net_profit']:+,.2f}"
    rec_tot = f"{results[0]['wins']}W-{results[0]['losses']}L-{results[0]['scratches']}BE"
    print(f"{'TOTAL':<10} {results[0]['total_trades']:<8} {rec_tot:<14} {results[0]['win_rate']:<9.1f}% {tot_1k_str:<16} ${results[0]['ending_balance']:<15,.2f} {tot_200_str:<15} ${results[1]['ending_balance']:<13,.2f}")
    print("=" * 105)

    # Milestones achieved
    print("\n🏆 MILESTONES ACHIEVED ($1,000 SEED):")
    for m, info in sorted(results[0]["milestones"].items()):
        print(f"  - ${m:>5,}: Hit on {info['date']} (Trade #{info['trade_no']}, Balance: ${info['balance']:,.2f}, Level {info['level']})")

    print("\n🏆 MILESTONES ACHIEVED ($200 SEED):")
    for m, info in sorted(results[1]["milestones"].items()):
        print(f"  - ${m:>5,}: Hit on {info['date']} (Trade #{info['trade_no']}, Balance: ${info['balance']:,.2f}, Level {info['level']})")


if __name__ == "__main__":
    main()
