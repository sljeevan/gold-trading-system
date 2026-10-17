#!/usr/bin/env python3
"""
Comprehensive 1-Year Backtest of Current Production Strategy (COMPOUNDING_TOUCH).
Evaluates 12 Full Months (October 2025 – October 2026, 5,702 1-Hour Candles).

Scenarios Evaluated:
1. Active Production Daemon: $1,000 Seed with Institutional Two-Stage Execution
   (50% scale-out @ +1.0R, BE+$0.50 runner trailing w/ $6.00 buffer to 2.5R)
2. Single-Target Fast TP: $1,000 Seed with 1.25R Target + Delayed BE @ +1.0R
3. Flat Risk Control Benchmark: $1,000 Fixed Capital (0.02 lots flat, Zero Compounding)
4. Sprint Account: $200 Seed (Level 5) with Institutional Two-Stage Execution
5. Micro-Seed Account: $100 Seed (Level 1) with Institutional Two-Stage Execution
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

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


def simulate_strategy_1year(
    candles: List[Dict[str, Any]],
    mode: str = "TWO_STAGE",  # "TWO_STAGE" or "SINGLE_TARGET_1.25R" or "FLAT_0.02"
    start_balance: float = 1000.0,
    donch_period: int = 10,
    trend_filter_ema: int = 50,
    target_rr: float = 1.25,
    session_hours: tuple = (7, 19),
    runner_buffer: float = 6.00,
    runner_t2_multiplier: float = 2.50,
) -> Dict[str, Any]:
    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles])
    lows = np.array([float(c["low"]) for c in candles])
    closes = np.array([float(c["close"]) for c in candles])
    opens = np.array([float(c["open"]) for c in candles])
    dates = [c["date"] for c in candles]

    dc = calc_donchian(highs.tolist(), lows.tolist(), donch_period)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema_trend = calc_ema(closes.tolist(), trend_filter_ema)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi14 = calc_rsi(closes.tolist(), 14)

    is_flat = (mode == "FLAT_0.02")
    balance = start_balance
    peak_balance = balance
    current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)
    start_level = current_level
    max_level_reached = current_level

    # Set initial ratchet floor
    ratchet_floor = 0.0
    if not is_flat:
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
    stage1_wins = 0
    runner_t2_wins = 0
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
        e_trend = ema_trend[i]
        atr_val = atr14[i]
        r = rsi14[i]

        if None in (d_up, d_dn, st_dir, atr_val, r, e_trend):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        bull_trend = (st_dir == 1) and (c > e_trend) and (r <= 75.0)
        bear_trend = (st_dir == -1) and (c < e_trend) and (r >= 25.0)

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

        # Progressive Ratchet Floor update (only in compounding modes)
        if not is_flat:
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

        # Dynamic ATR Stop Loss ($8.00 minimum noise clearance)
        sl_pts = max(round(1.5 * atr_val, 2), 8.00)

        if is_flat:
            lots = 0.02
            planned_risk_usd = sl_pts * lots * 100.0
            cb_active = False
            lvl_info = {"level": 0, "phase": "Flat Benchmark"}
        else:
            lvl_info = get_noise_adaptive_level_info(current_level)
            planned_risk_usd = lvl_info["risk_dollars"]
            cb_active = (consecutive_losses >= 2)
            if cb_active:
                planned_risk_usd *= 0.50
            lots = max(0.01, min(50.0, math.floor((planned_risk_usd / (sl_pts * 100.0)) * 100.0 + 0.0001) / 100.0))

        eff_entry = entry_price + SPREAD_USD if direction == "BUY" else entry_price
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + SPREAD_USD

        bars_held = 0
        trade_pnl = 0.0
        outcome = "TIME_EXIT"
        exit_reason = "TIME_EXPIRY"

        # -----------------------------------------------------------------
        # EXECUTION ENGINE A: SINGLE TARGET (1.25R Fast TP with Delayed BE)
        # -----------------------------------------------------------------
        if mode == "SINGLE_TARGET_1.25R":
            single_tp_pts = round(sl_pts * target_rr, 2)
            single_tp_price = eff_entry + single_tp_pts if direction == "BUY" else eff_entry - single_tp_pts - SPREAD_USD
            be_trigger_pts = round(sl_pts * 1.0, 2)
            be_price = eff_entry + be_trigger_pts if direction == "BUY" else eff_entry - be_trigger_pts
            is_be = False

            for k in range(i + 1, min(i + 49, n)):
                bars_held = k - i
                h_k, l_k = highs[k], lows[k]
                curr_sl = eff_entry if is_be else sl_price

                if direction == "BUY":
                    if l_k <= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_reason = "BREAKEVEN" if is_be else "STOP_LOSS"
                        break
                    if h_k >= be_price:
                        is_be = True
                    if h_k >= single_tp_price:
                        outcome = "WIN"
                        exit_reason = "TAKE_PROFIT_1.25R"
                        break
                else:
                    if h_k >= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_reason = "BREAKEVEN" if is_be else "STOP_LOSS"
                        break
                    if l_k <= be_price:
                        is_be = True
                    if l_k <= single_tp_price:
                        outcome = "WIN"
                        exit_reason = "TAKE_PROFIT_1.25R"
                        break

            if outcome == "WIN":
                trade_pnl = single_tp_pts * lots * 100.0
            elif outcome == "LOSS":
                trade_pnl = -sl_pts * lots * 100.0
            else:
                trade_pnl = 0.0

        # -----------------------------------------------------------------
        # EXECUTION ENGINE B: INSTITUTIONAL TWO-STAGE EXECUTION (Current Production)
        # -----------------------------------------------------------------
        else:
            t1_pts = round(sl_pts * 1.0, 2)
            t1_price = eff_entry + t1_pts if direction == "BUY" else eff_entry - t1_pts - SPREAD_USD
            t2_pts = round(sl_pts * runner_t2_multiplier, 2)
            t2_price = eff_entry + t2_pts if direction == "BUY" else eff_entry - t2_pts - SPREAD_USD

            stage1_hit = False
            stage1_lots = round(lots * 0.5, 2) if lots >= 0.02 else 0.0
            runner_lots = round(lots - stage1_lots, 2)
            realized_stage1 = 0.0
            peak_p = eff_entry
            curr_runner_sl = sl_price
            hit_t2 = False

            for k in range(i + 1, min(i + 49, n)):
                bars_held = k - i
                h_k, l_k = highs[k], lows[k]

                if direction == "BUY":
                    peak_p = max(peak_p, h_k)

                    # Check SL
                    if l_k <= curr_runner_sl:
                        if not stage1_hit:
                            outcome = "LOSS"
                            exit_reason = "STOP_LOSS"
                            trade_pnl = -sl_pts * lots * 100.0
                        else:
                            # Runner stopped out
                            runner_pnl = (curr_runner_sl - eff_entry) * runner_lots * 100.0
                            trade_pnl = realized_stage1 + runner_pnl
                            outcome = "WIN" if trade_pnl > 0.5 else "SCRATCH"
                            exit_reason = "RUNNER_TRAILED_STOP" if curr_runner_sl > eff_entry + 0.60 else "BE_RUNNER"
                        break

                    # Check Target 1 (+1.0R)
                    if not stage1_hit and h_k >= t1_price:
                        stage1_hit = True
                        stage1_wins += 1
                        if stage1_lots > 0:
                            realized_stage1 = t1_pts * stage1_lots * 100.0
                        # Shift runner SL to BE + $0.50
                        curr_runner_sl = max(curr_runner_sl, eff_entry + 0.50)

                    # Manage Runner if T1 hit
                    if stage1_hit:
                        trailed_sl = round(peak_p - runner_buffer, 2)
                        if trailed_sl >= curr_runner_sl + 0.50 and trailed_sl >= eff_entry + 0.50:
                            curr_runner_sl = trailed_sl

                        if h_k >= t2_price:
                            hit_t2 = True
                            runner_t2_wins += 1
                            runner_pnl = t2_pts * runner_lots * 100.0
                            trade_pnl = realized_stage1 + runner_pnl
                            outcome = "WIN"
                            exit_reason = "TARGET_2_FULL_RUNNER"
                            break

                else:  # SELL
                    peak_p = min(peak_p, l_k)

                    if h_k >= curr_runner_sl:
                        if not stage1_hit:
                            outcome = "LOSS"
                            exit_reason = "STOP_LOSS"
                            trade_pnl = -sl_pts * lots * 100.0
                        else:
                            runner_pnl = (eff_entry - curr_runner_sl) * runner_lots * 100.0
                            trade_pnl = realized_stage1 + runner_pnl
                            outcome = "WIN" if trade_pnl > 0.5 else "SCRATCH"
                            exit_reason = "RUNNER_TRAILED_STOP" if curr_runner_sl < eff_entry - 0.60 else "BE_RUNNER"
                        break

                    if not stage1_hit and l_k <= t1_price:
                        stage1_hit = True
                        stage1_wins += 1
                        if stage1_lots > 0:
                            realized_stage1 = t1_pts * stage1_lots * 100.0
                        curr_runner_sl = min(curr_runner_sl, eff_entry - 0.50)

                    if stage1_hit:
                        trailed_sl = round(peak_p + runner_buffer, 2)
                        if trailed_sl <= curr_runner_sl - 0.50 and trailed_sl <= eff_entry - 0.50:
                            curr_runner_sl = trailed_sl

                        if l_k <= t2_price:
                            hit_t2 = True
                            runner_t2_wins += 1
                            runner_pnl = t2_pts * runner_lots * 100.0
                            trade_pnl = realized_stage1 + runner_pnl
                            outcome = "WIN"
                            exit_reason = "TARGET_2_FULL_RUNNER"
                            break

            if outcome == "TIME_EXIT":
                if stage1_hit:
                    trade_pnl = realized_stage1
                    outcome = "WIN" if trade_pnl > 0.5 else "SCRATCH"
                else:
                    trade_pnl = 0.0
                    outcome = "SCRATCH"

        active_until = i + bars_held
        trade_pnl = round(trade_pnl, 2)
        prev_balance = balance

        if outcome == "WIN":
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)
            balance += trade_pnl
            actual_trade_pnl = trade_pnl
            gross_p += actual_trade_pnl
            if not is_flat and current_level < len(NOISE_ADAPTIVE_MATRIX):
                current_level += 1
            max_level_reached = max(max_level_reached, current_level)

        elif outcome == "SCRATCH":
            scratches += 1
            actual_trade_pnl = 0.0

        else:  # LOSS
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)
            if not is_flat:
                balance = max(ratchet_floor, balance - abs(trade_pnl))
                actual_trade_pnl = round(balance - prev_balance, 2)
                current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)
            else:
                balance -= abs(trade_pnl)
                actual_trade_pnl = -abs(trade_pnl)
            gross_l += abs(actual_trade_pnl)

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
        monthly_stats[month_key]["pnl"] = round(monthly_stats[month_key]["pnl"] + actual_trade_pnl, 2)
        monthly_stats[month_key]["end_balance"] = round(balance, 2)

        # Track milestone hits
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
            "date": dt,
            "direction": direction,
            "entry": round(entry_price, 2),
            "level": lvl_info["level"],
            "phase": lvl_info["phase"],
            "lots": lots,
            "outcome": outcome,
            "exit_reason": exit_reason,
            "pnl": round(trade_pnl, 2),
            "balance": round(balance, 2),
            "ratchet_floor": ratchet_floor,
            "circuit_breaker": cb_active,
            "bars_held": bars_held,
        })

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    profit_factor = (gross_p / gross_l) if gross_l > 0 else 999.0

    return {
        "mode": mode,
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
        "stage1_wins": stage1_wins,
        "runner_t2_wins": runner_t2_wins,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "milestones": milestones,
        "monthly_stats": dict(monthly_stats),
        "trades": trades,
    }


def main():
    print("=" * 115)
    print("🌟 COMPREHENSIVE 1-YEAR BACKTEST: CURRENT PRODUCTION STRATEGY (COMPOUNDING_TOUCH)")
    print("   Dataset: Real 1H XAU/USD Historical Candles (October 2025 – October 2026)")
    print("   Engine: 1H Donchian 10 Touch + SuperTrend (10, 3.0) + 50 EMA + Session Filter (07-19 UTC)")
    print("   Protections: Dynamic ATR Stop (min $8.00) + XM MT5 Spread ($0.35/oz) + 2-Loss Circuit Breaker")
    print("=" * 115)

    candles = fetch_gold_candles("XAUUSD", period="1y", interval="1h")
    print(f"Dataset Size: {len(candles)} 1H candles | Range: {candles[0]['date']} to {candles[-1]['date']}\n")

    scenarios = [
        {
            "id": "PROD_TWO_STAGE_1000",
            "name": "Active Production Daemon: $1,000 (Two-Stage Partial Scale-Out + Runner)",
            "mode": "TWO_STAGE",
            "balance": 1000.0,
        },
        {
            "id": "FAST_TP_1.25R_1000",
            "name": "Single Target 1.25R: $1,000 (Fast TP + Delayed BE @ +1.0R)",
            "mode": "SINGLE_TARGET_1.25R",
            "balance": 1000.0,
        },
        {
            "id": "FLAT_CONTROL_1000",
            "name": "Flat Risk Control: $1,000 (0.02 Lots Flat, Zero Compounding)",
            "mode": "FLAT_0.02",
            "balance": 1000.0,
        },
        {
            "id": "PROD_TWO_STAGE_200",
            "name": "Sprint Account: $200 Seed (Level 5 Sprint Initialization)",
            "mode": "TWO_STAGE",
            "balance": 200.0,
        },
        {
            "id": "PROD_TWO_STAGE_100",
            "name": "Micro Seed: $100 Seed (Level 1 Base Matrix Initialization)",
            "mode": "TWO_STAGE",
            "balance": 100.0,
        },
    ]

    all_results = {}
    print(f"{'#':<2} {'Scenario Description':<52} {'Trades':<7} {'WinRate':<8} {'PF':<6} {'Ending Bal':<13} {'Net P&L':<12} {'ROI %':<8} {'Max DD%':<8}")
    print("-" * 115)

    for idx, sc in enumerate(scenarios, 1):
        res = simulate_strategy_1year(candles, mode=sc["mode"], start_balance=sc["balance"])
        res["name"] = sc["name"]
        all_results[sc["id"]] = res
        net_str = f"${res['net_profit']:+,.2f}"
        print(f"{idx:<2} {sc['name']:<52} {res['total_trades']:<7} {res['win_rate']:<7.1f}% {res['profit_factor']:<6.2f} ${res['ending_balance']:<12,.2f} {net_str:<12} {res['roi_pct']:<7.1f}% {res['max_dd_pct']:<7.1f}%")

    print("=" * 115)

    # Save complete backtest telemetry to JSON
    save_path = "/home/mcpuser/.gold_trading_data/compounding_touch_1year_backtest_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Full backtest telemetry persisted to: {save_path}")

    # Detailed 12-Month Progression for Active Daemon
    prod = all_results["PROD_TWO_STAGE_1000"]
    single = all_results["FAST_TP_1.25R_1000"]
    flat = all_results["FLAT_CONTROL_1000"]

    print("\n" + "=" * 115)
    print("📅 12-MONTH PROGRESSION: ACTIVE PRODUCTION DAEMON ($1,000 SEED) VS ALTERNATIVES")
    print("=" * 115)
    print(f"{'Month':<9} {'Trades':<7} {'W - L - BE':<14} {'WR %':<7} {'Prod Daemon P&L':<17} {'Prod Balance':<15} {'Single 1.25R P&L':<17} {'Flat 0.02 P&L':<13}")
    print("-" * 115)

    m_keys = sorted(prod["monthly_stats"].keys())
    running_b = 1000.0
    for mk in m_keys:
        p_st = prod["monthly_stats"][mk]
        s_st = single["monthly_stats"][mk]
        f_st = flat["monthly_stats"][mk]
        dec = p_st["wins"] + p_st["losses"]
        wr = (p_st["wins"] / dec * 100.0) if dec > 0 else 0.0
        running_b += p_st["pnl"]

        p_str = f"${p_st['pnl']:+,.2f}"
        s_str = f"${s_st['pnl']:+,.2f}"
        f_str = f"${f_st['pnl']:+,.2f}"
        rec = f"{p_st['wins']}W-{p_st['losses']}L-{p_st['scratches']}BE"

        print(f"{mk:<9} {p_st['trades']:<7} {rec:<14} {wr:<6.1f}% {p_str:<17} ${p_st['end_balance']:<14,.2f} {s_str:<17} {f_str:<13}")

    print("-" * 115)
    tot_rec = f"{prod['wins']}W-{prod['losses']}L-{prod['scratches']}BE"
    print(f"{'TOTAL':<9} {prod['total_trades']:<7} {tot_rec:<14} {prod['win_rate']:<6.1f}% ${prod['net_profit']:<16,.2f} ${prod['ending_balance']:<14,.2f} ${single['net_profit']:<16,.2f} ${flat['net_profit']:<12,.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
