#!/usr/bin/env python3
"""
Comprehensive 1-Year Backtest of Current Production Strategy on Bitcoin (BTC/USD).
Evaluates 12 Full Months (October 2025 – October 2026, 8,730 1-Hour Candles).

Setup Evaluated:
- 1H Donchian 10 Intrabar Touch + SuperTrend (10, 3.0) + 50 EMA Filter
- Institutional Two-Stage Execution (50% scale-out @ +1.0R, BE+$30 runner trailing w/ 1.0x ATR buffer to 2.5R)
- Noise-Adaptive Compounding Ladder ($1,000 Starting Seed)
- Realistic Broker Conditions: $20.00/BTC Spread, Dynamic ATR Stop (1.5x ATR, min $500), 2-Loss Circuit Breaker

Scenarios Compared:
1. 24/7 Compounding Ladder (Full Crypto Availability, $1,000 Seed)
2. Core Volatility Hours Session Filter (07:00–21:00 UTC, $1,000 Seed)
3. Benchmark: Flat Risk Control (0.02 BTC Flat, No Compounding)
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

# Ensure project root in PYTHONPATH
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

SPREAD_USD = 20.0  # $20.00 per BTC typical XM / institutional spread


def simulate_btc_strategy(
    candles: List[Dict[str, Any]],
    mode: str = "TWO_STAGE",  # "TWO_STAGE" or "FLAT"
    start_balance: float = 1000.0,
    session_filter: bool = False,  # False for 24/7, True for 07-21 UTC
    min_sl_dollars: float = 500.0,
) -> Dict[str, Any]:
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

    balance = start_balance
    peak_balance = balance
    current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)

    ratchet_floor = 0.0
    if mode != "FLAT":
        if balance >= 8000.0:
            ratchet_floor = 6000.0
        elif balance >= 5000.0:
            ratchet_floor = 4000.0
        elif balance >= 4000.0:
            ratchet_floor = 3000.0
        elif balance >= 2500.0:
            ratchet_floor = 1800.0
        elif balance >= 1000.0:
            ratchet_floor = 700.0

    consecutive_losses = 0
    consecutive_wins = 0
    max_consec_wins = 0
    max_consec_losses = 0
    wins, losses, scratches = 0, 0, 0
    max_dd_pct = 0.0
    active_until = -1
    trades = []
    monthly_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "scratches": 0, "pnl": 0.0})

    for i in range(200, n - 48):
        if i <= active_until:
            continue

        dt = dates[i]
        month_key = dt[:7]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])

        if session_filter and not (7 <= hour <= 21):
            continue

        d_up = dc["upper"][i - 1]
        d_dn = dc["lower"][i - 1]
        st_dir = st["direction"][i]
        e_trend = ema50[i]
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

        # Progressive Ratchet Floor update
        if mode != "FLAT":
            if balance >= 8000.0:
                ratchet_floor = max(ratchet_floor, 6000.0)
            elif balance >= 5000.0:
                ratchet_floor = max(ratchet_floor, 4000.0)
            elif balance >= 4000.0:
                ratchet_floor = max(ratchet_floor, 3000.0)
            elif balance >= 2500.0:
                ratchet_floor = max(ratchet_floor, 1800.0)
            elif balance >= 1000.0:
                ratchet_floor = max(ratchet_floor, 700.0)

        sl_pts = max(round(1.5 * atr_val, 2), min_sl_dollars)

        if mode == "FLAT":
            lots = 0.02  # 0.02 BTC flat
        else:
            lvl_info = get_noise_adaptive_level_info(current_level)
            planned_risk_usd = lvl_info["risk_dollars"]
            if consecutive_losses >= 2:
                planned_risk_usd *= 0.50
            raw_lots = planned_risk_usd / sl_pts
            lots = max(0.01, min(10.0, math.floor(raw_lots * 100.0 + 0.0001) / 100.0))

        eff_entry = entry_price + SPREAD_USD if direction == "BUY" else entry_price
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + SPREAD_USD

        bars_held = 0
        trade_pnl = 0.0
        outcome = "TIME_EXIT"
        exit_reason = "TIME_EXPIRY"

        runner_buffer = round(1.0 * atr_val, 2)
        t1_pts = round(sl_pts * 1.0, 2)
        t1_price = eff_entry + t1_pts if direction == "BUY" else eff_entry - t1_pts - SPREAD_USD
        t2_pts = round(sl_pts * 2.50, 2)
        t2_price = eff_entry + t2_pts if direction == "BUY" else eff_entry - t2_pts - SPREAD_USD

        stage1_hit = False
        stage1_lots = round(lots * 0.5, 2) if lots >= 0.02 else 0.0
        runner_lots = round(lots - stage1_lots, 2)
        realized_stage1 = 0.0
        peak_p = eff_entry
        curr_runner_sl = sl_price

        for k in range(i + 1, min(i + 72, n)):
            bars_held = k - i
            h_k, l_k = highs[k], lows[k]

            if direction == "BUY":
                peak_p = max(peak_p, h_k)
                if l_k <= curr_runner_sl:
                    if not stage1_hit:
                        outcome = "LOSS"
                        exit_reason = "STOP_LOSS"
                        trade_pnl = -sl_pts * lots
                    else:
                        runner_pnl = (curr_runner_sl - eff_entry) * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN" if trade_pnl > 1.0 else "SCRATCH"
                        exit_reason = "RUNNER_TRAILED_STOP" if curr_runner_sl > eff_entry + 30.0 else "BE_RUNNER"
                    break

                if not stage1_hit and h_k >= t1_price:
                    stage1_hit = True
                    if stage1_lots > 0:
                        realized_stage1 = t1_pts * stage1_lots
                    curr_runner_sl = max(curr_runner_sl, eff_entry + 30.0)

                if stage1_hit:
                    trailed_sl = round(peak_p - runner_buffer, 2)
                    if trailed_sl >= curr_runner_sl + 30.0 and trailed_sl >= eff_entry + 30.0:
                        curr_runner_sl = trailed_sl
                    if h_k >= t2_price:
                        runner_pnl = t2_pts * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN"
                        exit_reason = "TARGET_2_FULL_RUNNER"
                        break
            else:
                peak_p = min(peak_p, l_k)
                if h_k >= curr_runner_sl:
                    if not stage1_hit:
                        outcome = "LOSS"
                        exit_reason = "STOP_LOSS"
                        trade_pnl = -sl_pts * lots
                    else:
                        runner_pnl = (eff_entry - curr_runner_sl) * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN" if trade_pnl > 1.0 else "SCRATCH"
                        exit_reason = "RUNNER_TRAILED_STOP" if curr_runner_sl < eff_entry - 30.0 else "BE_RUNNER"
                    break

                if not stage1_hit and l_k <= t1_price:
                    stage1_hit = True
                    if stage1_lots > 0:
                        realized_stage1 = t1_pts * stage1_lots
                    curr_runner_sl = min(curr_runner_sl, eff_entry - 30.0)

                if stage1_hit:
                    trailed_sl = round(peak_p + runner_buffer, 2)
                    if trailed_sl <= curr_runner_sl - 30.0 and trailed_sl <= eff_entry - 30.0:
                        curr_runner_sl = trailed_sl
                    if l_k <= t2_price:
                        runner_pnl = t2_pts * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN"
                        exit_reason = "TARGET_2_FULL_RUNNER"
                        break

        active_until = i + bars_held
        balance = round(balance + trade_pnl, 2)
        peak_balance = max(peak_balance, balance)
        cur_dd_pct = ((peak_balance - balance) / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_dd_pct = max(max_dd_pct, cur_dd_pct)

        if outcome == "WIN":
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            if mode != "FLAT":
                current_level = min(current_level + 1, len(NOISE_ADAPTIVE_MATRIX))
        elif outcome == "LOSS":
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            if mode != "FLAT":
                step_lvl = max(current_level - 1, 1)
                floor_lvl = find_closest_level_for_balance(ratchet_floor, matrix=NOISE_ADAPTIVE_MATRIX) if ratchet_floor > 0 else 1
                current_level = max(step_lvl, floor_lvl)
        else:
            scratches += 1

        max_consec_wins = max(max_consec_wins, consecutive_wins)
        max_consec_losses = max(max_consec_losses, consecutive_losses)

        monthly_stats[month_key]["trades"] += 1
        if outcome == "WIN":
            monthly_stats[month_key]["wins"] += 1
        elif outcome == "LOSS":
            monthly_stats[month_key]["losses"] += 1
        else:
            monthly_stats[month_key]["scratches"] += 1
        monthly_stats[month_key]["pnl"] += trade_pnl
        monthly_stats[month_key]["end_balance"] = balance

        trades.append({
            "trade_no": len(trades) + 1,
            "date": dt,
            "direction": direction,
            "lots": lots,
            "entry": round(entry_price, 2),
            "outcome": outcome,
            "exit_reason": exit_reason,
            "pnl": round(trade_pnl, 2),
            "balance": balance,
            "bars_held": bars_held,
        })

    decided = wins + losses
    wr = (wins / decided * 100.0) if decided > 0 else 0.0
    win_sum = sum(t["pnl"] for t in trades if t["outcome"] == "WIN")
    loss_sum = sum(t["pnl"] for t in trades if t["outcome"] == "LOSS")
    pf = (win_sum / abs(loss_sum)) if abs(loss_sum) > 0 else float("inf")

    return {
        "mode": mode,
        "session_filter": session_filter,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "starting_balance": start_balance,
        "ending_balance": balance,
        "net_profit": round(balance - start_balance, 2),
        "roi_pct": round((balance - start_balance) / start_balance * 100.0, 1),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "trades": trades,
        "monthly_stats": dict(monthly_stats),
    }


def main():
    print("=" * 115)
    print("🌟 COMPREHENSIVE 1-YEAR BACKTEST: BITCOIN (BTC/USD)")
    print("   Engine: 1H Donchian 10 Touch + SuperTrend (10, 3.0) + 50 EMA Filter")
    print("   Execution: Institutional Two-Stage (50% scale-out @ +1.0R, BE+$30 trailing runner to 2.5R)")
    print("   Protections: Dynamic ATR Stop (1.5x ATR, min $500) + XM MT5 Spread ($20/BTC) + 2-Loss Circuit Breaker")
    print("=" * 115)

    candles = fetch_gold_candles("BTC-USD", period="1y", interval="1h")
    print(f"Dataset Size: {len(candles)} 1H candles | Range: {candles[0]['date']} to {candles[-1]['date']}\n")

    scenarios = [
        {
            "id": "BTC_247_COMPOUNDING",
            "name": "BTC 24/7 Compounding Two-Stage ($1,000 Seed)",
            "mode": "TWO_STAGE",
            "session_filter": False,
            "balance": 1000.0,
        },
        {
            "id": "BTC_SESSION_COMPOUNDING",
            "name": "BTC Liquid Hours (07-21 UTC) Two-Stage ($1,000 Seed)",
            "mode": "TWO_STAGE",
            "session_filter": True,
            "balance": 1000.0,
        },
        {
            "id": "BTC_FLAT_BENCHMARK",
            "name": "BTC Flat Risk Control (0.02 BTC Flat, No Compounding)",
            "mode": "FLAT",
            "session_filter": False,
            "balance": 1000.0,
        },
    ]

    all_results = {}
    print(f"{'#':<2} {'Scenario Description':<52} {'Trades':<7} {'WinRate':<8} {'PF':<6} {'Ending Bal':<13} {'Net P&L':<12} {'ROI %':<8} {'Max DD%':<8}")
    print("-" * 115)

    for idx, sc in enumerate(scenarios, 1):
        res = simulate_btc_strategy(candles, mode=sc["mode"], start_balance=sc["balance"], session_filter=sc["session_filter"])
        res["name"] = sc["name"]
        all_results[sc["id"]] = res
        net_str = f"${res['net_profit']:+,.2f}"
        print(f"{idx:<2} {sc['name']:<52} {res['total_trades']:<7} {res['win_rate']:<7.1f}% {res['profit_factor']:<6.2f} ${res['ending_balance']:<12,.2f} {net_str:<12} {res['roi_pct']:<7.1f}% {res['max_dd_pct']:<7.1f}%")

    print("=" * 115)

    save_path = "/home/mcpuser/.gold_trading_data/bitcoin_compounding_touch_1year_backtest_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Telemetry persisted to: {save_path}")


if __name__ == "__main__":
    main()
