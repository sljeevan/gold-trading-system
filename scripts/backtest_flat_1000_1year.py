#!/usr/bin/env python3
"""
1-Year Standalone Backtest of High-Velocity TOUCH Strategy on Gold (XAU/USD).
Initial Capital: $1,000.00 Flat per Trade (Zero Compounding / Zero Reinvestment).
Period: October 2025 – October 2026 (12 Full Months, 5,701 1-Hour Candles).
"""

import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)

SPREAD_USD = 0.35  # $0.35/oz XM MT5 live spread


def run_1year_flat_backtest(
    candles: List[Dict[str, Any]],
    flat_capital: float = 1000.0,
    lots: float = 0.02,
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

    accumulated_pnl = 0.0
    running_balance = flat_capital
    peak_balance = flat_capital
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    wins = 0
    losses = 0
    scratches = 0
    gross_profit = 0.0
    gross_loss = 0.0

    durations = []
    win_pnls = []
    loss_pnls = []

    consecutive_losses = 0
    consecutive_wins = 0
    max_consec_wins = 0
    max_consec_losses = 0

    active_until = -1
    trades = []
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

        # Dynamic ATR Stop Loss ($8.00 minimum noise clearance)
        sl_pts = max(round(1.5 * atr_val, 2), 8.00)
        tp_pts = round(sl_pts * target_rr, 2)
        be_trigger_pts = round(sl_pts * 1.0, 2)  # Delay break-even to +1.0R

        dollar_per_pt = lots * 100.0

        eff_entry = entry_price + SPREAD_USD if direction == "BUY" else entry_price
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + SPREAD_USD
        tp_price = eff_entry + tp_pts if direction == "BUY" else eff_entry - tp_pts - SPREAD_USD
        be_price = eff_entry + be_trigger_pts if direction == "BUY" else eff_entry - be_trigger_pts

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
        durations.append(bars_held)
        trade_pnl = 0.0

        if outcome == "WIN":
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)

            trade_pnl = tp_pts * dollar_per_pt
            gross_profit += trade_pnl
            win_pnls.append(trade_pnl)

        elif outcome == "SCRATCH":
            scratches += 1

        else:  # LOSS
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)

            trade_pnl = -sl_pts * dollar_per_pt
            gross_loss += abs(trade_pnl)
            loss_pnls.append(abs(trade_pnl))

        accumulated_pnl += trade_pnl
        running_balance = flat_capital + accumulated_pnl
        peak_balance = max(peak_balance, running_balance)
        dd_dollars = peak_balance - running_balance
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

        trades.append({
            "trade_no": len(trades) + 1,
            "date": dt,
            "direction": direction,
            "entry": round(entry_price, 2),
            "sl_pts": round(sl_pts, 2),
            "tp_pts": round(tp_pts, 2),
            "lots": lots,
            "outcome": outcome,
            "pnl": round(trade_pnl, 2),
            "accum_pnl": round(accumulated_pnl, 2),
            "bars_held": bars_held,
        })

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    months_count = max(1, len(monthly_stats))
    trades_per_month = len(trades) / float(months_count)
    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    avg_duration = float(np.mean(durations)) if durations else 0.0

    return {
        "lots": lots,
        "flat_capital": flat_capital,
        "total_pnl": round(accumulated_pnl, 2),
        "return_pct_on_1000": round((accumulated_pnl / flat_capital) * 100.0, 2),
        "peak_balance": round(peak_balance, 2),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct_of_1000": round((max_dd_dollars / flat_capital) * 100.0, 2),
        "total_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "avg_duration_bars": round(avg_duration, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "monthly_stats": dict(monthly_stats),
        "trades": trades,
    }


def main():
    print("=" * 115)
    print("🗓️ 1-YEAR NON-COMPOUNDING BACKTEST: FLAT $1,000 CAPITAL PER TRADE (ZERO REINVESTMENT)")
    print("   Evaluating 12 Full Months on Spot Gold (XAU/USD) with 1H Donchian 10 Touch + 50 EMA + 1.25R")
    print("=" * 115)

    candles = fetch_gold_candles("XAUUSD", period="1y", interval="1h")
    print(f"Candles Loaded: {len(candles)} 1H bars ({candles[0]['date']} to {candles[-1]['date']})")
    print(f"Execution: Intrabar Donchian 10 Touch + 50 EMA + 1.25R Target | Spread: ${SPREAD_USD}/oz | ATR Stop >= $8.00\n")

    lot_configs = [
        {"name": "0.01 Lot (Micro)", "lots": 0.01},
        {"name": "0.02 Lots (Conservative)", "lots": 0.02},
        {"name": "0.03 Lots (Moderate)", "lots": 0.03},
        {"name": "0.05 Lots (Balanced Standard)", "lots": 0.05},
        {"name": "0.10 Lots (Aggressive)", "lots": 0.10},
    ]

    results = []
    for cfg in lot_configs:
        r = run_1year_flat_backtest(
            candles,
            flat_capital=1000.0,
            lots=cfg["lots"],
            donch_period=10,
            target_rr=1.25,
            session_hours=(7, 19),
        )
        r["label"] = cfg["name"]
        results.append(r)

    print(f"{'#':<3} {'Lot Sizing Model':<30} {'Trades':<8} {'Win Rate':<10} {'PF':<6} {'Total P&L ($)':<15} {'Annual ROI':<12} {'Max DD ($)':<12} {'Max DD %':<9}")
    print("-" * 115)
    for idx, r in enumerate(results, 1):
        pnl_str = f"${r['total_pnl']:+,.2f}"
        roi_str = f"{r['return_pct_on_1000']:+.1f}%"
        print(f"{idx:<3} {r['label']:<30} {r['total_trades']:<8} {r['win_rate']:<9.1f}% {r['profit_factor']:<6.2f} {pnl_str:<15} {roi_str:<12} ${r['max_dd_dollars']:<11,.2f} {r['max_dd_pct_of_1000']:<8.1f}%")
    print("=" * 115)

    # Detailed 12-Month Table
    print("\n" + "=" * 105)
    print("📅 EXACT 12-MONTH P&L BREAKDOWN (FLAT $1,000 CAPITAL, ZERO REINVESTMENT)")
    print("=" * 105)
    print(f"{'Month':<10} {'Trades':<8} {'W - L - BE':<14} {'Win Rate':<10} {'0.01 Lot':<12} {'0.02 Lots':<12} {'0.03 Lots':<12} {'0.05 Lots':<12}")
    print("-" * 105)

    m_keys = sorted(results[0]["monthly_stats"].keys())
    m_01 = results[0]["monthly_stats"]
    m_02 = results[1]["monthly_stats"]
    m_03 = results[2]["monthly_stats"]
    m_05 = results[3]["monthly_stats"]

    for mk in m_keys:
        t_stat = results[0]["monthly_stats"][mk]
        dec = t_stat["wins"] + t_stat["losses"]
        wr = (t_stat["wins"] / dec * 100.0) if dec > 0 else 0.0
        pnl_01_str = f"${m_01[mk]['pnl']:+,.2f}"
        pnl_02_str = f"${m_02[mk]['pnl']:+,.2f}"
        pnl_03_str = f"${m_03[mk]['pnl']:+,.2f}"
        pnl_05_str = f"${m_05[mk]['pnl']:+,.2f}"
        rec_str = f"{t_stat['wins']}W-{t_stat['losses']}L-{t_stat['scratches']}BE"
        print(f"{mk:<10} {t_stat['trades']:<8} {rec_str:<14} {wr:<9.1f}% {pnl_01_str:<12} {pnl_02_str:<12} {pnl_03_str:<12} {pnl_05_str:<12}")

    print("-" * 105)
    tot_01_str = f"${results[0]['total_pnl']:+,.2f}"
    tot_02_str = f"${results[1]['total_pnl']:+,.2f}"
    tot_03_str = f"${results[2]['total_pnl']:+,.2f}"
    tot_05_str = f"${results[3]['total_pnl']:+,.2f}"
    rec_tot = f"{results[0]['wins']}W-{results[0]['losses']}L-{results[0]['scratches']}BE"
    print(f"{'TOTAL':<10} {results[0]['total_trades']:<8} {rec_tot:<14} {results[0]['win_rate']:<9.1f}% {tot_01_str:<12} {tot_02_str:<12} {tot_03_str:<12} {tot_05_str:<12}")
    print("=" * 105)


if __name__ == "__main__":
    main()
