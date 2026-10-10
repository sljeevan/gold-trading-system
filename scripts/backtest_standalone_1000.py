#!/usr/bin/env python3
"""
Standalone 6-Month Backtest of High-Velocity TOUCH Strategy on Gold (XAU/USD).
Initial Capital: $1,000.00
No Compounding Ladder Integration (Pure Strategy Verification).

Evaluates:
1. Fixed Fractional 1.0% Risk per trade
2. Fixed Fractional 2.0% Risk per trade
3. Fixed Fractional 3.0% Risk per trade
4. Fixed Lot Sizing (0.02 Lots)
5. Fixed Lot Sizing (0.05 Lots)
6. Fixed Dollar Risk ($20.00 flat risk per trade)
"""

import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

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


def run_standalone_backtest(
    candles: List[Dict[str, Any]],
    initial_balance: float = 1000.0,
    sizing_mode: str = "PCT_RISK",  # "PCT_RISK", "FIXED_LOT", "FIXED_DOLLAR"
    sizing_param: float = 1.0,     # 1.0 = 1.0% risk, or 0.05 lots, or $20 risk
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

    balance = initial_balance
    peak_balance = balance
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    wins = 0
    losses = 0
    scratches = 0
    gross_profit = 0.0
    gross_loss = 0.0

    buy_trades = 0
    sell_trades = 0
    buy_wins = 0
    sell_wins = 0

    durations = []
    win_pnls = []
    loss_pnls = []

    consecutive_losses = 0
    consecutive_wins = 0
    max_consec_wins = 0
    max_consec_losses = 0

    active_until = -1
    trades = []
    equity_curve = [(dates[200], balance)]
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

        # Position Sizing (No Compounding Ladder)
        if sizing_mode == "PCT_RISK":
            risk_usd = balance * (sizing_param / 100.0)
            lots = max(0.01, min(50.0, round((risk_usd / (sl_pts * 100.0)), 2)))
        elif sizing_mode == "FIXED_LOT":
            lots = sizing_param
            risk_usd = lots * 100.0 * sl_pts
        elif sizing_mode == "FIXED_DOLLAR":
            risk_usd = sizing_param
            lots = max(0.01, min(50.0, round((risk_usd / (sl_pts * 100.0)), 2)))
        else:
            raise ValueError(f"Unknown sizing mode {sizing_mode}")

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

        if direction == "BUY":
            buy_trades += 1
        else:
            sell_trades += 1

        if outcome == "WIN":
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)
            if direction == "BUY":
                buy_wins += 1
            else:
                sell_wins += 1

            trade_pnl = tp_pts * dollar_per_pt
            gross_profit += trade_pnl
            win_pnls.append(trade_pnl)
            balance += trade_pnl

        elif outcome == "SCRATCH":
            scratches += 1
            # Slipped/breakeven scratch P&L is 0

        else:  # LOSS
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)

            trade_pnl = -sl_pts * dollar_per_pt
            gross_loss += abs(trade_pnl)
            loss_pnls.append(abs(trade_pnl))
            balance = max(0.0, balance - abs(trade_pnl))

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

        equity_curve.append((dt, round(balance, 2)))
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
            "balance": round(balance, 2),
            "bars_held": bars_held,
        })

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    all_win_rate = (wins / len(trades) * 100.0) if len(trades) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    trades_per_month = len(trades) / 6.0
    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    avg_duration = float(np.mean(durations)) if durations else 0.0

    return {
        "sizing_mode": sizing_mode,
        "sizing_param": sizing_param,
        "initial_balance": initial_balance,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - initial_balance, 2),
        "return_pct": round(((balance - initial_balance) / initial_balance) * 100.0, 2),
        "peak_balance": round(peak_balance, 2),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "total_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "decided": decided,
        "win_rate": round(win_rate, 2),
        "all_win_rate": round(all_win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "avg_duration_bars": round(avg_duration, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "buy_trades": buy_trades,
        "buy_win_rate": round((buy_wins / (buy_trades - scratches * (buy_trades / len(trades))) * 100.0), 1) if buy_trades > 0 else 0.0,
        "sell_trades": sell_trades,
        "sell_win_rate": round((sell_wins / (sell_trades - scratches * (sell_trades / len(trades))) * 100.0), 1) if sell_trades > 0 else 0.0,
        "monthly_stats": dict(monthly_stats),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def main():
    print("=" * 110)
    print("📊 6-MONTH STANDALONE BACKTEST: HIGH-VELOCITY TOUCH STRATEGY ($1,000 INITIAL CAPITAL)")
    print("   NO COMPOUNDING LADDER — PURE STRATEGY VERIFICATION")
    print("=" * 110)

    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    print(f"Historical Data: {len(candles)} 1-Hour candles ({candles[0]['date']} to {candles[-1]['date']})")
    print(f"Spread: ${SPREAD_USD}/oz | ATR Stop: >= $8.00 (1.5x ATR) | Target: 1.25R | BE: +1.0R | Hours: 07-19 UTC\n")

    configs = [
        {"name": "Fixed Lot: 0.01 Lot (Micro)", "mode": "FIXED_LOT", "param": 0.01},
        {"name": "Fixed Lot: 0.02 Lots (Conservative)", "mode": "FIXED_LOT", "param": 0.02},
        {"name": "Fixed Lot: 0.05 Lots (Balanced)", "mode": "FIXED_LOT", "param": 0.05},
        {"name": "Fixed Lot: 0.10 Lots (Aggressive)", "mode": "FIXED_LOT", "param": 0.10},
        {"name": "Dynamic Equity: 1.5% Risk", "mode": "PCT_RISK", "param": 1.5},
        {"name": "Dynamic Equity: 2.0% Risk", "mode": "PCT_RISK", "param": 2.0},
        {"name": "Dynamic Equity: 3.0% Risk", "mode": "PCT_RISK", "param": 3.0},
        {"name": "Dynamic Equity: 5.0% Risk", "mode": "PCT_RISK", "param": 5.0},
        {"name": "Fixed Dollar: $25 / Trade", "mode": "FIXED_DOLLAR", "param": 25.0},
        {"name": "Fixed Dollar: $50 / Trade", "mode": "FIXED_DOLLAR", "param": 50.0},
    ]

    results = []
    for cfg in configs:
        r = run_standalone_backtest(
            candles,
            initial_balance=1000.0,
            sizing_mode=cfg["mode"],
            sizing_param=cfg["param"],
            donch_period=10,
            target_rr=1.25,
            session_hours=(7, 19),
        )
        r["label"] = cfg["name"]
        results.append(r)

    print(f"{'#':<3} {'Position Sizing Model':<30} {'Trades':<8} {'Win Rate':<10} {'PF':<6} {'Ending Bal':<12} {'Net P&L':<11} {'ROI %':<9} {'Max DD %':<9}")
    print("-" * 110)
    for idx, r in enumerate(results, 1):
        net_str = f"${r['net_profit']:+,.2f}"
        print(f"{idx:<3} {r['label']:<30} {r['total_trades']:<8} {r['win_rate']:<9.1f}% {r['profit_factor']:<6.2f} ${r['ending_balance']:<11,.2f} {net_str:<12} {r['return_pct']:<8.1f}% {r['max_dd_pct']:<8.1f}%")
    print("=" * 110)

    # Detailed breakdown of key models: Fixed 0.02 lot, Fixed 0.05 lot, Dynamic 2.0% Risk, Dynamic 3.0% Risk
    for target_idx in (1, 2, 5, 6):
        r = results[target_idx]
        print(f"\n" + "=" * 80)
        print(f"🔍 IN-DEPTH AUDIT: {r['label'].upper()}")
        print("=" * 80)
        print(f"Starting Capital:     $1,000.00")
        print(f"Ending Balance:       ${r['ending_balance']:,.2f} ({r['return_pct']:+.1f}% ROI)")
        print(f"Peak Balance:         ${r['peak_balance']:,.2f}")
        print(f"Net Profit:           ${r['net_profit']:+,.2f}")
        print(f"Profit Factor:        {r['profit_factor']:.2f} (Gross Profit: ${r['gross_profit']:,.2f} | Gross Loss: ${r['gross_loss']:,.2f})")
        print(f"Total Executions:     {r['total_trades']} (~{r['trades_per_month']:.1f} trades / month)")
        print(f"Win/Loss/BE Record:   {r['wins']} Wins / {r['losses']} Losses / {r['scratches']} Scratches (Break-Even)")
        print(f"Decided Win Rate:     {r['win_rate']:.1f}% (All Win Rate: {r['all_win_rate']:.1f}%)")
        print(f"Avg Win / Avg Loss:   ${r['avg_win']:.2f} / ${r['avg_loss']:.2f} (Payoff Ratio: {r['payoff_ratio']:.2f}:1)")
        print(f"Avg Holding Duration: {r['avg_duration_bars']:.1f} hours ({r['avg_duration_bars']/24:.1f} days)")
        print(f"Max Consecutive:      {r['max_consec_wins']} Wins | {r['max_consec_losses']} Losses")
        print(f"Maximum Drawdown:     ${r['max_dd_dollars']:,.2f} ({r['max_dd_pct']:.2f}%)")

        print(f"\n  📅 Month-by-Month Performance Breakdown:")
        print(f"  {'Month':<10} {'Trades':<8} {'Record (W-L-BE)':<18} {'Win Rate':<10} {'P&L ($)':<12} {'End Balance':<12}")
        print(f"  {'-'*70}")
        bal_running = 1000.0
        for m_key, m_stat in sorted(r["monthly_stats"].items()):
            dec = m_stat["wins"] + m_stat["losses"]
            m_wr = (m_stat["wins"] / dec * 100.0) if dec > 0 else 0.0
            bal_running += m_stat["pnl"]
            pnl_str = f"${m_stat['pnl']:+,.2f}"
            print(f"  {m_key:<10} {m_stat['trades']:<8} {m_stat['wins']}W - {m_stat['losses']}L - {m_stat['scratches']}BE     {m_wr:<9.1f}% {pnl_str:<12} ${bal_running:<11,.2f}")


if __name__ == "__main__":
    main()
