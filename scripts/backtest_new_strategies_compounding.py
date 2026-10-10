#!/usr/bin/env python3
"""
Comprehensive 6-Month Backtest of New Trading Strategies on the Compounding Ladder.
Evaluates high-frequency variants against the baseline on real 1H Gold data (XAU/USD):
- Seed: $100.00
- Compounding Ladder: NOISE_ADAPTIVE_MATRIX (37 discrete levels, $100 -> $5,337)
- Capital Protection: Progressive Ratchet Floors ($150, $350, $700, $1800, $3000)
- Risk Management: Dynamic ATR lot sizing (min $8 stop) + 2-loss circuit breaker (50% risk cut)
- Broker Spread: $0.35/oz realistic XM MT5 spread
- Delayed Break-Even: Trailing SL moved to BE at +1.0R
"""

import math
import os
import sys
from collections import defaultdict
from datetime import datetime
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

SPREAD_USD = 0.35  # $0.35/oz live spread


def run_strategy_ladder_backtest(
    candles: List[Dict[str, Any]],
    name: str,
    donch_period: int = 10,
    target_rr: float = 1.25,
    trigger_mode: str = "TOUCH",  # "TOUCH" or "CLOSE"
    trend_filter: str = "EMA50",  # "EMA50" or "EMA200"
    allow_ema20_pullback: bool = False,
    session_hours: tuple = (7, 19),  # (start_hour, end_hour inclusive)
) -> Dict[str, Any]:
    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles])
    lows = np.array([float(c["low"]) for c in candles])
    closes = np.array([float(c["close"]) for c in candles])
    opens = np.array([float(c["open"]) for c in candles])
    dates = [c["date"] for c in candles]

    dc = calc_donchian(highs.tolist(), lows.tolist(), donch_period)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi14 = calc_rsi(closes.tolist(), 14)

    # State tracking
    balance = 100.0
    peak_balance = balance
    current_level = 1
    max_level_reached = 1
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
        e20 = ema20[i]
        e50 = ema50[i]
        e200 = ema200[i]
        atr_val = atr14[i]
        r = rsi14[i]

        if None in (d_up, d_dn, st_dir, atr_val, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        # Trend filter evaluation
        if trend_filter == "EMA200":
            bull_trend = (st_dir == 1) and (e200 is not None and c > e200) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (e200 is not None and c < e200) and (r >= 25.0)
        else:  # EMA50
            bull_trend = (st_dir == 1) and (e50 is not None and c > e50) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (e50 is not None and c < e50) and (r >= 25.0)

        direction = None
        trigger_desc = None
        entry_price = c

        if trigger_mode == "CLOSE":
            if c > d_up and bull_trend and (c > o):
                direction = "BUY"
                trigger_desc = f"Donchian-{donch_period} Breakout Close"
                entry_price = c
            elif c < d_dn and bear_trend and (c < o):
                direction = "SELL"
                trigger_desc = f"Donchian-{donch_period} Breakdown Close"
                entry_price = c
        elif trigger_mode == "TOUCH":
            if (h >= d_up or c >= d_up) and bull_trend:
                direction = "BUY"
                trigger_desc = f"Donchian-{donch_period} Touch"
                entry_price = max(o, d_up) if o <= d_up else o
            elif (l <= d_dn or c <= d_dn) and bear_trend:
                direction = "SELL"
                trigger_desc = f"Donchian-{donch_period} Touch"
                entry_price = min(o, d_dn) if o >= d_dn else o

        # Optional EMA 20 Pullback trigger
        if not direction and allow_ema20_pullback and e20 is not None:
            if bull_trend and l <= e20 and c > e20 and (c > o) and r >= 48.0:
                direction = "BUY"
                trigger_desc = "EMA20 Pullback"
                entry_price = c
            elif bear_trend and h >= e20 and c < e20 and (c < o) and r <= 52.0:
                direction = "SELL"
                trigger_desc = "EMA20 Pullback"
                entry_price = c

        if not direction:
            continue

        # Progressive Ratchet Floor updates
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

        # Circuit breaker: halve risk after 2 consecutive losses
        if consecutive_losses >= 2:
            planned_risk_usd *= 0.50

        # ATR Stop Loss ($8.00 minimum noise threshold)
        sl_pts = max(round(1.5 * atr_val, 2), 8.00)
        tp_pts = round(sl_pts * target_rr, 2)
        be_trigger_pts = round(sl_pts * 1.0, 2)  # Delay break-even to +1.0R

        lots = max(0.01, min(50.0, math.floor((planned_risk_usd / (sl_pts * 100.0)) * 100.0 + 0.0001) / 100.0))
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
        trade_pnl = 0.0

        if outcome == "WIN":
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)

            trade_pnl = tp_pts * dollar_per_pt
            gross_profit += trade_pnl
            balance += trade_pnl

            # Compounding ladder progression: advance 1 level on win
            if current_level < len(NOISE_ADAPTIVE_MATRIX):
                current_level += 1
            max_level_reached = max(max_level_reached, current_level)

        elif outcome == "SCRATCH":
            scratches += 1
            trade_pnl = 0.0  # Zero loss scratch
            # consecutive_losses not incremented on scratch

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

        # Monthly tracking
        monthly_stats[month_key]["trades"] += 1
        if outcome == "WIN":
            monthly_stats[month_key]["wins"] += 1
        elif outcome == "LOSS":
            monthly_stats[month_key]["losses"] += 1
        else:
            monthly_stats[month_key]["scratches"] += 1
        monthly_stats[month_key]["pnl"] += trade_pnl

        # Milestone detection
        for m in [200, 350, 500, 700, 1000, 1500, 2000, 3000, 5000]:
            if balance >= m and m not in milestones:
                milestones[m] = {"date": dt, "trade_no": len(trades) + 1, "balance": round(balance, 2), "level": current_level}

        trades.append({
            "trade_no": len(trades) + 1,
            "date": dt,
            "direction": direction,
            "trigger": trigger_desc,
            "entry": round(entry_price, 2),
            "level": lvl_info["level"],
            "phase": lvl_info["phase"],
            "lots": lots,
            "outcome": outcome,
            "pnl": round(trade_pnl, 2),
            "balance": round(balance, 2),
            "ratchet_floor": ratchet_floor,
        })

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    all_win_rate = (wins / len(trades) * 100.0) if len(trades) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    trades_per_month = len(trades) / 6.0

    return {
        "name": name,
        "donch_period": donch_period,
        "target_rr": target_rr,
        "trigger_mode": trigger_mode,
        "trend_filter": trend_filter,
        "session": f"{session_hours[0]:02d}:00-{session_hours[1]:02d}:00 UTC",
        "total_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "decided": decided,
        "win_rate": round(win_rate, 1),
        "all_win_rate": round(all_win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "starting_balance": 100.0,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - 100.0, 2),
        "return_pct": round(((balance - 100.0) / 100.0) * 100.0, 1),
        "peak_balance": round(peak_balance, 2),
        "max_level_reached": max_level_reached,
        "ending_level": current_level,
        "ratchet_floor": ratchet_floor,
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "milestones": milestones,
        "monthly_stats": dict(monthly_stats),
        "trades": trades,
    }


def main():
    print("=" * 105)
    print("🏆 6-MONTH BACKTEST: NEW HIGH-FREQUENCY STRATEGIES VS COMPOUNDING LADDER ($100 SEED)")
    print("=" * 105)

    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    print(f"Loaded {len(candles)} historical 1-Hour candles from {candles[0]['date']} to {candles[-1]['date']}.")

    strategies_to_test = [
        {
            "name": "1. Baseline Live Setup (Donch 15 Close, 200 EMA, 1.5R, 07-19 UTC)",
            "p": 15, "rr": 1.5, "mode": "CLOSE", "filter": "EMA200", "pb": False, "sess": (7, 19),
        },
        {
            "name": "2. High Velocity TOUCH (Donch 10 Touch, 50 EMA, 1.25R, 07-19 UTC)",
            "p": 10, "rr": 1.25, "mode": "TOUCH", "filter": "EMA50", "pb": False, "sess": (7, 19),
        },
        {
            "name": "3. High Velocity TOUCH (Donch 10 Touch, 50 EMA, 1.30R, 07-19 UTC)",
            "p": 10, "rr": 1.30, "mode": "TOUCH", "filter": "EMA50", "pb": False, "sess": (7, 19),
        },
        {
            "name": "4. Balanced Bar Close (Donch 10 Close, 50 EMA, 1.25R, 07-19 UTC)",
            "p": 10, "rr": 1.25, "mode": "CLOSE", "filter": "EMA50", "pb": False, "sess": (7, 19),
        },
        {
            "name": "5. Ultra-Low Drawdown (Donch 10 Close, 200 EMA, 1.30R, 07-19 UTC)",
            "p": 10, "rr": 1.30, "mode": "CLOSE", "filter": "EMA200", "pb": False, "sess": (7, 19),
        },
        {
            "name": "6. Extended Hours TOUCH (Donch 10 Touch, 50 EMA, 1.25R, 07-21 UTC)",
            "p": 10, "rr": 1.25, "mode": "TOUCH", "filter": "EMA50", "pb": False, "sess": (7, 21),
        },
        {
            "name": "7. Dual Engine (Donch 10 Close + EMA20 Pullback, 50 EMA, 1.25R, 07-21 UTC)",
            "p": 10, "rr": 1.25, "mode": "CLOSE", "filter": "EMA50", "pb": True, "sess": (7, 21),
        },
    ]

    all_results = []
    for s in strategies_to_test:
        res = run_strategy_ladder_backtest(
            candles,
            name=s["name"],
            donch_period=s["p"],
            target_rr=s["rr"],
            trigger_mode=s["mode"],
            trend_filter=s["filter"],
            allow_ema20_pullback=s["pb"],
            session_hours=s["sess"],
        )
        all_results.append(res)

    print("\n" + "=" * 115)
    print(f"{'#':<3} {'Strategy Architecture':<48} {'Trd/Mo':<8} {'Total':<6} {'WinRate':<9} {'PF':<6} {'Peak Lvl':<9} {'End Bal':<11} {'Max DD%':<8}")
    print("-" * 115)
    for idx, r in enumerate(all_results, 1):
        print(f"{idx:<3} {r['name'][:46]:<48} {r['trades_per_month']:<8} {r['total_trades']:<6} {r['win_rate']:<8.1f}% {r['profit_factor']:<6.2f} L{r['max_level_reached']:<8} ${r['ending_balance']:<10,.2f} {r['max_dd_pct']:<7.1f}%")
    print("=" * 115)

    # Detailed breakdown for top performers
    print("\n" + "=" * 80)
    print("📊 DETAILED MILESTONE & MONTHLY BREAKDOWN FOR KEY STRATEGIES")
    print("=" * 80)

    for r in all_results[:5]:
        print(f"\n>>> {r['name']} <<<")
        print(f"Ending Balance: ${r['ending_balance']:,.2f} (+{r['return_pct']}%) | Peak: ${r['peak_balance']:,.2f} | Locked Floor: ${r['ratchet_floor']:,.2f}")
        print(f"Trades: {r['total_trades']} ({r['trades_per_month']} / mo) | Decided WR: {r['win_rate']}% ({r['wins']}W / {r['losses']}L / {r['scratches']} BE) | PF: {r['profit_factor']}")
        print(f"Ladder Progression: Level 1 -> Peak Level {r['max_level_reached']} -> Ending Level {r['ending_level']}")
        print(f"Max Loss Streak: {r['max_consec_losses']} | Max DD: ${r['max_dd_dollars']:,.2f} ({r['max_dd_pct']}%)")
        
        print("  Milestones Hit:")
        for m, info in sorted(r["milestones"].items()):
            print(f"    - ${m:>4}: Hit on {info['date']} (Trade #{info['trade_no']}, Balance: ${info['balance']:,.2f}, Level {info['level']})")
        
        print("  Monthly Performance:")
        for m_key, m_stat in sorted(r["monthly_stats"].items()):
            dec = m_stat["wins"] + m_stat["losses"]
            m_wr = (m_stat["wins"] / dec * 100.0) if dec > 0 else 0.0
            print(f"    {m_key}: {m_stat['trades']:>2} trds | {m_stat['wins']:>2}W {m_stat['losses']:>2}L {m_stat['scratches']:>2}BE ({m_wr:>5.1f}% WR) | P&L: ${m_stat['pnl']:>+7.2f}")


if __name__ == "__main__":
    main()
