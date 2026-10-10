"""
Comprehensive Backtest Simulator for the 30-Level Gold Compounding Ladder ($20 -> $52,384).
Simulates both:
1. 15-Minute High-Frequency Value Zone Pullback Engine (4,535 candles / 60 days)
2. 1-Year Multi-Month Macro Trend Engine (5,700 candles / 12 months)
Under the ONE_STEP_LADDER policy (Win -> L+1, Scratch -> L, Loss -> L-1).
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple

from gold_trader.core.compounding_ladder import LADDER_MATRIX, get_ladder_level_info, find_closest_level_for_balance
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_supertrend, calc_rsi


def run_15m_backtest(candles_15m: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(candles_15m)
    highs = np.array([float(c["high"]) for c in candles_15m], dtype=float)
    lows = np.array([float(c["low"]) for c in candles_15m], dtype=float)
    closes = np.array([float(c["close"]) for c in candles_15m], dtype=float)
    opens = np.array([float(c["open"]) for c in candles_15m], dtype=float)
    dates = [c["date"] for c in candles_15m]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)

    pullback_discount = 0.80  # 8 pips ($0.80)
    spread = 0.35            # XM MT5 Gold spread ($0.35/oz)

    level = 1
    balance = 20.0
    peak_level = 1
    peak_balance = 20.0
    completions = 0
    reloads = 0
    wins, losses, scratches = 0, 0, 0
    total_banked_profit = 0.0

    completion_history = []
    trade_log = []
    level_distribution = {i: 0 for i in range(1, 31)}

    act = -1
    for i in range(200, n - 36):
        if i <= act:
            continue

        dt = dates[i]
        hour = 12
        try:
            sep = "T" if "T" in dt else " "
            hour = int(dt.split(sep)[1].split(":")[0])
        except Exception:
            pass

        # London & NY sessions filter (07:00 to 19:00 UTC)
        if not (7 <= hour <= 19):
            continue

        e9, e21, e50 = ema9[i], ema21[i], ema50[i]
        st_d = st["direction"][i]
        r = rsi[i]
        if None in (e9, e21, e50, st_d, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
            dir_ = "BUY"
        elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        # Check limit fill on subsequent candle
        k_fill = i + 1
        limit_p = c - pullback_discount if dir_ == "BUY" else c + pullback_discount
        if dir_ == "BUY" and lows[k_fill] <= limit_p:
            fill_p = limit_p
        elif dir_ == "SELL" and highs[k_fill] >= limit_p:
            fill_p = limit_p
        else:
            continue  # Limit order not filled

        lvl_info = get_ladder_level_info(level)
        level_distribution[level] += 1
        start_lvl = level

        eff_entry = fill_p + spread if dir_ == "BUY" else fill_p
        tp = eff_entry + 2.0 if dir_ == "BUY" else eff_entry - 2.0 - spread
        sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + spread
        be_trigger = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0

        won, lost, scratched = False, False, False
        is_be = False
        bars = 0

        for k in range(k_fill, min(k_fill + 28, n)):
            bars = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be else sl

            if dir_ == "BUY":
                if c_k >= o_k:
                    if l_k <= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
                    if h_k >= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= tp:
                        won = True
                        break
                else:
                    if h_k >= tp:
                        won = True
                        break
                    if h_k >= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
            else:
                if c_k <= o_k:
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= tp:
                        won = True
                        break
                else:
                    if l_k <= tp:
                        won = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break

        act = i + bars

        if won:
            wins += 1
            gain = lvl_info["profit_dollars"]
            balance += gain
            trade_outcome = "WIN"
            if level == 30:
                completions += 1
                total_banked_profit += balance
                completion_history.append({
                    "date": dt,
                    "trade_no": wins + losses + scratches,
                    "final_balance": round(balance, 2),
                })
                # Bank the $52k+ and reset to Level 1 with fresh $20 bill
                level = 1
                balance = 20.0
            else:
                level += 1
                peak_level = max(peak_level, level)
        elif scratched:
            scratches += 1
            gain = 0.0
            trade_outcome = "BREAKEVEN"
            # Level and balance remain unchanged
        elif lost:
            losses += 1
            gain = -lvl_info["risk_dollars"]
            balance = max(0.0, balance + gain)
            trade_outcome = "LOSS"
            if level > 1:
                level -= 1
            else:
                if balance < 5.0:
                    reloads += 1
                    balance = 20.0
                    level = 1

        peak_balance = max(peak_balance, balance)

        trade_log.append({
            "trade_no": wins + losses + scratches,
            "date": dt,
            "direction": dir_,
            "start_level": start_lvl,
            "outcome": trade_outcome,
            "pnl": gain,
            "new_level": level,
            "balance": round(balance, 2),
        })

    total_trades = wins + losses + scratches
    decided_trades = wins + losses
    win_rate = (wins / decided_trades * 100) if decided_trades > 0 else 0.0

    return {
        "dataset": "15-Minute Intraday (60 Days / 4,535 Candles)",
        "start_date": dates[200],
        "end_date": dates[-1],
        "total_trades": total_trades,
        "decided_trades": decided_trades,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_pct": round(win_rate, 2),
        "scratch_rate_pct": round(scratches / total_trades * 100, 2) if total_trades else 0.0,
        "trades_per_day": round(total_trades / 42.0, 2),  # ~42 active trading days
        "peak_level": peak_level,
        "peak_balance": round(peak_balance, 2),
        "level_30_completions": completions,
        "reloads_count": reloads,
        "total_banked_profit": round(total_banked_profit + balance, 2),
        "ending_level": level,
        "ending_balance": round(balance, 2),
        "completion_milestones": completion_history,
        "trade_sample": trade_log[:15] + trade_log[-15:],
        "level_distribution": level_distribution,
    }


def run_1year_hourly_backtest(candles_1h: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h], dtype=float)
    lows = np.array([float(c["low"]) for c in candles_1h], dtype=float)
    closes = np.array([float(c["close"]) for c in candles_1h], dtype=float)
    opens = np.array([float(c["open"]) for c in candles_1h], dtype=float)
    dates = [c["date"] for c in candles_1h]

    fast_ema = calc_ema(closes.tolist(), 9)
    slow_ema = calc_ema(closes.tolist(), 21)
    trend_ema = calc_ema(closes.tolist(), 100)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)

    pullback_discount = 0.80  # 8 pips limit discount
    spread = 0.35

    level = 1
    balance = 20.0
    peak_level = 1
    peak_balance = 20.0
    completions = 0
    reloads = 0
    wins, losses, scratches = 0, 0, 0
    total_banked_profit = 0.0

    completion_history = []
    trade_log = []

    act = -1
    for i in range(200, n - 24):
        if i <= act:
            continue

        dt = dates[i]
        hour = 12
        try:
            sep = "T" if "T" in dt else " "
            hour = int(dt.split(sep)[1].split(":")[0])
        except Exception:
            pass

        if not (7 <= hour <= 19):
            continue

        f, s, t = fast_ema[i], slow_ema[i], trend_ema[i]
        st_d = st["direction"][i]
        r = rsi[i]
        if None in (f, s, t, st_d, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        if c > t and f > s and (l <= f and c >= s) and c >= o and (45 <= r <= 68):
            dir_ = "BUY"
        elif c < t and f < s and (h >= f and c <= s) and c <= o and (32 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        k_fill = i + 1
        limit_p = c - pullback_discount if dir_ == "BUY" else c + pullback_discount
        if dir_ == "BUY" and lows[k_fill] <= limit_p:
            fill_p = limit_p
        elif dir_ == "SELL" and highs[k_fill] >= limit_p:
            fill_p = limit_p
        else:
            continue

        lvl_info = get_ladder_level_info(level)
        eff_entry = fill_p + spread if dir_ == "BUY" else fill_p
        tp = eff_entry + 2.0 if dir_ == "BUY" else eff_entry - 2.0 - spread
        sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + spread
        be_trigger = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0

        won, lost, scratched = False, False, False
        is_be = False
        bars = 0

        for k in range(k_fill, min(k_fill + 20, n)):
            bars = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be else sl

            if dir_ == "BUY":
                if c_k >= o_k:
                    if l_k <= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
                    if h_k >= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= tp:
                        won = True
                        break
                else:
                    if h_k >= tp:
                        won = True
                        break
                    if h_k >= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
            else:
                if c_k <= o_k:
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= tp:
                        won = True
                        break
                else:
                    if l_k <= tp:
                        won = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break

        act = i + bars

        if won:
            wins += 1
            gain = lvl_info["profit_dollars"]
            balance += gain
            if level == 30:
                completions += 1
                total_banked_profit += balance
                completion_history.append({
                    "date": dt,
                    "trade_no": wins + losses + scratches,
                    "final_balance": round(balance, 2),
                })
                level = 1
                balance = 20.0
            else:
                level += 1
                peak_level = max(peak_level, level)
        elif scratched:
            scratches += 1
            gain = 0.0
        elif lost:
            losses += 1
            gain = -lvl_info["risk_dollars"]
            balance = max(0.0, balance + gain)
            if level > 1:
                level -= 1
            else:
                if balance < 5.0:
                    reloads += 1
                    balance = 20.0
                    level = 1

        peak_balance = max(peak_balance, balance)

    total_trades = wins + losses + scratches
    decided = wins + losses
    win_rate = (wins / decided * 100) if decided > 0 else 0.0

    return {
        "dataset": "1-Year Hourly Historical (5,700 Candles)",
        "start_date": dates[200],
        "end_date": dates[-1],
        "total_trades": total_trades,
        "decided_trades": decided,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_pct": round(win_rate, 2),
        "scratch_rate_pct": round(scratches / total_trades * 100, 2) if total_trades else 0.0,
        "trades_per_day": round(total_trades / 252.0, 2),
        "peak_level": peak_level,
        "peak_balance": round(peak_balance, 2),
        "level_30_completions": completions,
        "reloads_count": reloads,
        "total_banked_profit": round(total_banked_profit + balance, 2),
        "ending_level": level,
        "ending_balance": round(balance, 2),
        "completion_milestones": completion_history,
    }


if __name__ == "__main__":
    print("Fetching 15m and 1h candles...")
    candles_15m = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
    candles_1h = fetch_gold_candles("XAUUSD", period="1y", interval="1h")

    print(f"Loaded {len(candles_15m)} 15m candles and {len(candles_1h)} 1h candles.")
    
    print("\n--- Running 15m High-Frequency Compounding Backtest ---")
    res_15m = run_15m_backtest(candles_15m)
    
    print("\n--- Running 1-Year Hourly Compounding Backtest ---")
    res_1h = run_1year_hourly_backtest(candles_1h)

    output = {
        "15m_high_frequency_results": res_15m,
        "1year_hourly_results": res_1h,
    }

    out_path = os.path.expanduser("~/.gold_trading_data/compounding_backtest_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== BACKTEST SUMMARY RESULTS ===")
    print(f"15m Engine (60 Days):")
    print(f"  Total Trades: {res_15m['total_trades']} | Trades/Day: {res_15m['trades_per_day']}")
    print(f"  Wins: {res_15m['wins']} | Losses: {res_15m['losses']} | Scratches: {res_15m['scratches']}")
    print(f"  Win Rate (decided): {res_15m['win_rate_pct']}%")
    print(f"  Level 30 Completions: {res_15m['level_30_completions']} times ($52,384 each)")
    print(f"  Peak Balance: ${res_15m['peak_balance']:,.2f} | Reloads at L1: {res_15m['reloads_count']}")
    print(f"  Total Banked Wealth: ${res_15m['total_banked_profit']:,.2f}")
    if res_15m['completion_milestones']:
        for m in res_15m['completion_milestones']:
            print(f"    ⭐ Level 30 Achieved on {m['date']} (Trade #{m['trade_no']}) with ${m['final_balance']:,.2f}")

    print(f"\n1-Year Hourly Engine (365 Days):")
    print(f"  Total Trades: {res_1h['total_trades']} | Trades/Day: {res_1h['trades_per_day']}")
    print(f"  Wins: {res_1h['wins']} | Losses: {res_1h['losses']} | Scratches: {res_1h['scratches']}")
    print(f"  Win Rate (decided): {res_1h['win_rate_pct']}%")
    print(f"  Level 30 Completions: {res_1h['level_30_completions']} times ($52,384 each)")
    print(f"  Peak Balance: ${res_1h['peak_balance']:,.2f} | Reloads at L1: {res_1h['reloads_count']}")
    print(f"  Total Banked Wealth: ${res_1h['total_banked_profit']:,.2f}")
    if res_1h['completion_milestones']:
        for m in res_1h['completion_milestones']:
            print(f"    ⭐ Level 30 Achieved on {m['date']} (Trade #{m['trade_no']}) with ${m['final_balance']:,.2f}")
