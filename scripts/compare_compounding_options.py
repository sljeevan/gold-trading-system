#!/usr/bin/env python3
"""
Comprehensive Head-to-Head Backtest Comparison of 3 Compounding Strategies:
1. 1H Compounding Hybrid (Donchian 15 + SuperTrend + 200 EMA + Dynamic ATR Stop)
2. 15m High-Frequency Compounding (EMA 9/21 Wholesale Pullback + 8-pip Limit Order)
3. Fast-Track Staged Compounding ($100 -> $5,000 with Ratchet Floors & Dual Engine)

Evaluated across the exact same synchronized historical XAU/USD market data.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Adjust path to find gold_trader package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from gold_trader.core.compounding_ladder import LADDER_MATRIX, get_ladder_level_info, find_closest_level_for_balance
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)

SPREAD_USD = 0.35  # Realistic XM broker spread: 3.5 pips ($0.35/oz)

# ==============================================================================
# 1. OPTION 1: 1H Compounding Hybrid (Donchian 15 + SuperTrend + ATR Volatility Stops)
# ==============================================================================
def backtest_option1_1h_hybrid(
    candles_1h: List[Dict[str, Any]],
    candles_15m: List[Dict[str, Any]],
    starting_balance: float = 20.0,
    target_balance: float = 52384.0,
    milestone_preservation: bool = True,
) -> Dict[str, Any]:
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h])
    lows = np.array([float(c["low"]) for c in candles_1h])
    closes = np.array([float(c["close"]) for c in candles_1h])
    opens = np.array([float(c["open"]) for c in candles_1h])
    dates = [c["date"] for c in candles_1h]

    # Indicators
    donch15 = calc_donchian(highs.tolist(), lows.tolist(), 15)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    # State
    balance = starting_balance
    level = find_closest_level_for_balance(balance)
    peak_balance = balance
    peak_level = level
    completions = 0
    reloads = 0
    trades: List[Dict[str, Any]] = []
    milestone_dates = {}

    wins, losses, scratches = 0, 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    daily_trades: Dict[str, int] = {}
    active_until_idx = -1

    for i in range(200, n - 24):
        if i <= active_until_idx:
            continue

        dt_str = dates[i]
        day = dt_str.split("T" if "T" in dt_str else " ")[0]
        hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])

        # Session filter: 07:00 to 19:00 UTC
        if not (7 <= hour <= 19):
            continue

        # Daily trade cap (max 4 trades/day)
        if daily_trades.get(day, 0) >= 4:
            continue

        d_up = donch15["upper"][i - 1]
        d_dn = donch15["lower"][i - 1]
        st_dir = st["direction"][i]
        e200 = ema200[i]
        atr_val = atr14[i]

        if None in (d_up, d_dn, st_dir, e200, atr_val):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        # Breakout Signals
        direction = None
        if c > d_up and st_dir == 1 and c > e200:
            direction = "BUY"
        elif c < d_dn and st_dir == -1 and c < e200:
            direction = "SELL"

        if not direction:
            continue

        # Position Sizing & ATR Stop
        lvl_info = get_ladder_level_info(level)
        planned_risk_dollars = lvl_info["risk_dollars"]
        sl_points = max(1.5 * atr_val, 8.0)  # Min $8.00 ATR stop
        tp_points = 1.8 * sl_points          # 1.8 R:R

        # Dynamic Lot calculation
        lots = max(0.01, round(planned_risk_dollars / (sl_points * 100.0), 2))
        dollar_risk_per_point = lots * 100.0
        max_dollar_loss = sl_points * dollar_risk_per_point

        # Entry Price & Bracket
        eff_entry = c + SPREAD_USD if direction == "BUY" else c
        tp_price = eff_entry + tp_points if direction == "BUY" else eff_entry - tp_points - SPREAD_USD
        sl_price = eff_entry - sl_points if direction == "BUY" else eff_entry + sl_points + SPREAD_USD
        be_trigger_price = eff_entry + sl_points if direction == "BUY" else eff_entry - sl_points  # 1.0R move

        # Simulate intrabar outcome across next 48 1-hour candles
        outcome = "TIME_EXIT"
        exit_pnl = 0.0
        is_be = False
        bars_held = 0

        for k in range(i + 1, min(i + 49, n)):
            bars_held = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = (eff_entry + SPREAD_USD if direction == "BUY" else eff_entry - SPREAD_USD) if is_be else sl_price

            if direction == "BUY":
                if c_k >= o_k:  # O -> L -> H -> C
                    if l_k <= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -max_dollar_loss
                        break
                    if h_k >= be_trigger_price:
                        is_be = True
                    if h_k >= tp_price:
                        outcome = "WIN"
                        exit_pnl = tp_points * dollar_risk_per_point
                        break
                else:          # O -> H -> L -> C
                    if h_k >= tp_price:
                        outcome = "WIN"
                        exit_pnl = tp_points * dollar_risk_per_point
                        break
                    if h_k >= be_trigger_price:
                        is_be = True
                    curr_sl = (eff_entry + SPREAD_USD) if is_be else sl_price
                    if l_k <= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -max_dollar_loss
                        break
            else:  # SELL
                if c_k <= o_k:  # O -> H -> L -> C
                    if h_k >= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -max_dollar_loss
                        break
                    if l_k <= be_trigger_price:
                        is_be = True
                    if l_k <= tp_price:
                        outcome = "WIN"
                        exit_pnl = tp_points * dollar_risk_per_point
                        break
                else:          # O -> L -> H -> C
                    if l_k <= tp_price:
                        outcome = "WIN"
                        exit_pnl = tp_points * dollar_risk_per_point
                        break
                    if l_k <= be_trigger_price:
                        is_be = True
                    curr_sl = (eff_entry - SPREAD_USD) if is_be else sl_price
                    if h_k >= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -max_dollar_loss
                        break

        active_until_idx = i + bars_held
        daily_trades[day] = daily_trades.get(day, 0) + 1

        # Apply Compounding Ladder Progression
        if outcome == "WIN":
            wins += 1
            gross_profit += exit_pnl
            balance += exit_pnl
            if level == 30 or balance >= target_balance:
                completions += 1
                level = 1
                balance = starting_balance
            else:
                level = min(30, level + 1)
        elif outcome == "SCRATCH":
            scratches += 1
        else:  # LOSS
            losses += 1
            gross_loss += abs(exit_pnl)
            balance = max(0.0, balance - abs(exit_pnl))

            # Milestone floors
            if milestone_preservation:
                floor = 20.0
                if balance >= 2900.0:
                    floor = 2900.0
                elif balance >= 200.0:
                    floor = 200.0
                balance = max(floor, balance)

            if level > 1:
                level -= 1
            else:
                if balance < 10.0:
                    reloads += 1
                    balance = starting_balance
                    level = 1

        peak_balance = max(peak_balance, balance)
        peak_level = max(peak_level, level)

        # Track Milestones
        for m in [100, 200, 500, 1000, 2500, 5000]:
            if balance >= m and m not in milestone_dates:
                milestone_dates[m] = {"date": dt_str, "trade_no": wins + losses + scratches}

        trades.append({
            "date": dt_str,
            "direction": direction,
            "outcome": outcome,
            "pnl": exit_pnl,
            "level": level,
            "balance": balance,
            "lots": lots,
        })

    return {
        "strategy": "Option 1: 1H Compounding Hybrid",
        "timeframe": "1h",
        "starting_balance": starting_balance,
        "final_balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "peak_level": peak_level,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(wins / max(1, wins + losses) * 100, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / max(1.0, gross_loss), 2),
        "completions": completions,
        "reloads": reloads,
        "milestones": milestone_dates,
        "trades": trades,
    }

# ==============================================================================
# 2. OPTION 2: 15m High-Frequency Value Zone Pullbacks (Wholesale Limit Orders)
# ==============================================================================
def backtest_option2_15m_ladder(
    candles_15m: List[Dict[str, Any]],
    starting_balance: float = 20.0,
    target_balance: float = 52384.0,
    pullback_disc_pips: float = 8.0,
) -> Dict[str, Any]:
    n = len(candles_15m)
    highs = np.array([float(c["high"]) for c in candles_15m])
    lows = np.array([float(c["low"]) for c in candles_15m])
    closes = np.array([float(c["close"]) for c in candles_15m])
    opens = np.array([float(c["open"]) for c in candles_15m])
    dates = [c["date"] for c in candles_15m]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    rsi = calc_rsi(closes.tolist(), 14)

    disc_usd = pullback_disc_pips * 0.10  # 8 pips = $0.80

    balance = starting_balance
    level = find_closest_level_for_balance(balance)
    peak_balance = balance
    peak_level = level
    completions = 0
    reloads = 0
    milestone_dates = {}

    wins, losses, scratches = 0, 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    active_until_idx = -1
    trades = []

    for i in range(200, n - 36):
        if i <= active_until_idx:
            continue

        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19):
            continue

        e9, e21, e50 = ema9[i], ema21[i], ema50[i]
        r = rsi[i]
        if None in (e9, e21, e50, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        # Bullish Pullback
        if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
            dir_ = "BUY"
        # Bearish Pullback
        elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        # Wholesale Limit Order Fill check on next candle
        k_fill = i + 1
        limit_price = c - disc_usd if dir_ == "BUY" else c + disc_usd
        if dir_ == "BUY" and lows[k_fill] <= limit_price:
            fill_price = limit_price
        elif dir_ == "SELL" and highs[k_fill] >= limit_price:
            fill_price = limit_price
        else:
            continue  # Limit order expired unfilled

        lvl_info = get_ladder_level_info(level)
        lot_size = lvl_info["lot_size"]
        dollar_per_pt = lot_size * 100.0

        eff_entry = fill_price + SPREAD_USD if dir_ == "BUY" else fill_price
        tp = eff_entry + 2.00 if dir_ == "BUY" else eff_entry - 2.00 - SPREAD_USD
        sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + SPREAD_USD
        be_price = eff_entry + 1.00 if dir_ == "BUY" else eff_entry - 1.00

        won, lost, scratched = False, False, False
        is_be = False
        bars = 0

        for k in range(k_fill, min(k_fill + 25, n)):
            bars = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be else sl

            if dir_ == "BUY":
                if c_k >= o_k:
                    if l_k <= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break
                    if h_k >= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= tp: won = True; break
                else:
                    if h_k >= tp: won = True; break
                    if h_k >= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break
            else:
                if c_k <= o_k:
                    if h_k >= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break
                    if l_k <= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= tp: won = True; break
                else:
                    if l_k <= tp: won = True; break
                    if l_k <= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break

        active_until_idx = i + bars

        pnl = 0.0
        if won:
            wins += 1
            pnl = 2.00 * dollar_per_pt
            gross_profit += pnl
            balance += pnl
            if level == 30 or balance >= target_balance:
                completions += 1
                level = 1
                balance = starting_balance
            else:
                level = min(30, level + 1)
        elif scratched:
            scratches += 1
            pnl = 0.0
        else:
            losses += 1
            pnl = -1.54 * dollar_per_pt
            gross_loss += abs(pnl)
            balance = max(0.0, balance - abs(pnl))
            if level > 1:
                level -= 1
            else:
                if balance < 5.0:
                    reloads += 1
                    balance = starting_balance
                    level = 1

        peak_balance = max(peak_balance, balance)
        peak_level = max(peak_level, level)

        for m in [100, 200, 500, 1000, 2500, 5000]:
            if balance >= m and m not in milestone_dates:
                milestone_dates[m] = {"date": dt, "trade_no": wins + losses + scratches}

        trades.append({
            "date": dt,
            "direction": dir_,
            "outcome": "WIN" if won else "SCRATCH" if scratched else "LOSS",
            "pnl": pnl,
            "level": level,
            "balance": balance,
            "lots": lot_size,
        })

    return {
        "strategy": "Option 2: 15m Value Zone Pullback",
        "timeframe": "15m",
        "starting_balance": starting_balance,
        "final_balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "peak_level": peak_level,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(wins / max(1, wins + losses) * 100, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / max(1.0, gross_loss), 2),
        "completions": completions,
        "reloads": reloads,
        "milestones": milestone_dates,
        "trades": trades,
    }

# ==============================================================================
# 3. OPTION 3: Fast-Track Staged Compounding ($100 -> $5,000)
# ==============================================================================
def backtest_option3_fast_track(
    candles_1h: List[Dict[str, Any]],
    candles_15m: List[Dict[str, Any]],
    starting_balance: float = 100.0,
    target_balance: float = 5000.0,
) -> Dict[str, Any]:
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h])
    lows = np.array([float(c["low"]) for c in candles_1h])
    closes = np.array([float(c["close"]) for c in candles_1h])
    opens = np.array([float(c["open"]) for c in candles_1h])
    dates = [c["date"] for c in candles_1h]

    donch15 = calc_donchian(highs.tolist(), lows.tolist(), 15)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    balance = starting_balance
    peak_balance = balance
    ratchet_floor = 0.0
    completions = 0
    reloads = 0
    milestone_dates = {}

    wins, losses, scratches = 0, 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    trades = []
    active_until_idx = -1

    for i in range(200, n - 24):
        if i <= active_until_idx:
            continue

        dt_str = dates[i]
        hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])
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

        # Dynamic Risk Tiering:
        # Phase 1: $100 -> $500: 8.5%
        # Phase 2: $500 -> $2000: 7.0%
        # Phase 3: $2000 -> $5000: 5.0%
        if balance < 500.0:
            risk_pct = 0.085
        elif balance < 2000.0:
            risk_pct = 0.070
        else:
            risk_pct = 0.050

        # Automated Ratchet Floors:
        if balance >= 2500.0:
            ratchet_floor = max(ratchet_floor, 1800.0)
        elif balance >= 1000.0:
            ratchet_floor = max(ratchet_floor, 700.0)
        elif balance >= 500.0:
            ratchet_floor = max(ratchet_floor, 350.0)
        elif balance >= 250.0:
            ratchet_floor = max(ratchet_floor, 150.0)

        sl_points = max(1.5 * atr_val, 8.0)
        risk_capital = max(balance - ratchet_floor, balance * 0.5) if ratchet_floor > 0 else balance
        planned_risk_usd = risk_capital * risk_pct

        lots = max(0.01, round(planned_risk_usd / (sl_points * 100.0), 2))
        dollar_per_pt = lots * 100.0

        eff_entry = c + SPREAD_USD if direction == "BUY" else c
        # 50% target at 1.5R, 50% target trailing swing (blended 2.2R target)
        blended_tp_points = 2.2 * sl_points
        tp_price = eff_entry + blended_tp_points if direction == "BUY" else eff_entry - blended_tp_points - SPREAD_USD
        sl_price = eff_entry - sl_points if direction == "BUY" else eff_entry + sl_points + SPREAD_USD
        be_trigger_price = eff_entry + sl_points if direction == "BUY" else eff_entry - sl_points

        outcome = "TIME_EXIT"
        exit_pnl = 0.0
        is_be = False
        bars_held = 0

        for k in range(i + 1, min(i + 49, n)):
            bars_held = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = (eff_entry + SPREAD_USD if direction == "BUY" else eff_entry - SPREAD_USD) if is_be else sl_price

            if direction == "BUY":
                if c_k >= o_k:
                    if l_k <= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -(sl_points * dollar_per_pt)
                        break
                    if h_k >= be_trigger_price:
                        is_be = True
                    if h_k >= tp_price:
                        outcome = "WIN"
                        exit_pnl = blended_tp_points * dollar_per_pt
                        break
                else:
                    if h_k >= tp_price:
                        outcome = "WIN"
                        exit_pnl = blended_tp_points * dollar_per_pt
                        break
                    if h_k >= be_trigger_price:
                        is_be = True
                    curr_sl = (eff_entry + SPREAD_USD) if is_be else sl_price
                    if l_k <= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -(sl_points * dollar_per_pt)
                        break
            else:
                if c_k <= o_k:
                    if h_k >= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -(sl_points * dollar_per_pt)
                        break
                    if l_k <= be_trigger_price:
                        is_be = True
                    if l_k <= tp_price:
                        outcome = "WIN"
                        exit_pnl = blended_tp_points * dollar_per_pt
                        break
                else:
                    if l_k <= tp_price:
                        outcome = "WIN"
                        exit_pnl = blended_tp_points * dollar_per_pt
                        break
                    if l_k <= be_trigger_price:
                        is_be = True
                    curr_sl = (eff_entry - SPREAD_USD) if is_be else sl_price
                    if h_k >= curr_sl:
                        outcome = "SCRATCH" if is_be else "LOSS"
                        exit_pnl = 0.0 if is_be else -(sl_points * dollar_per_pt)
                        break

        active_until_idx = i + bars_held

        if outcome == "WIN":
            wins += 1
            gross_profit += exit_pnl
            balance += exit_pnl
            if balance >= target_balance:
                completions += 1
        elif outcome == "SCRATCH":
            scratches += 1
        else:
            losses += 1
            gross_loss += abs(exit_pnl)
            balance = max(ratchet_floor, balance - abs(exit_pnl))
            if balance < 25.0:
                reloads += 1
                balance = starting_balance
                ratchet_floor = 0.0

        peak_balance = max(peak_balance, balance)

        for m in [200, 500, 1000, 2500, 5000]:
            if balance >= m and m not in milestone_dates:
                milestone_dates[m] = {"date": dt_str, "trade_no": wins + losses + scratches}

        trades.append({
            "date": dt_str,
            "direction": direction,
            "outcome": outcome,
            "pnl": exit_pnl,
            "balance": balance,
            "lots": lots,
            "floor": ratchet_floor,
        })

    return {
        "strategy": "Option 3A: Fast-Track 1H ($100 -> $5,000)",
        "timeframe": "1h",
        "starting_balance": starting_balance,
        "final_balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "peak_level": "N/A (Tiered)",
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(wins / max(1, wins + losses) * 100, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / max(1.0, gross_loss), 2),
        "completions": completions,
        "reloads": reloads,
        "milestones": milestone_dates,
        "trades": trades,
    }


def backtest_option3_fast_track_15m(
    candles_15m: List[Dict[str, Any]],
    starting_balance: float = 100.0,
    target_balance: float = 5000.0,
    pullback_disc_pips: float = 8.0,
) -> Dict[str, Any]:
    """Fast-Track Staged Compounding on 15m Value Zone Pullbacks with Wholesale Limit Orders."""
    n = len(candles_15m)
    highs = np.array([float(c["high"]) for c in candles_15m])
    lows = np.array([float(c["low"]) for c in candles_15m])
    closes = np.array([float(c["close"]) for c in candles_15m])
    opens = np.array([float(c["open"]) for c in candles_15m])
    dates = [c["date"] for c in candles_15m]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    rsi = calc_rsi(closes.tolist(), 14)

    disc_usd = pullback_disc_pips * 0.10

    balance = starting_balance
    peak_balance = balance
    ratchet_floor = 0.0
    completions = 0
    reloads = 0
    milestone_dates = {}

    wins, losses, scratches = 0, 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    active_until_idx = -1
    trades = []

    for i in range(200, n - 36):
        if i <= active_until_idx:
            continue

        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19):
            continue

        e9, e21, e50 = ema9[i], ema21[i], ema50[i]
        r = rsi[i]
        if None in (e9, e21, e50, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
            dir_ = "BUY"
        elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        k_fill = i + 1
        limit_price = c - disc_usd if dir_ == "BUY" else c + disc_usd
        if dir_ == "BUY" and lows[k_fill] <= limit_price:
            fill_price = limit_price
        elif dir_ == "SELL" and highs[k_fill] >= limit_price:
            fill_price = limit_price
        else:
            continue

        # Fast-track risk tiers
        if balance < 500.0:
            risk_pct = 0.085
        elif balance < 2000.0:
            risk_pct = 0.070
        else:
            risk_pct = 0.050

        # Ratchet floors
        if balance >= 2500.0:
            ratchet_floor = max(ratchet_floor, 1800.0)
        elif balance >= 1000.0:
            ratchet_floor = max(ratchet_floor, 700.0)
        elif balance >= 500.0:
            ratchet_floor = max(ratchet_floor, 350.0)
        elif balance >= 250.0:
            ratchet_floor = max(ratchet_floor, 150.0)

        sl_points = 1.54  # 15.4 pips
        tp_points = 2.00  # 20.0 pips (1.30 R:R)
        risk_capital = max(balance - ratchet_floor, balance * 0.5) if ratchet_floor > 0 else balance
        planned_risk_usd = risk_capital * risk_pct

        lots = max(0.01, round(planned_risk_usd / (sl_points * 100.0), 2))
        dollar_per_pt = lots * 100.0

        eff_entry = fill_price + SPREAD_USD if dir_ == "BUY" else fill_price
        tp = eff_entry + tp_points if dir_ == "BUY" else eff_entry - tp_points - SPREAD_USD
        sl = eff_entry - sl_points if dir_ == "BUY" else eff_entry + sl_points + SPREAD_USD
        be_price = eff_entry + 1.00 if dir_ == "BUY" else eff_entry - 1.00

        won, lost, scratched = False, False, False
        is_be = False
        bars = 0

        for k in range(k_fill, min(k_fill + 25, n)):
            bars = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be else sl

            if dir_ == "BUY":
                if c_k >= o_k:
                    if l_k <= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break
                    if h_k >= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= tp: won = True; break
                else:
                    if h_k >= tp: won = True; break
                    if h_k >= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break
            else:
                if c_k <= o_k:
                    if h_k >= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break
                    if l_k <= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= tp: won = True; break
                else:
                    if l_k <= tp: won = True; break
                    if l_k <= be_price: is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= curr_sl:
                        if is_be: scratched = True
                        else: lost = True
                        break

        active_until_idx = i + bars

        pnl = 0.0
        if won:
            wins += 1
            pnl = tp_points * dollar_per_pt
            gross_profit += pnl
            balance += pnl
            if balance >= target_balance:
                completions += 1
        elif scratched:
            scratches += 1
            pnl = 0.0
        else:
            losses += 1
            pnl = -sl_points * dollar_per_pt
            gross_loss += abs(pnl)
            balance = max(ratchet_floor, balance - abs(pnl))
            if balance < 25.0:
                reloads += 1
                balance = starting_balance
                ratchet_floor = 0.0

        peak_balance = max(peak_balance, balance)

        for m in [200, 500, 1000, 2500, 5000]:
            if balance >= m and m not in milestone_dates:
                milestone_dates[m] = {"date": dt, "trade_no": wins + losses + scratches}

        trades.append({
            "date": dt,
            "direction": dir_,
            "outcome": "WIN" if won else "SCRATCH" if scratched else "LOSS",
            "pnl": pnl,
            "balance": balance,
            "lots": lots,
            "floor": ratchet_floor,
        })

    return {
        "strategy": "Option 3B: Fast-Track 15m ($100 -> $5,000)",
        "timeframe": "15m",
        "starting_balance": starting_balance,
        "final_balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "peak_level": "N/A (Tiered)",
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(wins / max(1, wins + losses) * 100, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / max(1.0, gross_loss), 2),
        "completions": completions,
        "reloads": reloads,
        "milestones": milestone_dates,
        "trades": trades,
    }


def main():
    print("=" * 80)
    print("🔬 FETCHING HISTORICAL GOLD CANDLES (Synchronized 60-Day Dataset)...")
    c_15m = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
    c_1h = fetch_gold_candles("XAUUSD", period="60d", interval="1h")
    print(f"Data Loaded: {len(c_15m)} 15m candles | {len(c_1h)} 1h candles.")
    print(f"Date Range: {c_15m[0]['date']} to {c_15m[-1]['date']}")
    print("=" * 80)

    print("\nRunning Backtest on Option 1 (1H Compounding Hybrid - $20 Seed)...")
    res1 = backtest_option1_1h_hybrid(c_1h, c_15m, starting_balance=20.0)

    print("Running Backtest on Option 1 (1H Compounding Hybrid - $100 Seed)...")
    res1_100 = backtest_option1_1h_hybrid(c_1h, c_15m, starting_balance=100.0)

    print("Running Backtest on Option 2 (15m Value Zone Pullback - $20 Seed)...")
    res2 = backtest_option2_15m_ladder(c_15m, starting_balance=20.0)

    print("Running Backtest on Option 3A (Fast-Track 1H - $100 Seed)...")
    res3a = backtest_option3_fast_track(c_1h, c_15m, starting_balance=100.0)

    print("Running Backtest on Option 3B (Fast-Track 15m Value Zone - $100 Seed)...")
    res3b = backtest_option3_fast_track_15m(c_15m, starting_balance=100.0)

    results = {
        "option1_20": res1,
        "option1_100": res1_100,
        "option2_20": res2,
        "option3a_100": res3a,
        "option3b_100": res3b,
    }

    # Save results to json
    out_file = os.path.expanduser("~/.gold_trading_data/compounding_options_backtest_results.json")
    with open(out_file, "w") as f:
        compact_res = {}
        for k, v in results.items():
            compact_res[k] = {ik: iv for ik, iv in v.items() if ik != "trades"}
        json.dump(compact_res, f, indent=2)

    print("\n" + "=" * 115)
    print(f"{'Option / Strategy':<42} | {'Seed':<5} | {'Trades':<6} | {'Win Rate':<8} | {'PF':<5} | {'Peak Bal ($)':<12} | {'Final Bal ($)':<12} | {'Reloads'}")
    print("-" * 115)
    for k, v in results.items():
        name = v["strategy"]
        seed = f"${v['starting_balance']:,.0f}"
        tr = v["total_trades"]
        wr = f"{v['win_rate']}%"
        pf = f"{v['profit_factor']}"
        pk = f"${v['peak_balance']:,.2f}"
        fn = f"${v['final_balance']:,.2f}"
        rel = v["reloads"]
        print(f"{name:<42} | {seed:<5} | {tr:<6} | {wr:<8} | {pf:<5} | {pk:<12} | {fn:<12} | {rel}")
    print("=" * 115)

    print("\n🎯 KEY MILESTONES REACHED:")
    for k, v in results.items():
        print(f"\n--- {v['strategy']} (Seed: ${v['starting_balance']:,.0f}) ---")
        if not v["milestones"]:
            print("  No major milestones reached.")
        for m, inf in sorted(v["milestones"].items()):
            print(f"  Passed ${m:,} on {inf['date']} (Trade #{inf['trade_no']})")

if __name__ == "__main__":
    main()
