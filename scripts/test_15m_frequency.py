#!/usr/bin/env python3
"""
Intraday 15-Minute Candle Evaluation (60-day historical data: 4,515 candles).
Tests if a 15m execution engine paired with 1H trend filters and Noise-Adaptive ATR stops ($8+)
can generate 30 - 50 trades/month while maintaining a 50%+ win rate.
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
    calc_rsi,
    calc_supertrend,
)

SPREAD_USD = 0.35

def evaluate_15m_engine(
    candles_15m: List[Dict[str, Any]],
    ema_fast: int = 9,
    ema_slow: int = 21,
    target_rr: float = 1.3,
    min_sl_pts: float = 8.0,
    session_start: int = 7,
    session_end: int = 19,
    strategy_type: str = "VALUE_ZONE_PULLBACK",  # "VALUE_ZONE_PULLBACK" or "DONCHIAN_15M"
) -> Dict[str, Any]:
    n = len(candles_15m)
    highs = np.array([float(c["high"]) for c in candles_15m])
    lows = np.array([float(c["low"]) for c in candles_15m])
    closes = np.array([float(c["close"]) for c in candles_15m])
    opens = np.array([float(c["open"]) for c in candles_15m])
    dates = [c["date"] for c in candles_15m]

    efast = calc_ema(closes.tolist(), ema_fast)
    eslow = calc_ema(closes.tolist(), ema_slow)
    e200 = calc_ema(closes.tolist(), 200)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi = calc_rsi(closes.tolist(), 14)
    dc20 = calc_donchian(highs.tolist(), lows.tolist(), 20)

    balance = 100.0
    peak_balance = balance
    current_level = 1
    ratchet_floor = 0.0
    consecutive_losses = 0

    wins, losses, scratches = 0, 0, 0
    gross_p, gross_l = 0.0, 0.0
    active_until = -1
    trades = []
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    for i in range(200, n - 48):
        if i <= active_until:
            continue

        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (session_start <= hour <= session_end):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        f_val = efast[i]
        s_val = eslow[i]
        m200 = e200[i]
        st_dir = st["direction"][i]
        a = atr[i]
        r = rsi[i]

        if None in (f_val, s_val, m200, st_dir, a, r):
            continue

        direction = None

        if strategy_type == "VALUE_ZONE_PULLBACK":
            # 15m Value Zone (between EMA 9 and EMA 21) pullback in trend
            bull_env = (f_val > s_val) and (c > m200) and (st_dir == 1)
            bear_env = (f_val < s_val) and (c < m200) and (st_dir == -1)

            # Bullish pullback: low reached into value zone (<= efast), but closed above eslow and green
            if bull_env and (l <= f_val) and (c >= s_val) and (c > o) and (45.0 <= r <= 70.0):
                direction = "BUY"
            # Bearish pullback: high reached into value zone (>= efast), closed below eslow and red
            elif bear_env and (h >= f_val) and (c <= s_val) and (c < o) and (30.0 <= r <= 55.0):
                direction = "SELL"

        elif strategy_type == "DONCHIAN_15M":
            # 15m Donchian 20 Breakout aligned with 200 EMA + SuperTrend
            d_up = dc20["upper"][i - 1]
            d_dn = dc20["lower"][i - 1]
            if d_up is not None and d_dn is not None:
                if c > d_up and (c > m200) and (st_dir == 1) and (c > o):
                    direction = "BUY"
                elif c < d_dn and (c < m200) and (st_dir == -1) and (c < o):
                    direction = "SELL"

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

        # CRITICAL: Always enforce noise clearance ($8.00 minimum ATR stop) even on 15m!
        sl_pts = max(round(1.5 * a, 2), min_sl_pts)
        tp_pts = round(sl_pts * target_rr, 2)
        be_trigger_pts = round(sl_pts * 1.0, 2)

        lots = max(0.01, min(50.0, math.floor((planned_risk_usd / (sl_pts * 100.0)) * 100.0 + 0.0001) / 100.0))
        dollar_per_pt = lots * 100.0

        eff_entry = c + SPREAD_USD if direction == "BUY" else c
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + SPREAD_USD
        tp_price = eff_entry + tp_pts if direction == "BUY" else eff_entry - tp_pts - SPREAD_USD
        be_price = eff_entry + be_trigger_pts if direction == "BUY" else eff_entry - be_trigger_pts

        outcome = "TIME_EXIT"
        is_be = False
        bars_held = 0

        for k in range(i + 1, min(i + 96, n)):  # up to 24 hours (96 15m bars)
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
            consecutive_losses = 0
            trade_pnl = tp_pts * dollar_per_pt
            gross_p += trade_pnl
            balance += trade_pnl
            if current_level < len(NOISE_ADAPTIVE_MATRIX):
                current_level += 1
        elif outcome == "SCRATCH":
            scratches += 1
            trade_pnl = 0.0
        else:
            losses += 1
            consecutive_losses += 1
            trade_pnl = -sl_pts * dollar_per_pt
            gross_l += abs(trade_pnl)
            balance = max(ratchet_floor, balance - abs(trade_pnl))
            current_level = find_closest_level_for_balance(balance, matrix=NOISE_ADAPTIVE_MATRIX)

        peak_balance = max(peak_balance, balance)
        dd_usd = peak_balance - balance
        dd_pct = (dd_usd / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_dd_dollars = max(max_dd_dollars, dd_usd)
        max_dd_pct = max(max_dd_pct, dd_pct)

        trades.append({"outcome": outcome, "pnl": trade_pnl, "balance": balance})

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    pf = (gross_p / gross_l) if gross_l > 0 else 999.0
    # 60 days = ~2.0 months
    trades_per_month = len(trades) / 2.0

    return {
        "strategy": strategy_type,
        "target_rr": target_rr,
        "total_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - 100.0, 2),
        "max_dd_pct": round(max_dd_pct, 1),
    }

def main():
    print("=" * 75)
    print("⚡ TESTING INTRADAY 15-MINUTE STRATEGIES (60-DAY DATASET: 4,515 CANDLES)")
    print("=" * 75)

    c_15m = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
    if not c_15m:
        print("Failed to fetch 15m candles.")
        return
    print(f"Loaded {len(c_15m)} 15m candles from {c_15m[0]['date']} to {c_15m[-1]['date']}.\n")

    setups = [
        {"name": "15m Value Zone Pullback (1.2R Fast TP)", "strat": "VALUE_ZONE_PULLBACK", "rr": 1.2},
        {"name": "15m Value Zone Pullback (1.3R Target)", "strat": "VALUE_ZONE_PULLBACK", "rr": 1.3},
        {"name": "15m Value Zone Pullback (1.5R Target)", "strat": "VALUE_ZONE_PULLBACK", "rr": 1.5},
        {"name": "15m Donchian 20 Breakout (1.2R Fast TP)", "strat": "DONCHIAN_15M", "rr": 1.2},
        {"name": "15m Donchian 20 Breakout (1.3R Target)", "strat": "DONCHIAN_15M", "rr": 1.3},
        {"name": "15m Donchian 20 Breakout (1.5R Target)", "strat": "DONCHIAN_15M", "rr": 1.5},
    ]

    print(f"{'#':<3} {'Strategy Setup':<42} {'Trades/Mo':<10} {'Total':<6} {'Win Rate':<10} {'PF':<6} {'MaxDD':<8} {'End Equity ($)'}")
    print("-" * 92)
    for idx, s in enumerate(setups, 1):
        res = evaluate_15m_engine(c_15m, target_rr=s["rr"], strategy_type=s["strat"])
        print(f"{idx:<3} {s['name']:<42} {res['trades_per_month']:<10} {res['total_trades']:<6} {res['win_rate']:<9.1f}% {res['profit_factor']:<6.2f} {res['max_dd_pct']:<7.1f}% ${res['ending_balance']:<10,.2f}")

    print("=" * 75)

if __name__ == "__main__":
    main()
