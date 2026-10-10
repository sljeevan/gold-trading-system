#!/usr/bin/env python3
"""
High-Frequency Multi-Engine Optimizer for Gold (XAUUSD 1H, 6 Months).
Tests targeted combinations designed to achieve 20-30 trades/month while keeping win rate >= 52-56%:
1. Donchian 10 (Close) + Donchian 10 (Touch)
2. Donchian 10 + SuperTrend (10, 3.0) + Fast 1.25R - 1.35R Target
3. Multi-Session (London + NY Full Session: 07:00 - 21:00 UTC)
4. Dynamic Pullback into EMA20 on 1H
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


def run_hybrid_sim(
    candles: List[Dict[str, Any]],
    donch_period: int = 10,
    target_rr: float = 1.3,
    allow_touch: bool = True,
    allow_ema20_pullback: bool = True,
    session_hours: tuple = (7, 21),  # 07:00 to 21:00 UTC (14 hours)
    use_ema200_filter: bool = False, # If False, use SuperTrend + EMA50
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

    balance = 100.0
    peak_balance = balance
    current_level = 1
    ratchet_floor = 0.0
    consecutive_losses = 0

    wins, losses, scratches = 0, 0, 0
    gross_p, gross_l = 0.0, 0.0
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    active_until = -1
    trades = []
    milestones = {}

    for i in range(200, n - 48):
        if i <= active_until:
            continue

        dt = dates[i]
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

        # Trend filter
        if use_ema200_filter:
            bull_trend = (st_dir == 1) and (e200 is not None and c > e200) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (e200 is not None and c < e200) and (r >= 25.0)
        else:
            # SuperTrend + 50 EMA (much more agile, avoids 200 EMA delay)
            bull_trend = (st_dir == 1) and (e50 is not None and c > e50) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (e50 is not None and c < e50) and (r >= 25.0)

        direction = None
        trigger_type = None

        # 1. Breakout on Bar Close
        if c > d_up and bull_trend and (c > o):
            direction = "BUY"
            trigger_type = f"Donchian-{donch_period} Breakout Close"
        elif c < d_dn and bear_trend and (c < o):
            direction = "SELL"
            trigger_type = f"Donchian-{donch_period} Breakdown Close"

        # 2. Touch / Intra-bar penetration
        elif allow_touch:
            if (h >= d_up or c >= d_up) and bull_trend:
                direction = "BUY"
                trigger_type = f"Donchian-{donch_period} Touch"
            elif (l <= d_dn or c <= d_dn) and bear_trend:
                direction = "SELL"
                trigger_type = f"Donchian-{donch_period} Touch"

        # 3. EMA 20 Pullback in established trend
        if not direction and allow_ema20_pullback and e20 is not None:
            if bull_trend and l <= e20 and c > e20 and (c > o) and r >= 48.0:
                direction = "BUY"
                trigger_type = "EMA20 Trend Pullback"
            elif bear_trend and h >= e20 and c < e20 and (c < o) and r <= 52.0:
                direction = "SELL"
                trigger_type = "EMA20 Trend Pullback"

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

        # Enforce minimum ATR stop clearance ($8.00)
        sl_pts = max(round(1.5 * atr_val, 2), 8.00)
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

        trades.append({
            "dt": dt,
            "trigger": trigger_type,
            "outcome": outcome,
            "pnl": trade_pnl,
            "balance": balance,
            "level": current_level,
        })

        for m in [200, 350, 500, 700, 1000, 1500, 2000, 3000, 5000]:
            if balance >= m and m not in milestones:
                milestones[m] = {"date": dt, "trade_no": len(trades), "balance": round(balance, 2)}

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    pf = (gross_p / gross_l) if gross_l > 0 else 999.0
    trades_per_month = len(trades) / 6.0

    return {
        "donch_period": donch_period,
        "target_rr": target_rr,
        "allow_touch": allow_touch,
        "allow_pullback": allow_ema20_pullback,
        "ema200_filter": use_ema200_filter,
        "session": f"{session_hours[0]:02d}-{session_hours[1]:02d}",
        "total_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "ending_balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "ratchet_floor": ratchet_floor,
        "max_dd_pct": round(max_dd_pct, 1),
        "milestones": milestones,
        "trades": trades,
    }


def main():
    print("=" * 80)
    print("🎯 MULTI-ENGINE FREQUENCY BOOST TESTS (6 MONTHS XAUUSD 1H)")
    print("=" * 80)

    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    if not candles:
        return

    experiments = [
        # 1. Baseline
        {"name": "1. Baseline (Donch 15, Close, ST3+200 EMA, 1.5R, 07-19 UTC)", "p": 15, "rr": 1.5, "touch": False, "pb": False, "e200": True, "sess": (7, 19)},
        # 2. Donchian 10 (Close, ST3+200 EMA, 1.3R, 07-19 UTC)
        {"name": "2. Donchian 10 (Close, ST3+200 EMA, 1.3R, 07-19 UTC)", "p": 10, "rr": 1.3, "touch": False, "pb": False, "e200": True, "sess": (7, 19)},
        # 3. Donchian 10 (Close, ST3+50 EMA, 1.3R, 07-19 UTC)
        {"name": "3. Donchian 10 (Close, ST3+50 EMA, 1.3R, 07-19 UTC)", "p": 10, "rr": 1.3, "touch": False, "pb": False, "e200": False, "sess": (7, 19)},
        # 4. Donchian 10 (Close, ST3+50 EMA, 1.25R, 07-19 UTC)
        {"name": "4. Donchian 10 (Close, ST3+50 EMA, 1.25R, 07-19 UTC)", "p": 10, "rr": 1.25, "touch": False, "pb": False, "e200": False, "sess": (7, 19)},
        # 5. Donchian 10 TOUCH (Instant Penetration, ST3+50 EMA, 1.3R, 07-19 UTC)
        {"name": "5. Donchian 10 TOUCH (ST3+50 EMA, 1.3R, 07-19 UTC)", "p": 10, "rr": 1.3, "touch": True, "pb": False, "e200": False, "sess": (7, 19)},
        # 6. Donchian 10 TOUCH (Instant Penetration, ST3+50 EMA, 1.25R, 07-19 UTC)
        {"name": "6. Donchian 10 TOUCH (ST3+50 EMA, 1.25R, 07-19 UTC)", "p": 10, "rr": 1.25, "touch": True, "pb": False, "e200": False, "sess": (7, 19)},
        # 7. Donchian 10 (Close) + Extended Hours (07-21 UTC, 1.3R)
        {"name": "7. Donchian 10 (Close, ST3+50 EMA, 1.3R, 07-21 UTC)", "p": 10, "rr": 1.3, "touch": False, "pb": False, "e200": False, "sess": (7, 21)},
        # 8. Donchian 10 (Close) + Extended Hours (07-21 UTC, 1.25R)
        {"name": "8. Donchian 10 (Close, ST3+50 EMA, 1.25R, 07-21 UTC)", "p": 10, "rr": 1.25, "touch": False, "pb": False, "e200": False, "sess": (7, 21)},
        # 9. Donchian 10 TOUCH + Extended Hours (07-21 UTC, 1.3R)
        {"name": "9. Donchian 10 TOUCH (ST3+50 EMA, 1.3R, 07-21 UTC)", "p": 10, "rr": 1.3, "touch": True, "pb": False, "e200": False, "sess": (7, 21)},
        # 10. Donchian 10 TOUCH + Extended Hours (07-21 UTC, 1.25R)
        {"name": "10. Donchian 10 TOUCH (ST3+50 EMA, 1.25R, 07-21 UTC)", "p": 10, "rr": 1.25, "touch": True, "pb": False, "e200": False, "sess": (7, 21)},
        # 11. DUAL: Donchian 10 (Close) + EMA20 Pullbacks (ST3+50 EMA, 1.3R, 07-21 UTC)
        {"name": "11. DUAL: Donch 10 + EMA20 Pullback (1.3R, 07-21 UTC)", "p": 10, "rr": 1.3, "touch": False, "pb": True, "e200": False, "sess": (7, 21)},
        # 12. DUAL: Donchian 10 (Close) + EMA20 Pullbacks (ST3+50 EMA, 1.25R, 07-21 UTC)
        {"name": "12. DUAL: Donch 10 + EMA20 Pullback (1.25R, 07-21 UTC)", "p": 10, "rr": 1.25, "touch": False, "pb": True, "e200": False, "sess": (7, 21)},
    ]

    results = []
    for exp in experiments:
        r = run_hybrid_sim(
            candles,
            donch_period=exp["p"],
            target_rr=exp["rr"],
            allow_touch=exp["touch"],
            allow_ema20_pullback=exp["pb"],
            session_hours=exp["sess"],
            use_ema200_filter=exp["e200"],
        )
        r["name"] = exp["name"]
        results.append(r)

    print(f"{'#':<3} {'Configuration':<52} {'Trd/Mo':<8} {'Total':<6} {'Win Rate':<10} {'PF':<6} {'MaxDD':<7} {'End Equity ($)'}")
    print("-" * 102)
    for idx, r in enumerate(results, 1):
        print(f"{idx:<3} {r['name'][:50]:<52} {r['trades_per_month']:<8} {r['total_trades']:<6} {r['win_rate']:<9.1f}% {r['profit_factor']:<6.2f} {r['max_dd_pct']:<6.1f}% ${r['ending_balance']:<10,.2f}")

    print("=" * 102)

if __name__ == "__main__":
    main()
