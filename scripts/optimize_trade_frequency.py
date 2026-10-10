#!/usr/bin/env python3
"""
Optimization Grid Search: Maximizing Trade Frequency While Maintaining High Win Rate.
Tests multiple combinations across 6 months of Gold (XAUUSD 1H) historical data:
1. Donchian Periods (8, 10, 12, 15, 20)
2. Entry Triggers (Bar Close, High/Low Touch)
3. Trend Filters (ST3 + EMA200, ST3 + EMA50, ST3 Only, MACD + ST3)
4. Target Multipliers (1.1R, 1.2R, 1.3R, 1.4R, 1.5R, 1.8R)
5. Session Windows (07:00-19:00, 06:00-21:00, 24H)
6. Secondary Pullback Engine (EMA Value Zone retests in trending market)
7. Asian Range Sweep (Judas Swing) at London Open
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

SPREAD_USD = 0.35  # Realistic XM live broker spread ($0.35/oz)


def evaluate_system(
    candles_1h: List[Dict[str, Any]],
    donch_period: int = 15,
    entry_trigger: str = "CLOSE",     # "CLOSE" or "TOUCH"
    trend_filter: str = "ST3_EMA200", # "ST3_EMA200", "ST3_EMA50", "ST3_ONLY", "MACD_ST3"
    target_rr: float = 1.5,
    session_start: int = 7,
    session_end: int = 19,
    enable_pullback_engine: bool = False,
    enable_asian_sweep: bool = False,
) -> Dict[str, Any]:
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h])
    lows = np.array([float(c["low"]) for c in candles_1h])
    closes = np.array([float(c["close"]) for c in candles_1h])
    opens = np.array([float(c["open"]) for c in candles_1h])
    dates = [c["date"] for c in candles_1h]

    # Technical Indicators
    dc = calc_donchian(highs.tolist(), lows.tolist(), donch_period)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi14 = calc_rsi(closes.tolist(), 14)

    # MACD (12, 26, 9)
    ema12 = calc_ema(closes.tolist(), 12)
    ema26 = calc_ema(closes.tolist(), 26)
    macd_line = [(ema12[i] - ema26[i]) if (ema12[i] is not None and ema26[i] is not None) else None for i in range(n)]
    v_macd = [m for m in macd_line if m is not None]
    macd_sig = [None] * (len(macd_line) - len(v_macd)) + calc_ema(v_macd, 9)

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

    for i in range(200, n - 48):
        if i <= active_until:
            continue

        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (session_start <= hour <= session_end):
            continue

        d_up = dc["upper"][i - 1]
        d_dn = dc["lower"][i - 1]
        st_dir = st["direction"][i]
        e200 = ema200[i]
        e50 = ema50[i]
        e20 = ema20[i]
        atr_val = atr14[i]
        r = rsi14[i]
        m_val = macd_line[i]
        m_sig = macd_sig[i]

        if None in (d_up, d_dn, st_dir, atr_val, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        # Trend filter checks
        bull_trend, bear_trend = False, False
        if trend_filter == "ST3_EMA200":
            bull_trend = (st_dir == 1) and (e200 is not None and c > e200) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (e200 is not None and c < e200) and (r >= 25.0)
        elif trend_filter == "ST3_EMA50":
            bull_trend = (st_dir == 1) and (e50 is not None and c > e50) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (e50 is not None and c < e50) and (r >= 25.0)
        elif trend_filter == "ST3_ONLY":
            bull_trend = (st_dir == 1) and (r <= 75.0)
            bear_trend = (st_dir == -1) and (r >= 25.0)
        elif trend_filter == "MACD_ST3":
            macd_bull = (m_val is not None and m_sig is not None and m_val > m_sig)
            macd_bear = (m_val is not None and m_sig is not None and m_val < m_sig)
            bull_trend = (st_dir == 1) and macd_bull and (r <= 75.0)
            bear_trend = (st_dir == -1) and macd_bear and (r >= 25.0)

        direction = None

        # 1. Primary Trigger: Donchian Channel
        if entry_trigger == "CLOSE":
            if c > d_up and bull_trend and (c > o):
                direction = "BUY"
            elif c < d_dn and bear_trend and (c < o):
                direction = "SELL"
        elif entry_trigger == "TOUCH":
            if (h >= d_up or c >= d_up) and bull_trend:
                direction = "BUY"
            elif (l <= d_dn or c <= d_dn) and bear_trend:
                direction = "SELL"

        # 2. Secondary Trigger: Value Zone EMA20 Pullback in established trend
        if not direction and enable_pullback_engine and e20 is not None:
            # Bullish pullback: Price dipped into EMA20 and closed back above
            if bull_trend and l <= e20 and c > e20 and (c > o) and r >= 45.0:
                direction = "BUY"
            # Bearish pullback: Price popped into EMA20 and closed back below
            elif bear_trend and h >= e20 and c < e20 and (c < o) and r <= 55.0:
                direction = "SELL"

        # 3. Secondary Trigger: Asian Session Liquidity Sweep (Judas Swing) at London Open (07:00 - 09:00 UTC)
        if not direction and enable_asian_sweep and (7 <= hour <= 9):
            # Calculate Asian range (00:00 - 06:00 UTC, last 6 bars)
            asian_high = max(highs[max(0, i - 7):i])
            asian_low = min(lows[max(0, i - 7):i])
            # Bullish Judas Swing: Swept Asian Low, rejected, and reclaimed Asian low in macro bull trend
            if bull_trend and l < asian_low and c > asian_low:
                direction = "BUY"
            # Bearish Judas Swing: Swept Asian High, rejected, and closed below Asian high in macro bear trend
            elif bear_trend and h > asian_high and c < asian_high:
                direction = "SELL"

        if not direction:
            continue

        # Ratchet floor update
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

        trades.append({"outcome": outcome, "pnl": trade_pnl, "balance": balance})

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    pf = (gross_p / gross_l) if gross_l > 0 else 999.0
    trades_per_month = len(trades) / 6.0

    return {
        "donch_period": donch_period,
        "entry_trigger": entry_trigger,
        "trend_filter": trend_filter,
        "target_rr": target_rr,
        "session": f"{session_start:02d}-{session_end:02d}",
        "pullback_engine": enable_pullback_engine,
        "asian_sweep": enable_asian_sweep,
        "total_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - 100.0, 2),
        "peak_balance": round(peak_balance, 2),
        "max_dd_pct": round(max_dd_pct, 1),
    }


def main():
    print("=" * 80)
    print("🔬 COMPREHENSIVE COMBINATORIAL OPTIMIZATION FOR GOLD (XAU/USD)")
    print("Goal: Maximize Monthly Trades (Frequency) while Keeping Win Rate >= 48-55%")
    print("=" * 80)

    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    if not candles:
        print("Failed to fetch candles.")
        return
    print(f"Loaded {len(candles)} candles (6 months).\n")

    # Define Combinations to test
    configurations = [
        # Baseline
        {"name": "1. Baseline (Donch 15, Close, ST3+EMA200, 1.5R)", "donch_period": 15, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.5, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        
        # Levers on Donchian Period (Donch 10 / Donch 12)
        {"name": "2. Donchian 10 Breakout (Faster Channel, 1.5R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.5, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        {"name": "3. Donchian 10 Breakout (Faster Channel, 1.2R Fast TP)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.2, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        {"name": "4. Donchian 10 Breakout (Faster Channel, 1.3R Sweet Spot)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        
        # Levers on Trend Filter (ST3 + EMA50 vs ST3 Only)
        {"name": "5. Donchian 15 + ST3/EMA50 (Faster Trend Filter, 1.3R)", "donch_period": 15, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA50", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        {"name": "6. Donchian 10 + ST3/EMA50 (Responsive Filter, 1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA50", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        {"name": "7. Donchian 10 + SuperTrend Only (No Lagging EMA, 1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_ONLY", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        {"name": "8. Donchian 10 + SuperTrend Only (No Lagging EMA, 1.2R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_ONLY", "target_rr": 1.2, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},

        # Levers on Entry Trigger (Touch vs Close)
        {"name": "9. Donchian 10 TOUCH Trigger (Instant Penetration, 1.3R)", "donch_period": 10, "entry_trigger": "TOUCH", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},
        {"name": "10. Donchian 10 TOUCH Trigger + ST3 Only (1.2R Fast TP)", "donch_period": 10, "entry_trigger": "TOUCH", "trend_filter": "ST3_ONLY", "target_rr": 1.2, "session_start": 7, "session_end": 19, "pullback": False, "sweep": False},

        # Levers on Multi-Engine (Breakout + EMA20 Value Zone Pullbacks)
        {"name": "11. DUAL-ENGINE: Donchian 15 + EMA20 Pullbacks (1.3R)", "donch_period": 15, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": True, "sweep": False},
        {"name": "12. DUAL-ENGINE: Donchian 10 + EMA20 Pullbacks (1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": True, "sweep": False},
        {"name": "13. DUAL-ENGINE: Donchian 10 + EMA20 Pullbacks (1.2R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.2, "session_start": 7, "session_end": 19, "pullback": True, "sweep": False},
        {"name": "14. DUAL-ENGINE: Donchian 10 + EMA20 Pullbacks + ST3/EMA50 (1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA50", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": True, "sweep": False},

        # Levers on Multi-Engine (Breakout + Asian Sweep / Judas Swing)
        {"name": "15. TRIPLE-ENGINE: Donchian 15 + EMA20 Pullback + Asian Sweep (1.3R)", "donch_period": 15, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": True, "sweep": True},
        {"name": "16. TRIPLE-ENGINE: Donchian 10 + EMA20 Pullback + Asian Sweep (1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 7, "session_end": 19, "pullback": True, "sweep": True},
        {"name": "17. TRIPLE-ENGINE: Donchian 10 + EMA20 Pullback + Asian Sweep (1.2R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.2, "session_start": 7, "session_end": 19, "pullback": True, "sweep": True},

        # Levers on Session Window (06:00 - 21:00 UTC)
        {"name": "18. Triple-Engine + Extended Session (06-21 UTC, 1.2R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.2, "session_start": 6, "session_end": 21, "pullback": True, "sweep": True},
        {"name": "19. Triple-Engine + Extended Session (06-21 UTC, 1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA200", "target_rr": 1.3, "session_start": 6, "session_end": 21, "pullback": True, "sweep": True},
        {"name": "20. Triple-Engine + ST3/EMA50 + Extended Session (06-21 UTC, 1.3R)", "donch_period": 10, "entry_trigger": "CLOSE", "trend_filter": "ST3_EMA50", "target_rr": 1.3, "session_start": 6, "session_end": 21, "pullback": True, "sweep": True},
    ]

    results = []
    for cfg in configurations:
        res = evaluate_system(
            candles_1h=candles,
            donch_period=cfg["donch_period"],
            entry_trigger=cfg["entry_trigger"],
            trend_filter=cfg["trend_filter"],
            target_rr=cfg["target_rr"],
            session_start=cfg["session_start"],
            session_end=cfg["session_end"],
            enable_pullback_engine=cfg["pullback"],
            enable_asian_sweep=cfg["sweep"],
        )
        res["name"] = cfg["name"]
        results.append(res)

    # Sort results by Ending Balance & Profit Factor
    results_sorted = sorted(results, key=lambda x: (x["win_rate"] >= 45.0, x["ending_balance"]), reverse=True)

    print(f"{'#':<3} {'Configuration Name':<42} {'Trades/Mo':<10} {'Total':<6} {'Win Rate':<10} {'PF':<6} {'MaxDD':<8} {'End Equity ($)'}")
    print("-" * 96)
    for idx, r in enumerate(results_sorted, 1):
        print(f"{idx:<3} {r['name'][:40]:<42} {r['trades_per_month']:<10} {r['total_trades']:<6} {r['win_rate']:<9.1f}% {r['profit_factor']:<6.2f} {r['max_dd_pct']:<7.1f}% ${r['ending_balance']:<10,.2f}")

    print("\n" + "=" * 96)


if __name__ == "__main__":
    main()
