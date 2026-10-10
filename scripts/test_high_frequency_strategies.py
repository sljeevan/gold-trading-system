"""
Quantitative High-Frequency Gold Strategy Engine:
Evaluates intraday trade frequency (1-4 trades/day) and win rate (> 55%) on 5m & 15m candles.
"""
from __future__ import annotations

import json
import numpy as np

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_bollinger,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)


def simulate_intraday_bracket(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    entry_idx: int,
    direction: str,
    entry_price: float,
    tp_points: float = 2.00,
    sl_points: float = 1.54,
    spread_pts: float = 0.35,
    be_trigger_points: float = 1.00,
    max_bars: int = 36,
):
    n = len(closes)
    eff_entry = entry_price + spread_pts if direction == "BUY" else entry_price
    tp = eff_entry + tp_points if direction == "BUY" else eff_entry - tp_points - spread_pts
    sl = eff_entry - sl_points if direction == "BUY" else eff_entry + sl_points + spread_pts
    be_price = eff_entry + be_trigger_points if direction == "BUY" else eff_entry - be_trigger_points

    is_be = False
    won, lost, scratched = False, False, False
    bars_held = 0

    for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
        bars_held = k - entry_idx
        o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
        curr_sl = eff_entry if is_be else sl

        if direction == "BUY":
            if c_k >= o_k:  # O -> L -> H -> C
                if l_k <= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
                if h_k >= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if h_k >= tp: won = True; break
            else:  # O -> H -> L -> C
                if h_k >= tp: won = True; break
                if h_k >= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if l_k <= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
        else:  # SELL
            if c_k <= o_k:  # O -> H -> L -> C
                if h_k >= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
                if l_k <= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if l_k <= tp: won = True; break
            else:  # O -> L -> H -> C
                if l_k <= tp: won = True; break
                if l_k <= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if h_k >= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break

    if not (won or lost or scratched):
        # Time exit
        last_c = closes[min(entry_idx + max_bars, n - 1)]
        if direction == "BUY":
            won = last_c > eff_entry
            lost = not won
        else:
            won = eff_entry > last_c
            lost = not won

    return ("WIN" if won else "SCRATCH" if scratched else "LOSS"), bars_held


def test_timeframe(timeframe: str = "15m", period: str = "60d"):
    candles = fetch_gold_candles("XAUUSD", period=period, interval=timeframe)
    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # Technical Indicators
    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    rsi = calc_rsi(closes.tolist(), 14)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)

    # -------------------------------------------------------------------------
    # Setup 1: Dynamic Value Zone Pullback (EMA 9 / 21 in Trend)
    # Price pulls back between EMA 9 and EMA 21 in trend direction
    # -------------------------------------------------------------------------
    sigs_value_zone = []
    for i in range(200, n - 36):
        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19): continue  # Liquid trading day

        e9, e21, e50, e200 = ema9[i], ema21[i], ema50[i], ema200[i]
        st_d = st["direction"][i]
        r = rsi[i]
        if None in (e9, e21, e50, e200, st_d, r): continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        rng = max(0.01, h - l)

        # Bullish: Trend Up (c > e50 and e9 > e21) + Low dipped between e9 & e21 + closed green
        if c > e50 and e9 > e21 and (l <= e9 and c >= e21):
            if c >= o and (45 <= r <= 65):
                sigs_value_zone.append({"idx": i, "dir": "BUY", "price": c, "date": dt})

        # Bearish: Trend Down (c < e50 and e9 < e21) + High spiked between e9 & e21 + closed red
        if c < e50 and e9 < e21 and (h >= e9 and c <= e21):
            if c <= o and (35 <= r <= 55):
                sigs_value_zone.append({"idx": i, "dir": "SELL", "price": c, "date": dt})

    # -------------------------------------------------------------------------
    # Setup 2: Micro Liquidity Sweep (Sweep of 5-candle Low/High + Rejection Wick)
    # -------------------------------------------------------------------------
    sigs_micro_sweep = []
    for i in range(50, n - 36):
        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19): continue

        e50 = ema50[i]
        if e50 is None: continue
        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        rng = max(0.01, h - l)
        low_5 = np.min(lows[i-5:i])
        high_5 = np.max(highs[i-5:i])

        # Bullish sweep in uptrend
        if c > e50 and l < low_5 and c > low_5:
            lower_wick = min(o, c) - l
            if lower_wick / rng >= 0.40 and c >= o:
                sigs_micro_sweep.append({"idx": i, "dir": "BUY", "price": c, "date": dt})

        # Bearish sweep in downtrend
        if c < e50 and h > high_5 and c < high_5:
            upper_wick = h - max(o, c)
            if upper_wick / rng >= 0.40 and c <= o:
                sigs_micro_sweep.append({"idx": i, "dir": "SELL", "price": c, "date": dt})

    # -------------------------------------------------------------------------
    # Setup 3: RSI Extreme Pullback in Trend (RSI < 38 in Uptrend / > 62 in Downtrend)
    # -------------------------------------------------------------------------
    sigs_rsi_extreme = []
    for i in range(50, n - 36):
        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19): continue

        e50 = ema50[i]
        r = rsi[i]
        rp = rsi[i-1]
        if None in (e50, r, rp): continue
        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        # In Uptrend, RSI was oversold (<38) and hooked up
        if c > e50 and rp <= 38 and r > rp and c >= o:
            sigs_rsi_extreme.append({"idx": i, "dir": "BUY", "price": c, "date": dt})

        # In Downtrend, RSI was overbought (>62) and hooked down
        if c < e50 and rp >= 62 and r < rp and c <= o:
            sigs_rsi_extreme.append({"idx": i, "dir": "SELL", "price": c, "date": dt})

    # -------------------------------------------------------------------------
    # Setup 4: Combined Multi-Trigger Strategy Engine (Active Intraday Scanner)
    # Merges all 3 setups (deduplicating overlapping candles)
    # -------------------------------------------------------------------------
    combined_dict = {}
    for s in (sigs_value_zone + sigs_micro_sweep + sigs_rsi_extreme):
        idx = s["idx"]
        if idx not in combined_dict:
            combined_dict[idx] = s
    sigs_combined = sorted(combined_dict.values(), key=lambda x: x["idx"])

    # Evaluate Each Setup
    setups = [
        ("Setup 1: EMA 9/21 Value Zone Pullback", sigs_value_zone),
        ("Setup 2: Micro Liquidity Sweep (5-bar)", sigs_micro_sweep),
        ("Setup 3: RSI Extreme Dip in Trend", sigs_rsi_extreme),
        ("Setup 4: Combined Multi-Trigger Engine (All Active)", sigs_combined),
    ]

    trading_days = 42 # ~42 active trading days in 60 calendar days
    print(f"\n==========================================================================================")
    print(f"📊 HIGH-FREQUENCY STRATEGY EVALUATION — {timeframe.upper()} TIMEFRAME ({period})")
    print(f"==========================================================================================")
    print(f"{'Strategy Setup':<45} | {'Trades':<6} | {'Trades/Day':<10} | {'Wins':<5} | {'Losses':<6} | {'Scratches':<9} | {'Win Rate'}")
    print("-" * 105)

    for name, sigs in setups:
        wins, losses, scratches, executed = 0, 0, 0, 0
        act = -1
        for s in sigs:
            idx = s["idx"]
            if idx <= act: continue
            out, bars = simulate_intraday_bracket(
                highs, lows, closes, opens, idx, s["dir"], s["price"],
                tp_points=2.00, sl_points=1.54, spread_pts=0.35, be_trigger_points=1.00
            )
            act = idx + bars
            executed += 1
            if out == "WIN": wins += 1
            elif out == "SCRATCH": scratches += 1
            elif out == "LOSS": losses += 1

        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
        t_per_day = executed / trading_days
        print(f"{name:<45} | {executed:<6} | {t_per_day:>8.1f}/d | {wins:<5} | {losses:<6} | {scratches:<9} | {wr:>6.1f}%")

    return sigs_combined


if __name__ == "__main__":
    print("Testing 15-Minute Candles:")
    test_timeframe("15m", "60d")
    print("\nTesting 5-Minute Candles:")
    test_timeframe("5m", "60d")
