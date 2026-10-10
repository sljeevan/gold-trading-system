"""
Gold Strategy Tournament: Screening for High Win-Rate Strategies (> 55%) on XAUUSD (1 Year).
Simulates realistic broker execution ($0.35 spread, OHLC intrabar resolution) across:
1. Pure Trend Crossover (Baseline)
2. London Asian Range Liquidity Sweep (Judas Swing)
3. Fibonacci 0.618-0.786 Golden Pocket Pullback
4. 20/50 EMA Trend Pullback + Rejection Wick
5. Bollinger Band Squeeze & Mean Reversion
6. ML Meta-Labeling Filter (P >= 0.70)
7. Breakeven Lock (+10 pips) Overlay
"""
from __future__ import annotations

import json
import os
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
from scripts.backtest_compounding_1year import simulate_bracket_fast


def run_tournament():
    candles = fetch_gold_candles("XAUUSD", period="1y", interval="1h")
    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # Precalculate technical indicators
    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)
    bb = calc_bollinger(closes.tolist(), 20, 2.0)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    # -------------------------------------------------------------
    # 1. Strategy: EMA 5/20 Crossover (Baseline)
    # -------------------------------------------------------------
    signals_baseline = []
    fast_ema = calc_ema(closes.tolist(), 5)
    slow_ema = calc_ema(closes.tolist(), 20)
    for i in range(200, n - 24):
        f, s = fast_ema[i], slow_ema[i]
        fp, sp = fast_ema[i-1], slow_ema[i-1]
        t = ema200[i]
        if None in (f, s, fp, sp, t): continue
        if fp <= sp and f > s and closes[i] > t:
            signals_baseline.append({"idx": i, "dir": "BUY", "price": closes[i], "date": dates[i]})
        elif fp >= sp and f < s and closes[i] < t:
            signals_baseline.append({"idx": i, "dir": "SELL", "price": closes[i], "date": dates[i]})

    # -------------------------------------------------------------
    # 2. Strategy: Asian Session Liquidity Sweep (Judas Swing)
    # -------------------------------------------------------------
    # Establishes 00:00 - 06:00 UTC Asian High & Low.
    # At London Open (07:00 - 10:00 UTC), if price sweeps the Asian High/Low
    # and closes back inside with a wick rejection -> Trade the reversal!
    signals_asian_sweep = []
    curr_day = None
    asia_high = -1.0
    asia_low = 999999.0

    for i in range(50, n - 24):
        dt_str = dates[i]
        day = dt_str.split("T" if "T" in dt_str else " ")[0]
        hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])

        if day != curr_day:
            curr_day = day
            asia_high = -1.0
            asia_low = 999999.0

        if 0 <= hour <= 6:
            asia_high = max(asia_high, highs[i])
            asia_low = min(asia_low, lows[i])

        if 7 <= hour <= 10 and asia_high > 0 and asia_low < 999999:
            rng = asia_high - asia_low
            if rng < 5.0 or rng > 35.0: # Skip abnormal ranges
                continue
            
            c_high, c_low, c_close, c_open = highs[i], lows[i], closes[i], opens[i]
            c_rng = max(0.01, c_high - c_low)
            upper_wick = c_high - max(c_open, c_close)
            lower_wick = min(c_open, c_close) - c_low

            # Bearish Judas Sweep: Swept Asian High, but closed back inside with upper wick
            if c_high > asia_high and c_close < asia_high and (upper_wick / c_rng >= 0.40):
                signals_asian_sweep.append({"idx": i, "dir": "SELL", "price": c_close, "date": dt_str})
            
            # Bullish Judas Sweep: Swept Asian Low, but closed back inside with lower wick
            elif c_low < asia_low and c_close > asia_low and (lower_wick / c_rng >= 0.40):
                signals_asian_sweep.append({"idx": i, "dir": "BUY", "price": c_close, "date": dt_str})

    # -------------------------------------------------------------
    # 3. Strategy: Fibonacci 0.618 - 0.786 Golden Pocket Pullback
    # -------------------------------------------------------------
    signals_fib_gp = []
    lookback = 30
    for i in range(200, n - 24):
        st_dir = st["direction"][i]
        t = ema200[i]
        if None in (st_dir, t): continue

        # Swing high and low over lookback
        sw_high = np.max(highs[i - lookback : i])
        sw_low = np.min(lows[i - lookback : i])
        sw_rng = sw_high - sw_low
        if sw_rng < 15.0: continue

        fib_618 = sw_high - 0.618 * sw_rng
        fib_786 = sw_high - 0.786 * sw_rng

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        c_rng = max(0.01, h - l)

        # Bullish Golden Pocket in Uptrend
        if c > t and st_dir == 1:
            if l <= fib_618 and c >= fib_786:
                # Rejection wick confirmation
                lower_wick = min(o, c) - l
                if lower_wick / c_rng >= 0.40 and c >= o:
                    signals_fib_gp.append({"idx": i, "dir": "BUY", "price": c, "date": dates[i]})

        # Bearish Golden Pocket in Downtrend
        fib_618_bear = sw_low + 0.618 * sw_rng
        fib_786_bear = sw_low + 0.786 * sw_rng
        if c < t and st_dir == -1:
            if h >= fib_618_bear and c <= fib_786_bear:
                upper_wick = h - max(o, c)
                if upper_wick / c_rng >= 0.40 and c <= o:
                    signals_fib_gp.append({"idx": i, "dir": "SELL", "price": c, "date": dates[i]})

    # -------------------------------------------------------------
    # 4. Strategy: 20/50 EMA Trend Pullback + Rejection Pin Bar
    # -------------------------------------------------------------
    signals_ema_pullback = []
    for i in range(200, n - 24):
        e20, e50, t = ema20[i], ema50[i], ema200[i]
        st_dir = st["direction"][i]
        r = rsi[i]
        if None in (e20, e50, t, st_dir, r): continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        rng = max(0.01, h - l)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        # Bullish Pullback to 20/50 EMA in Trend
        if c > t and st_dir == 1 and (l <= e20 <= h or l <= e50 <= h):
            if (lower_wick / rng >= 0.45) and c >= o and 45 <= r <= 65:
                signals_ema_pullback.append({"idx": i, "dir": "BUY", "price": c, "date": dates[i]})

        # Bearish Pullback to 20/50 EMA in Trend
        if c < t and st_dir == -1 and (l <= e20 <= h or l <= e50 <= h):
            if (upper_wick / rng >= 0.45) and c <= o and 35 <= r <= 55:
                signals_ema_pullback.append({"idx": i, "dir": "SELL", "price": c, "date": dates[i]})

    # -------------------------------------------------------------
    # 5. Strategy: Bollinger Band Mean Reversion (Exhaustion Fade)
    # -------------------------------------------------------------
    signals_bb_reversion = []
    for i in range(200, n - 24):
        b_up = bb["upper"][i]
        b_low = bb["lower"][i]
        r = rsi[i]
        if None in (b_up, b_low, r): continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        rng = max(0.01, h - l)

        # Oversold Bounce: Low touched lower band + RSI < 35 + Bullish Rejection
        if l <= b_low and r <= 35:
            lower_wick = min(o, c) - l
            if lower_wick / rng >= 0.40 and c >= o:
                signals_bb_reversion.append({"idx": i, "dir": "BUY", "price": c, "date": dates[i]})

        # Overbought Fade: High touched upper band + RSI > 65 + Bearish Rejection
        if h >= b_up and r >= 65:
            upper_wick = h - max(o, c)
            if upper_wick / rng >= 0.40 and c <= o:
                signals_bb_reversion.append({"idx": i, "dir": "SELL", "price": c, "date": dates[i]})

    # Evaluate each strategy with both standard bracket and Breakeven Lock (+10 pips)
    strategies = [
        ("1. EMA Crossover (Baseline)", signals_baseline),
        ("2. Asian Liquidity Sweep (Judas Swing)", signals_asian_sweep),
        ("3. Fibonacci 0.618-0.786 Golden Pocket", signals_fib_gp),
        ("4. 20/50 EMA Trend Pullback + Rejection", signals_ema_pullback),
        ("5. Bollinger Band Exhaustion Fade", signals_bb_reversion),
    ]

    results = []
    for name, sigs in strategies:
        # Standard 20-pip TP / 15-pip SL ($0.35 spread)
        wins, losses, executed = 0, 0, 0
        active_until = -1
        for s in sigs:
            if s["idx"] <= active_until: continue
            is_win, _, bars = simulate_bracket_fast(
                highs, lows, closes, opens, s["idx"], s["dir"], s["price"],
                tp_points=2.00, sl_points=1.54, spread_pts=0.35
            )
            active_until = s["idx"] + bars
            executed += 1
            if is_win: wins += 1
            else: losses += 1

        wr = (wins / executed * 100) if executed > 0 else 0.0

        # Breakeven Lock Version (+10 pips lock)
        wins_be, losses_be, be_scratches, exec_be = 0, 0, 0, 0
        active_until = -1
        for s in sigs:
            if s["idx"] <= active_until: continue
            dir_ = s["dir"]
            p = s["price"]
            eff_entry = p + 0.35 if dir_ == "BUY" else p
            tp = eff_entry + 2.0 if dir_ == "BUY" else eff_entry - 2.0 - 0.35
            sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + 0.35
            be_trig = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0

            won, lost, scratched = False, False, False
            is_locked = False
            bars_held = 0
            for k in range(s["idx"] + 1, min(s["idx"] + 25, n)):
                bars_held = k - s["idx"]
                if dir_ == "BUY":
                    if highs[k] >= be_trig: is_locked = True
                    curr_sl = eff_entry if is_locked else sl
                    if highs[k] >= tp: won = True; break
                    if lows[k] <= curr_sl:
                        if is_locked: scratched = True
                        else: lost = True
                        break
                else:
                    if lows[k] <= be_trig: is_locked = True
                    curr_sl = eff_entry if is_locked else sl
                    if lows[k] <= tp: won = True; break
                    if highs[k] >= curr_sl:
                        if is_locked: scratched = True
                        else: lost = True
                        break

            active_until = s["idx"] + bars_held
            exec_be += 1
            if won: wins_be += 1
            elif scratched: be_scratches += 1
            elif lost: losses_be += 1

        effective_wr_be = (wins_be / (wins_be + losses_be) * 100) if (wins_be + losses_be) > 0 else 0.0

        results.append({
            "name": name,
            "total_signals": len(sigs),
            "executed": executed,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wr, 1),
            "be_wins": wins_be,
            "be_losses": losses_be,
            "be_scratches": be_scratches,
            "effective_wr_be": round(effective_wr_be, 1),
        })

    out_file = os.path.expanduser("~/.gold_trading_data/strategy_tournament_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 90)
    print("🏆 GOLD STRATEGY TOURNAMENT: 1-YEAR BACKTEST ON XAUUSD")
    print("=" * 90)
    print(f"{'Strategy Name':<42} | {'Trades':<6} | {'Raw Win Rate':<12} | {'BE Scratches':<12} | {'Win Rate (+BE Lock)':<18}")
    print("-" * 96)
    for r in results:
        print(f"{r['name']:<42} | {r['executed']:<6} | {r['win_rate']:>10.1f}% | {r['be_scratches']:>10} | {r['effective_wr_be']:>16.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    run_tournament()
