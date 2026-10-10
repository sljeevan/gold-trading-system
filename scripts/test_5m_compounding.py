"""
5-Minute Intraday Backtest Simulator for the 30-Level Gold Compounding Ladder ($20 -> $52,384).
Compares 5-Minute execution dynamics against 15-Minute execution:
- Tests sensitivity across Limit Pullback Discounts (0 pips market, 4 pips, 6 pips, 8 pips)
- Evaluates trade frequency, win rates, breakeven scratch efficiency, Level 30 completion count, and reloads.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
from datetime import datetime
from typing import Any, Dict, List

from gold_trader.core.compounding_ladder import LADDER_MATRIX, get_ladder_level_info
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_supertrend, calc_rsi


def run_5m_study():
    print("Loading 5m (60-day) and 1h (60-day) historical Gold candles...")
    candles_5m = fetch_gold_candles("XAUUSD", period="60d", interval="5m")
    candles_1h = fetch_gold_candles("XAUUSD", period="60d", interval="1h")

    n = len(candles_5m)
    print(f"Total 5m candles loaded: {n} (from {candles_5m[0]['date']} to {candles_5m[-1]['date']})")

    highs = np.array([float(c["high"]) for c in candles_5m], dtype=float)
    lows = np.array([float(c["low"]) for c in candles_5m], dtype=float)
    closes = np.array([float(c["close"]) for c in candles_5m], dtype=float)
    opens = np.array([float(c["open"]) for c in candles_5m], dtype=float)
    dates = [c["date"] for c in candles_5m]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)

    # 1H trend alignment lookup
    h1_closes = [float(c["close"]) for c in candles_1h]
    h1_highs = [float(c["high"]) for c in candles_1h]
    h1_lows = [float(c["low"]) for c in candles_1h]
    h1_ema50 = calc_ema(h1_closes, 50)
    h1_st = calc_supertrend(h1_highs, h1_lows, h1_closes, 10, 3.0)
    h1_dates = [c["date"] for c in candles_1h]

    h1_trend_map = {}
    for idx, d in enumerate(h1_dates):
        key = d[:13]
        c_val = h1_closes[idx]
        e_val = h1_ema50[idx]
        s_val = h1_st["direction"][idx] if h1_st["direction"] else 0
        bull = bool(e_val and c_val > e_val and s_val == 1)
        bear = bool(e_val and c_val < e_val and s_val == -1)
        h1_trend_map[key] = (bull, bear)

    avg_range = float(np.mean(highs - lows))
    print(f"Average 5m candle range: ${avg_range:.2f}")

    def simulate(disc_pips: float):
        disc = disc_pips * 0.10
        spread = 0.35
        level = 1
        balance = 20.0
        peak_level = 1
        peak_balance = 20.0
        completions = 0
        reloads = 0
        wins, losses, scratches = 0, 0, 0
        total_banked = 0.0
        comp_history = []
        act = -1

        for i in range(200, n - 40):
            if i <= act:
                continue
            dt = dates[i]
            try:
                sep = "T" if "T" in dt else " "
                hour = int(dt.split(sep)[1].split(":")[0])
            except Exception:
                hour = 12

            if not (7 <= hour <= 19):
                continue

            h_key = dt[:13]
            h_bull, h_bear = h1_trend_map.get(h_key, (True, True))

            e9_i, e21_i, e50_i = ema9[i], ema21[i], ema50[i]
            r = rsi[i]
            if None in (e9_i, e21_i, e50_i, r):
                continue
            c, h, l, o = closes[i], highs[i], lows[i], opens[i]

            dir_ = None
            if h_bull and c > e50_i and e9_i > e21_i and (l <= e9_i and c >= e21_i) and c >= o and (45 <= r <= 65):
                dir_ = "BUY"
            elif h_bear and c < e50_i and e9_i < e21_i and (h >= e9_i and c <= e21_i) and c <= o and (35 <= r <= 55):
                dir_ = "SELL"

            if not dir_:
                continue

            k_fill = i + 1
            limit_p = c - disc if dir_ == "BUY" else c + disc
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
            be_price = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0

            won, lost, scratched = False, False, False
            is_be = False
            bars = 0
            for k in range(k_fill, min(k_fill + 36, n)):
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
                        if h_k >= be_price:
                            is_be = True
                        curr_sl = eff_entry if is_be else sl
                        if h_k >= tp:
                            won = True
                            break
                    else:
                        if h_k >= tp:
                            won = True
                            break
                        if h_k >= be_price:
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
                        if l_k <= be_price:
                            is_be = True
                        curr_sl = eff_entry if is_be else sl
                        if l_k <= tp:
                            won = True
                            break
                    else:
                        if l_k <= tp:
                            won = True
                            break
                        if l_k <= be_price:
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
                balance += lvl_info["profit_dollars"]
                if level == 30:
                    completions += 1
                    total_banked += balance
                    comp_history.append({
                        "date": dt,
                        "trade_no": wins + losses + scratches,
                        "balance": round(balance, 2),
                    })
                    level = 1
                    balance = 20.0
                else:
                    level += 1
                    peak_level = max(peak_level, level)
            elif scratched:
                scratches += 1
            elif lost:
                losses += 1
                balance -= lvl_info["risk_dollars"]
                if level > 1:
                    level -= 1
                else:
                    if balance < 5.0:
                        reloads += 1
                        balance = 20.0
                        level = 1
            peak_balance = max(peak_balance, balance)

        tot = wins + losses + scratches
        dec = wins + losses
        wr = (wins / dec * 100) if dec else 0.0
        return {
            "disc_pips": disc_pips,
            "total_trades": tot,
            "trades_per_day": round(tot / 42.0, 1),
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "win_rate_pct": round(wr, 2),
            "completions": completions,
            "reloads": reloads,
            "peak_level": peak_level,
            "peak_balance": round(peak_balance, 2),
            "total_banked": round(total_banked + balance, 2),
            "history": comp_history,
        }

    results = {}
    print("\n--- Testing 5m Intervals across Pullback Discounts ---")
    for d in [0.0, 4.0, 6.0, 8.0]:
        res = simulate(d)
        results[f"disc_{d}pips"] = res
        print(f"Discount {d:.1f} pips: Total={res['total_trades']} ({res['trades_per_day']}/day) | WR={res['win_rate_pct']}% | Scratches={res['scratches']} | L30 Compl={res['completions']} | Reloads={res['reloads']} | Peak L={res['peak_level']} | Banked=${res['total_banked']:,.2f}")
        for h in res["history"]:
            print(f"   ⭐ Level 30 Achieved on {h['date']} (Trade #{h['trade_no']}) with ${h['balance']:,.2f}")

    out_file = os.path.expanduser("~/.gold_trading_data/5m_compounding_study_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved 5m study results to {out_file}")


if __name__ == "__main__":
    run_5m_study()
