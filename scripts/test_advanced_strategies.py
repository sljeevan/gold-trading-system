"""
Advanced Gold Strategies Backtest: Testing High Win-Rate Confluence Models on XAUUSD
Evaluates 5 advanced institutional and algorithmic models over the past 6 months (March - September 2026).
"""
import sys
import numpy as np
import json
import os

sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_bollinger,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)

def run_advanced_tests():
    print("=" * 80)
    print("Testing Advanced High Win-Rate Gold Strategies on 6-Month XAUUSD Data...")
    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    if not candles:
        print("Failed to fetch candles.")
        return

    n = len(candles)
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)
    donch20 = calc_donchian(highs.tolist(), lows.tolist(), 20)

    INITIAL_CAPITAL = 10000.0
    RISK_DOLLARS = 100.0
    SPREAD = 0.35

    # -------------------------------------------------------------------------
    # Model 1: ICT Silver Bullet (Killzone + Liquidity Sweep + MSS + 1.5R)
    # -------------------------------------------------------------------------
    sig_silver_bullet = []
    curr_day = None
    asia_high, asia_low = -1.0, 999999.0

    for i in range(50, n - 24):
        dt_str = dates[i]
        day = dt_str.split("T" if "T" in dt_str else " ")[0]
        hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])

        if day != curr_day:
            curr_day = day
            asia_high, asia_low = -1.0, 999999.0

        if 0 <= hour <= 6:
            asia_high = max(asia_high, highs[i])
            asia_low = min(asia_low, lows[i])

        # Killzones: London Open (07-10 UTC) or NY AM (12-15 UTC)
        if (7 <= hour <= 10 or 12 <= hour <= 15) and asia_high > 0:
            c, h, l, o = closes[i], highs[i], lows[i], opens[i]
            prev_c, prev_o = closes[i-1], opens[i-1]
            a = atr[i] if atr[i] else 10.0

            # Bearish Sweep + Engulfing Reversal:
            # Swept above Asian high, but this bar is a strong bearish engulfing candle
            if h > asia_high and c < o and c < prev_o and (o - c) >= 0.5 * a:
                sl = h + 1.50
                tp = c - 1.5 * (sl - c)  # 1.5:1 RR
                sig_silver_bullet.append({"idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp})

            # Bullish Sweep + Engulfing Reversal:
            elif l < asia_low and c > o and c > prev_o and (c - o) >= 0.5 * a:
                sl = l - 1.50
                tp = c + 1.5 * (c - sl)  # 1.5:1 RR
                sig_silver_bullet.append({"idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp})

    # -------------------------------------------------------------------------
    # Model 2: Multi-Timeframe Trend Pullback + Bullish/Bearish Engulfing Confirmation
    # -------------------------------------------------------------------------
    sig_mtf_engulfing = []
    for i in range(200, n - 24):
        e20, e50, e200, a = ema20[i], ema50[i], ema200[i], atr[i]
        if None in (e20, e50, e200, a): continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        prev_c, prev_o = closes[i-1], opens[i-1]

        # Bullish: Trend Up (Price > 200 EMA & 20 > 50 EMA), pullback to 20/50 zone, confirmed by Bullish Engulfing
        if c > e200 and e20 > e50 and (l <= e20 or l <= e50):
            if c > o and prev_c < prev_o and c > prev_o and o <= prev_c:
                sl = l - 1.50
                tp = c + 1.5 * (c - sl)  # 1.5:1 RR
                sig_mtf_engulfing.append({"idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp})

        # Bearish: Trend Down (Price < 200 EMA & 20 < 50 EMA), pullback to 20/50 zone, confirmed by Bearish Engulfing
        elif c < e200 and e20 < e50 and (h >= e20 or h >= e50):
            if c < o and prev_c > prev_o and c < prev_o and o >= prev_c:
                sl = h + 1.50
                tp = c - 1.5 * (sl - c)  # 1.5:1 RR
                sig_mtf_engulfing.append({"idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp})

    # -------------------------------------------------------------------------
    # Model 3: Donchian 20 Breakout + SuperTrend Directional Confluence (Optimized)
    # -------------------------------------------------------------------------
    sig_donchian_st = []
    for i in range(200, n - 24):
        d_up = donch20["upper"][i-1]
        d_low = donch20["lower"][i-1]
        st_dir = st["direction"][i]
        e200 = ema200[i]
        a = atr[i]
        if None in (d_up, d_low, st_dir, e200, a): continue
        c, h, l = closes[i], highs[i], lows[i]

        # Breakout confirmed by SuperTrend and 200 EMA
        if h > d_up and st_dir == 1 and c > e200:
            sl = c - max(1.5 * a, 8.0)
            tp = c + 1.8 * (c - sl)
            sig_donchian_st.append({"idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp})
        elif l < d_low and st_dir == -1 and c < e200:
            sl = c + max(1.5 * a, 8.0)
            tp = c - 1.8 * (sl - c)
            sig_donchian_st.append({"idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp})

    # -------------------------------------------------------------------------
    # Model 4: 200 EMA Macro Trend + RSI Pullback Exhaustion (50-60% WR Target)
    # -------------------------------------------------------------------------
    sig_rsi_trend_pullback = []
    for i in range(200, n - 24):
        e200 = ema200[i]
        r = rsi[i]
        prev_r = rsi[i-1]
        a = atr[i]
        if None in (e200, r, prev_r, a): continue
        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        # Strong Uptrend + RSI dip below 40 then turns up (deep pullback in bull market)
        if c > e200 and prev_r < 40 and r >= 40 and c > o:
            sl = l - max(1.2 * a, 7.0)
            tp = c + 1.5 * (c - sl)
            sig_rsi_trend_pullback.append({"idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp})

        # Strong Downtrend + RSI spike above 60 then turns down (deep pullback in bear market)
        elif c < e200 and prev_r > 60 and r <= 60 and c < o:
            sl = h + max(1.2 * a, 7.0)
            tp = c - 1.5 * (sl - c)
            sig_rsi_trend_pullback.append({"idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp})

    models = [
        ("1. Multi-Timeframe Trend + Engulfing Reversal", sig_mtf_engulfing),
        ("2. 200 EMA + RSI Trend Dip-Buyer", sig_rsi_trend_pullback),
        ("3. Donchian 20 Breakout + SuperTrend Confluence", sig_donchian_st),
        ("4. ICT Silver Bullet (Liquidity Sweep + Engulfing)", sig_silver_bullet),
    ]

    print("\n" + "=" * 95)
    print(f"{'Strategy Model':<45} | {'Trades':<6} | {'Win Rate':<9} | {'PF':<5} | {'Net P&L ($)':<12} | {'Max DD %':<8}")
    print("-" * 95)

    for name, sigs in models:
        wins, losses = 0, 0
        gross_p, gross_l = 0.0, 0.0
        active_until = -1
        balance = INITIAL_CAPITAL
        peak = INITIAL_CAPITAL
        max_dd = 0.0

        for s in sigs:
            if s["idx"] <= active_until: continue
            dir_ = s["dir"]
            c = s["price"]
            sl = s["sl"]
            tp = s["tp"]

            if dir_ == "BUY":
                eff_entry = c + SPREAD
                risk_pts = eff_entry - sl
                if risk_pts <= 1.0: continue
                lots = round(RISK_DOLLARS / (risk_pts * 100), 2)
                lots = max(0.01, min(lots, 2.0))

                won = False
                exit_p = eff_entry
                bars_held = 0
                for k in range(s["idx"] + 1, min(s["idx"] + 48, n)):
                    bars_held = k - s["idx"]
                    if lows[k] <= sl:
                        won = False; exit_p = sl; break
                    if highs[k] >= tp:
                        won = True; exit_p = tp; break
                else:
                    exit_p = closes[min(s["idx"] + 48, n - 1)]
                    won = exit_p > eff_entry

                pnl = (exit_p - eff_entry) * lots * 100

            else:
                eff_entry = c
                risk_pts = sl - eff_entry
                if risk_pts <= 1.0: continue
                lots = round(RISK_DOLLARS / (risk_pts * 100), 2)
                lots = max(0.01, min(lots, 2.0))

                won = False
                exit_p = eff_entry
                bars_held = 0
                for k in range(s["idx"] + 1, min(s["idx"] + 48, n)):
                    bars_held = k - s["idx"]
                    if (highs[k] + SPREAD) >= sl:
                        won = False; exit_p = sl; break
                    if (lows[k] + SPREAD) <= tp:
                        won = True; exit_p = tp; break
                else:
                    exit_p = closes[min(s["idx"] + 48, n - 1)] + SPREAD
                    won = eff_entry > exit_p

                pnl = (eff_entry - exit_p) * lots * 100

            active_until = s["idx"] + bars_held
            if pnl > 0:
                wins += 1; gross_p += pnl
            else:
                losses += 1; gross_l += abs(pnl)

            balance += pnl
            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100.0 if peak > 0 else 0
            if dd > max_dd: max_dd = dd

        total_t = wins + losses
        wr = (wins / total_t * 100) if total_t > 0 else 0.0
        pf = (gross_p / gross_l) if gross_l > 0 else (99.0 if gross_p > 0 else 0.0)
        net_pnl = gross_p - gross_l
        pnl_s = f"+${net_pnl:,.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):,.2f}"
        print(f"{name:<45} | {total_t:<6} | {wr:>7.1f}%  | {pf:>5.2f} | {pnl_s:>12} | {max_dd:>7.1f}%")

    print("=" * 95)

if __name__ == "__main__":
    run_advanced_tests()
