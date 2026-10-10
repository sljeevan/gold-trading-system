"""
Gold Strategy 6-Month Tournament: Comprehensive Comparative Backtest.
Evaluates 6 widely used institutional and retail Gold trading strategies on XAUUSD
over the past 6 months (March 27, 2026 - September 25, 2026).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
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


def run_6mo_tournament():
    print("=" * 80)
    print("Fetching past 6 months of historical Gold (XAUUSD) 1-Hour candles...")
    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    if not candles or len(candles) < 500:
        print(f"Error: Only {len(candles) if candles else 0} candles retrieved.")
        return

    n = len(candles)
    start_date = candles[0]["date"]
    end_date = candles[-1]["date"]
    print(f"Loaded {n} candles: {start_date} to {end_date}")

    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    volumes = np.array([float(c.get("volume", 0)) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # Precalculate technical indicators
    ema9 = calc_ema(closes.tolist(), 9)
    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)
    bb = calc_bollinger(closes.tolist(), 20, 2.0)
    donch = calc_donchian(highs.tolist(), lows.tolist(), 20)
    donch10 = calc_donchian(highs.tolist(), lows.tolist(), 10)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    # MACD Calculation
    ema12 = calc_ema(closes.tolist(), 12)
    ema26 = calc_ema(closes.tolist(), 26)
    macd_line = []
    for i in range(n):
        if ema12[i] is not None and ema26[i] is not None:
            macd_line.append(ema12[i] - ema26[i])
        else:
            macd_line.append(None)
    valid_macd = [m for m in macd_line if m is not None]
    macd_sig_valid = calc_ema(valid_macd, 9)
    prefix_len = len(macd_line) - len(valid_macd)
    macd_signal = [None] * prefix_len + macd_sig_valid

    INITIAL_CAPITAL = 10000.0
    RISK_DOLLARS_PER_TRADE = 100.0  # 1.0% risk per trade on $10,000 capital
    SPREAD_PTS = 0.35  # XM broker spread on Gold ($0.35/oz)

    # =========================================================================
    # Strategy 1: London Session Asian Range Liquidity Sweep (Judas Swing)
    # =========================================================================
    sig_asian_sweep = []
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
        if 7 <= hour <= 11 and asia_high > 0 and asia_low < 999999:
            rng = asia_high - asia_low
            if rng < 5.0 or rng > 40.0:
                continue
            c_h, c_l, c_c, c_o = highs[i], lows[i], closes[i], opens[i]
            c_rng = max(0.01, c_h - c_l)
            upper_wick = c_h - max(c_o, c_c)
            lower_wick = min(c_o, c_c) - c_l
            cur_atr = atr[i] if atr[i] else 10.0

            # Bearish Judas: Swept Asian High, closed inside with upper wick
            if c_h > asia_high and c_c < asia_high and (upper_wick / c_rng >= 0.35):
                sl = c_h + 1.00
                tp = c_c - (sl - c_c) * 1.8  # 1.8 R:R
                sig_asian_sweep.append({
                    "idx": i, "dir": "SELL", "price": c_c, "sl": sl, "tp": tp, "date": dt_str
                })
            # Bullish Judas: Swept Asian Low, closed inside with lower wick
            elif c_l < asia_low and c_c > asia_low and (lower_wick / c_rng >= 0.35):
                sl = c_l - 1.00
                tp = c_c + (c_c - sl) * 1.8  # 1.8 R:R
                sig_asian_sweep.append({
                    "idx": i, "dir": "BUY", "price": c_c, "sl": sl, "tp": tp, "date": dt_str
                })

    # =========================================================================
    # Strategy 2: 20/50 EMA Trend Pullback + Value Zone Rejection (Institutional Core)
    # =========================================================================
    sig_ema_pullback = []
    for i in range(200, n - 24):
        e20, e50, e200_val, r, a = ema20[i], ema50[i], ema200[i], rsi[i], atr[i]
        st_dir = st["direction"][i]
        if None in (e20, e50, e200_val, r, a, st_dir):
            continue
        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        rng = max(0.01, h - l)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        # Bullish Pullback in Uptrend
        if c > e200_val and e20 > e50 and (l <= max(e20, e50) and c >= min(e20, e50)):
            if (lower_wick / rng >= 0.35) and c >= o and (40 <= r <= 65):
                sl_dist = max(1.5 * a, 8.0)
                sl = c - sl_dist
                tp = c + (1.8 * sl_dist)
                sig_ema_pullback.append({
                    "idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp, "date": dates[i]
                })

        # Bearish Pullback in Downtrend
        elif c < e200_val and e20 < e50 and (h >= min(e20, e50) and c <= max(e20, e50)):
            if (upper_wick / rng >= 0.35) and c <= o and (35 <= r <= 60):
                sl_dist = max(1.5 * a, 8.0)
                sl = c + sl_dist
                tp = c - (1.8 * sl_dist)
                sig_ema_pullback.append({
                    "idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp, "date": dates[i]
                })

    # =========================================================================
    # Strategy 3: SuperTrend + MACD Momentum Continuation
    # =========================================================================
    sig_supertrend_macd = []
    for i in range(200, n - 24):
        st_dir = st["direction"][i]
        st_prev = st["direction"][i-1]
        m_curr, m_sig = macd_line[i], macd_signal[i]
        m_prev, m_sig_prev = macd_line[i-1], macd_signal[i-1]
        e200_val = ema200[i]
        a = atr[i]
        if None in (st_dir, st_prev, m_curr, m_sig, m_prev, m_sig_prev, e200_val, a):
            continue
        c = closes[i]

        # Bullish: Supertrend green, Price > 200 EMA, MACD cross above signal
        if st_dir == 1 and c > e200_val and (m_prev <= m_sig_prev and m_curr > m_sig):
            sl_dist = max(1.5 * a, 9.0)
            sl = c - sl_dist
            tp = c + (1.8 * sl_dist)
            sig_supertrend_macd.append({
                "idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp, "date": dates[i]
            })

        # Bearish: Supertrend red, Price < 200 EMA, MACD cross below signal
        elif st_dir == -1 and c < e200_val and (m_prev >= m_sig_prev and m_curr < m_sig):
            sl_dist = max(1.5 * a, 9.0)
            sl = c + sl_dist
            tp = c - (1.8 * sl_dist)
            sig_supertrend_macd.append({
                "idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp, "date": dates[i]
            })

    # =========================================================================
    # Strategy 4: Bollinger Band Mean Reversion (Exhaustion Fade)
    # =========================================================================
    sig_bb_reversion = []
    for i in range(200, n - 24):
        b_up, b_mid, b_low = bb["upper"][i], bb["middle"][i], bb["lower"][i]
        r, a = rsi[i], atr[i]
        if None in (b_up, b_mid, b_low, r, a):
            continue
        c, h, l, o = closes[i], highs[i], lows[i], opens[i]
        rng = max(0.01, h - l)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        # Oversold Reversal (Fade back to Middle SMA)
        if l <= b_low and r <= 32 and (lower_wick / rng >= 0.40) and c >= o:
            sl = l - 2.0
            tp = b_mid  # Target 20-SMA middle band
            if (tp - c) >= 1.2 * (c - sl):
                sig_bb_reversion.append({
                    "idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp, "date": dates[i]
                })

        # Overbought Reversal (Fade back to Middle SMA)
        elif h >= b_up and r >= 68 and (upper_wick / rng >= 0.40) and c <= o:
            sl = h + 2.0
            tp = b_mid
            if (c - tp) >= 1.2 * (sl - c):
                sig_bb_reversion.append({
                    "idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp, "date": dates[i]
                })

    # =========================================================================
    # Strategy 5: Donchian 20-Bar Channel Breakout (Turtle Trend)
    # =========================================================================
    sig_donchian_breakout = []
    for i in range(200, n - 24):
        d_up = donch["upper"][i-1]
        d_low = donch["lower"][i-1]
        e200_val, a = ema200[i], atr[i]
        if None in (d_up, d_low, e200_val, a):
            continue
        c, h, l = closes[i], highs[i], lows[i]

        # Breakout High with 200 EMA trend filter
        if h > d_up and c > e200_val:
            sl_dist = max(1.5 * a, 10.0)
            sl = c - sl_dist
            tp = c + (2.2 * sl_dist)  # Wider 2.2 R:R for trend breakouts
            sig_donchian_breakout.append({
                "idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp, "date": dates[i]
            })

        # Breakout Low with 200 EMA trend filter
        elif l < d_low and c < e200_val:
            sl_dist = max(1.5 * a, 10.0)
            sl = c + sl_dist
            tp = c - (2.2 * sl_dist)
            sig_donchian_breakout.append({
                "idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp, "date": dates[i]
            })

    # =========================================================================
    # Strategy 6: ICT Fair Value Gap (FVG) / Smart Money Imbalance
    # =========================================================================
    sig_ict_fvg = []
    for i in range(200, n - 24):
        # 3-bar pattern: bar i-2, i-1, i
        # Bullish FVG: Low of candle i > High of candle i-2 (Gap between i-2 high and i low)
        e200_val, a = ema200[i], atr[i]
        if None in (e200_val, a):
            continue
        h_prev2 = highs[i-2]
        l_curr = lows[i]
        c = closes[i]

        if l_curr > h_prev2 and (l_curr - h_prev2) >= 1.50 and c > e200_val:
            fvg_top = l_curr
            fvg_bottom = h_prev2
            fvg_mid = (fvg_top + fvg_bottom) / 2.0
            sl = fvg_bottom - 2.0
            tp = c + (c - sl) * 2.0
            sig_ict_fvg.append({
                "idx": i, "dir": "BUY", "price": c, "sl": sl, "tp": tp, "date": dates[i]
            })

        # Bearish FVG: High of candle i < Low of candle i-2
        l_prev2 = lows[i-2]
        h_curr = highs[i]
        if h_curr < l_prev2 and (l_prev2 - h_curr) >= 1.50 and c < e200_val:
            fvg_bottom = h_curr
            fvg_top = l_prev2
            fvg_mid = (fvg_top + fvg_bottom) / 2.0
            sl = fvg_top + 2.0
            tp = c - (sl - c) * 2.0
            sig_ict_fvg.append({
                "idx": i, "dir": "SELL", "price": c, "sl": sl, "tp": tp, "date": dates[i]
            })

    strategies = [
        ("1. London Asian Range Liquidity Sweep (Judas)", sig_asian_sweep),
        ("2. 20/50 EMA Value Zone Pullback + Rejection", sig_ema_pullback),
        ("3. SuperTrend + MACD Momentum Trend", sig_supertrend_macd),
        ("4. Bollinger Bands + RSI Exhaustion Fade", sig_bb_reversion),
        ("5. Donchian 20-Bar Channel Breakout", sig_donchian_breakout),
        ("6. ICT Fair Value Gap (FVG) Imbalance", sig_ict_fvg),
    ]

    all_results = []

    for name, sigs in strategies:
        # Simulate trades sequentially with max 1 open position at a time
        balance = INITIAL_CAPITAL
        peak_balance = INITIAL_CAPITAL
        max_drawdown_dollars = 0.0
        max_drawdown_pct = 0.0

        executed = 0
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        active_until = -1
        trade_pnls = []

        for s in sigs:
            if s["idx"] <= active_until:
                continue

            entry_idx = s["idx"]
            direction = s["dir"]
            entry_price = s["price"]
            sl_price = s["sl"]
            tp_price = s["tp"]

            # Effective entry with spread
            if direction == "BUY":
                eff_entry = entry_price + SPREAD_PTS
                risk_pts = eff_entry - sl_price
                if risk_pts <= 0.50:
                    continue
                # Position sizing for strict $100 risk per trade
                lots = round(RISK_DOLLARS_PER_TRADE / (risk_pts * 100), 2)
                lots = max(0.01, min(lots, 2.00))

                won = False
                exit_price = eff_entry
                bars_held = 0
                for k in range(entry_idx + 1, min(entry_idx + 48, n)):
                    bars_held = k - entry_idx
                    h_k, l_k = highs[k], lows[k]
                    if l_k <= sl_price:
                        won = False
                        exit_price = sl_price
                        break
                    if h_k >= tp_price:
                        won = True
                        exit_price = tp_price
                        break
                else:
                    exit_price = closes[min(entry_idx + 48, n - 1)]
                    won = exit_price > eff_entry

                pnl = (exit_price - eff_entry) * lots * 100

            else:  # SELL
                eff_entry = entry_price
                risk_pts = sl_price - eff_entry
                if risk_pts <= 0.50:
                    continue
                lots = round(RISK_DOLLARS_PER_TRADE / (risk_pts * 100), 2)
                lots = max(0.01, min(lots, 2.00))

                won = False
                exit_price = eff_entry
                bars_held = 0
                for k in range(entry_idx + 1, min(entry_idx + 48, n)):
                    bars_held = k - entry_idx
                    h_k, l_k = highs[k], lows[k]
                    # On SELL exit via buy, spread applies
                    if (h_k + SPREAD_PTS) >= sl_price:
                        won = False
                        exit_price = sl_price
                        break
                    if (l_k + SPREAD_PTS) <= tp_price:
                        won = True
                        exit_price = tp_price
                        break
                else:
                    exit_price = closes[min(entry_idx + 48, n - 1)] + SPREAD_PTS
                    won = eff_entry > exit_price

                pnl = (eff_entry - exit_price) * lots * 100

            executed += 1
            active_until = entry_idx + bars_held
            trade_pnls.append(pnl)

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

            balance += pnl
            if balance > peak_balance:
                peak_balance = balance
            dd = peak_balance - balance
            dd_pct = (dd / peak_balance) * 100.0 if peak_balance > 0 else 0.0
            if dd > max_drawdown_dollars:
                max_drawdown_dollars = dd
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

        net_pnl = gross_profit - gross_loss
        win_rate = (wins / executed * 100.0) if executed > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        avg_win = (gross_profit / wins) if wins > 0 else 0.0
        avg_loss = (gross_loss / losses) if losses > 0 else 0.0
        return_pct = (net_pnl / INITIAL_CAPITAL) * 100.0

        # Sharpe ratio approximation (annualized from hourly trade distribution)
        if len(trade_pnls) > 5 and np.std(trade_pnls) > 0:
            sharpe = (np.mean(trade_pnls) / np.std(trade_pnls)) * np.sqrt(executed)
        else:
            sharpe = 0.0

        all_results.append({
            "name": name,
            "total_signals": len(sigs),
            "trades": executed,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(return_pct, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown_dollars": round(max_drawdown_dollars, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 1),
            "final_balance": round(balance, 2),
            "sharpe": round(sharpe, 2),
        })

    # Sort results by Profit Factor / Net PnL descending
    all_results.sort(key=lambda x: x["net_pnl"], reverse=True)

    print("\n" + "=" * 100)
    print("🏆 6-MONTH GOLD STRATEGY TOURNAMENT RESULTS (March 27, 2026 - September 25, 2026)")
    print("=" * 100)
    header = f"{'Strategy Name':<44} | {'Trades':<6} | {'Win Rate':<8} | {'PF':<5} | {'Net P&L ($)':<12} | {'Return %':<8} | {'Max DD %':<8}"
    print(header)
    print("-" * 100)
    for r in all_results:
        pnl_str = f"+${r['net_pnl']:,.2f}" if r['net_pnl'] >= 0 else f"-${abs(r['net_pnl']):,.2f}"
        ret_str = f"+{r['return_pct']:.1f}%" if r['return_pct'] >= 0 else f"{r['return_pct']:.1f}%"
        print(f"{r['name']:<44} | {r['trades']:<6} | {r['win_rate']:>6.1f}% | {r['profit_factor']:>5.2f} | {pnl_str:>12} | {ret_str:>8} | {r['max_drawdown_pct']:>7.1f}%")
    print("=" * 100)

    # Save detailed JSON output
    out_path = os.path.expanduser("~/.gold_trading_data/tournament_6mo_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed metrics saved to: {out_path}")


if __name__ == "__main__":
    run_6mo_tournament()
