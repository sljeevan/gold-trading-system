"""
Quantitative 6-Month Backtest of Donchian Breakout on 15-Minute Timeframe.
Evaluates 12,000 actual XM MT5 15m Gold candles (March 26, 2026 - September 28, 2026).
Compares:
1. Standard 20-bar Donchian (5-hour window)
2. Equivalent 80-bar Donchian (20-hour window, matching 1h strategy)
Under multiple execution policies (Deferred BE, Partial Scale-Out, Pure Runner).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure src is in python path
sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.strategy_engine import calculate_lot_size


def run_15m_donchian_study():
    print("=" * 80)
    print("Fetching 12,000 15-minute Gold candles directly from XM MT5 Broker...")
    broker = SocketBroker()
    candles = broker.get_rates("GOLD", "15m", 12000)
    if not candles or len(candles) < 1000:
        print(f"Error: only received {len(candles) if candles else 0} candles.")
        return

    n = len(candles)
    start_dt = candles[0]["date"]
    end_dt = candles[-1]["date"]
    print(f"Loaded {n} 15m candles from {start_dt} to {end_dt} (Full 6 Months)")
    print("=" * 80)

    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # Precalculate indicators
    dc_20 = calc_donchian(highs.tolist(), lows.tolist(), 20)
    dc_80 = calc_donchian(highs.tolist(), lows.tolist(), 80)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    ema50 = calc_ema(closes.tolist(), 50)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi = calc_rsi(closes.tolist(), 14)

    test_variants = [
        ("A. 20-Bar Donchian (5h) + Deferred BE (1.0R)", dc_20, "DEFERRED_BE", 20),
        ("B. 20-Bar Donchian (5h) + Partial Scale-Out (50% @ 1.2R)", dc_20, "PARTIAL_SCALE", 20),
        ("C. 20-Bar Donchian (5h) + Pure Runner (2.0R)", dc_20, "NO_BE", 20),
        ("D. 80-Bar Donchian (20h) + Deferred BE (1.0R)", dc_80, "DEFERRED_BE", 80),
        ("E. 80-Bar Donchian (20h) + Partial Scale-Out (50% @ 1.2R)", dc_80, "PARTIAL_SCALE", 80),
        ("F. 80-Bar Donchian (20h) + Pure Runner (2.0R)", dc_80, "NO_BE", 80),
    ]

    initial_balance = 10000.0
    risk_pct = 1.0
    spread_usd = 0.35
    session_start_hour = 7
    session_end_hour = 19
    max_daily_trades = 4

    summary_table = []

    for label, dc, be_mode, period in test_variants:
        equity = initial_balance
        peak_equity = initial_balance
        max_dd_usd = 0.0
        max_dd_pct = 0.0

        closed_trades = []
        daily_trades = {}
        active_until = -1
        loss_cooldown_until = -1

        for i in range(250, n - 48):
            if i <= active_until:
                continue

            dt_str = dates[i]
            day_str = dt_str.split(" ")[0]
            try:
                hour = int(dt_str.split(" ")[1].split(":")[0])
            except Exception:
                hour = 12

            # Session Filter (07:00 - 19:00 UTC)
            if not (session_start_hour <= hour <= session_end_hour):
                continue

            # Daily cap
            if daily_trades.get(day_str, 0) >= max_daily_trades:
                continue

            # Loss cooldown: 4 bars (60 minutes) on 15m
            if i <= loss_cooldown_until:
                continue

            d_up = dc["upper"][i - 1]
            d_low = dc["lower"][i - 1]
            st_dir = st["direction"][i]
            e200_val = ema200[i]
            e50_val = ema50[i]
            a = atr[i]
            r = rsi[i]

            if None in (d_up, d_low, st_dir, a, r):
                continue

            # Macro Trend confirmation
            macro_bullish = closes[i] > e200_val if e200_val is not None else (closes[i] > e50_val if e50_val is not None else True)
            macro_bearish = closes[i] < e200_val if e200_val is not None else (closes[i] < e50_val if e50_val is not None else True)

            c, h, l, o = closes[i], highs[i], lows[i], opens[i]

            triggered_side = None
            if (h >= d_up or c >= d_up) and st_dir == 1 and macro_bullish and r <= 75.0:
                triggered_side = "BUY"
            elif (l <= d_low or c <= d_low) and st_dir == -1 and macro_bearish and r >= 25.0:
                triggered_side = "SELL"

            if not triggered_side:
                continue

            daily_trades[day_str] = daily_trades.get(day_str, 0) + 1

            # Stop loss distance: min $6.00 on 15m
            stop_dist = max(round(1.5 * a, 2), 6.00)
            lots = calculate_lot_size(
                account_balance=equity,
                risk_pct=risk_pct,
                entry_price=c,
                stop_loss_price=c - stop_dist if triggered_side == "BUY" else c + stop_dist,
            )
            lots = max(0.01, min(lots, 5.0))

            if triggered_side == "BUY":
                entry_price = c + spread_usd
                sl_price = entry_price - stop_dist
                tp1_price = entry_price + (1.2 * stop_dist)
                tp2_price = entry_price + (2.0 * stop_dist)
            else:
                entry_price = c
                sl_price = entry_price + stop_dist
                tp1_price = entry_price - (1.2 * stop_dist)
                tp2_price = entry_price - (2.0 * stop_dist)

            be_active = False
            half_closed = False
            realized_pnl = 0.0
            remaining_lots = lots
            trade_closed = False
            exit_reason = "TIMEOUT"
            exit_price = entry_price
            bars_held = 0

            # BE trigger distance
            if be_mode in ("DEFERRED_BE", "PARTIAL_SCALE"):
                be_trigger = max(round(stop_dist * 1.0, 2), 6.00)
            else:
                be_trigger = 999999.0

            # Max hold: 96 bars = 24 hours on 15m
            for k in range(i + 1, min(i + 96, n)):
                bars_held = k - i
                bar_h = highs[k]
                bar_l = lows[k]

                if triggered_side == "BUY":
                    if not be_active and (bar_h - entry_price) >= be_trigger:
                        be_active = True
                        sl_price = entry_price + spread_usd

                    if be_mode == "PARTIAL_SCALE" and not half_closed and bar_h >= tp1_price:
                        half_closed = True
                        part_lots = round(lots * 0.5, 2)
                        realized_pnl += (tp1_price - entry_price) * part_lots * 100.0
                        remaining_lots = max(0.01, round(lots - part_lots, 2))
                        be_active = True
                        sl_price = entry_price + spread_usd

                    if bar_l <= sl_price:
                        trade_closed = True
                        exit_price = sl_price
                        exit_reason = "BE_SCRATCH" if be_active and sl_price >= entry_price else "STOP_LOSS"
                        break

                    target_price = tp2_price
                    if bar_h >= target_price:
                        trade_closed = True
                        exit_price = target_price
                        exit_reason = "TAKE_PROFIT"
                        break
                else:  # SELL
                    eff_h = bar_h + spread_usd
                    eff_l = bar_l + spread_usd

                    if not be_active and (entry_price - bar_l) >= be_trigger:
                        be_active = True
                        sl_price = entry_price - spread_usd

                    if be_mode == "PARTIAL_SCALE" and not half_closed and eff_l <= tp1_price:
                        half_closed = True
                        part_lots = round(lots * 0.5, 2)
                        realized_pnl += (entry_price - tp1_price) * part_lots * 100.0
                        remaining_lots = max(0.01, round(lots - part_lots, 2))
                        be_active = True
                        sl_price = entry_price - spread_usd

                    if eff_h >= sl_price:
                        trade_closed = True
                        exit_price = sl_price
                        exit_reason = "BE_SCRATCH" if be_active and sl_price <= entry_price else "STOP_LOSS"
                        break

                    target_price = tp2_price
                    if eff_l <= target_price:
                        trade_closed = True
                        exit_price = target_price
                        exit_reason = "TAKE_PROFIT"
                        break

            # Calculate remaining PnL
            if not trade_closed:
                final_bar_idx = min(i + 96 - 1, n - 1)
                exit_price = closes[final_bar_idx] if triggered_side == "BUY" else closes[final_bar_idx] + spread_usd
                exit_reason = "TIMEOUT"

            diff = (exit_price - entry_price) if triggered_side == "BUY" else (entry_price - exit_price)
            realized_pnl += diff * remaining_lots * 100.0

            equity = max(100.0, round(equity + realized_pnl, 2))
            if equity > peak_equity:
                peak_equity = equity
            dd_usd = peak_equity - equity
            dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
                max_dd_pct = dd_pct

            active_until = i + bars_held
            if realized_pnl < -0.5:
                loss_cooldown_until = active_until + 4  # 4 bars cooldown

            closed_trades.append({
                "side": triggered_side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(realized_pnl, 2),
                "reason": exit_reason,
            })

        # Calculate metrics
        num_trades = len(closed_trades)
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] < 0]
        scratches = [t for t in closed_trades if abs(t["pnl"]) <= 0.5]
        win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0.0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        net_pnl = equity - initial_balance
        return_pct = (net_pnl / initial_balance) * 100.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        summary_table.append({
            "variant": label,
            "trades": num_trades,
            "wins": len(wins),
            "losses": len(losses),
            "scratches": len(scratches),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(return_pct, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(payoff, 2),
            "max_dd_pct": round(max_dd_pct, 1),
            "final_balance": round(equity, 2),
        })

    # Print summary
    print("\n" + "=" * 90)
    print("🏆 6-MONTH DONCHIAN BREAKOUT BACKTEST RESULTS ON 15-MINUTE TIMEFRAME")
    print(f"Data: 12,000 candles from {start_dt} to {end_dt} | $10,000 Initial Capital | 1% Risk/Trade")
    print("=" * 90)
    fmt = "{:<55} | {:<6} | {:<8} | {:<5} | {:<10} | {:<8} | {:<6}"
    print(fmt.format("Configuration", "Trades", "Win Rate", "P.F.", "Net P&L", "Payoff", "Max DD"))
    print("-" * 90)
    for row in summary_table:
        sign = "+" if row["net_pnl"] >= 0 else ""
        print(fmt.format(
            row["variant"][:55],
            f"{row['trades']}",
            f"{row['win_rate']}%",
            f"{row['profit_factor']:.2f}",
            f"{sign}${row['net_pnl']:.2f}",
            f"{row['payoff_ratio']:.2f}:1",
            f"{row['max_dd_pct']:.1f}%",
        ))
    print("=" * 90)

    # Save to JSON
    out_file = os.path.expanduser("~/.gold_trading_data/donchian_15m_backtest_results.json")
    with open(out_file, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"Saved full results to {out_file}")


if __name__ == "__main__":
    run_15m_donchian_study()
