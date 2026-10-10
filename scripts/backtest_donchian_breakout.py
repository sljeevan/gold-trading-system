"""
Comprehensive Backtest of the Tournament-Winning Donchian 20-Bar Breakout + SuperTrend Strategy.
Evaluates 6 months of historical Gold (XAUUSD) 1-hour candles under 4 execution configurations:
1. Pure Trend Runner (No Trailing / Fixed TP 2.0R & SL 1.5 ATR)
2. Defective Early BE (Snapping to Entry at $3.50 / 0.6R - Diagnosed Phase 2 Defect)
3. Repaired Deferred BE (Moving SL to Entry only after >= 1.0R / min $8.00 profit)
4. Hybrid Partial Scale-Out (50% Volume closed at 1.2R TP1 + Deferred BE on 50% Runner to 2.0R TP2)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.strategy_engine import calculate_lot_size


def run_donchian_backtest(
    period: str = "6mo",
    interval: str = "1h",
    initial_balance: float = 10000.0,
    risk_pct: float = 1.0,
    spread_usd: float = 0.35,
    session_filter: bool = True,
    session_start_hour: int = 7,
    session_end_hour: int = 19,
    max_daily_trades: int = 4,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Loading {period} of Gold ({interval}) market data for backtest...")
    candles = fetch_gold_candles("XAUUSD", period=period, interval=interval)
    if not candles or len(candles) < 100:
        raise RuntimeError(f"Insufficient candle history: {len(candles) if candles else 0} candles.")

    n = len(candles)
    start_date = candles[0]["date"]
    end_date = candles[-1]["date"]
    print(f"Loaded {n} candles from {start_date} to {end_date}.")

    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # Precalculate indicators
    dc = calc_donchian(highs.tolist(), lows.tolist(), 20)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    ema50 = calc_ema(closes.tolist(), 50)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi = calc_rsi(closes.tolist(), 14)

    configs = [
        ("1. Pure Runner (Fixed 2.0R Target, No Trailing BE)", "NO_BE"),
        ("2. Defective Early BE (Snapping to BE at $3.50 / 0.6R)", "EARLY_BE"),
        ("3. Repaired Deferred BE (Snapping to BE at >= 1.0R / $8.00)", "DEFERRED_BE"),
        ("4. Hybrid Scale-Out (50% at 1.2R + Deferred BE on 2.0R Runner)", "PARTIAL_SCALE"),
    ]

    all_results = {}

    for cfg_name, cfg_mode in configs:
        equity = initial_balance
        peak_equity = initial_balance
        max_drawdown_usd = 0.0
        max_drawdown_pct = 0.0

        closed_trades: List[Dict[str, Any]] = []
        daily_trade_count: Dict[str, int] = {}
        active_until = -1
        loss_cooldown_until = -1

        for i in range(200, n - 24):
            if i <= active_until:
                continue

            dt_str = dates[i]
            day_str = dt_str.split("T" if "T" in dt_str else " ")[0]
            try:
                hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])
            except Exception:
                hour = 12

            # Session Filter (07:00 - 19:00 UTC)
            if session_filter and not (session_start_hour <= hour <= session_end_hour):
                continue

            # Daily trades cap
            if daily_trade_count.get(day_str, 0) >= max_daily_trades:
                continue

            # Loss cooldown (60m / 1 bar)
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

            # Macro Trend confirmation (EMA 200 or 50)
            macro_bullish = closes[i] > e200_val if e200_val is not None else (closes[i] > e50_val if e50_val is not None else True)
            macro_bearish = closes[i] < e200_val if e200_val is not None else (closes[i] < e50_val if e50_val is not None else True)

            c, h, l, o = closes[i], highs[i], lows[i], opens[i]

            triggered_side: Optional[str] = None
            if (h >= d_up or c >= d_up) and st_dir == 1 and macro_bullish and r <= 75.0:
                triggered_side = "BUY"
            elif (l <= d_low or c <= d_low) and st_dir == -1 and macro_bearish and r >= 25.0:
                triggered_side = "SELL"

            if not triggered_side:
                continue

            daily_trade_count[day_str] = daily_trade_count.get(day_str, 0) + 1

            stop_dist = max(round(1.5 * a, 2), 8.00)
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

            # Execution simulation bar by bar
            be_active = False
            half_closed = False
            realized_pnl = 0.0
            remaining_lots = lots
            trade_closed = False
            exit_reason = "TIMEOUT"
            exit_price = entry_price
            bars_held = 0

            # BE thresholds
            if cfg_mode == "EARLY_BE":
                be_trigger = max(round(stop_dist * 0.60, 2), 3.50)
            elif cfg_mode in ("DEFERRED_BE", "PARTIAL_SCALE"):
                be_trigger = max(round(stop_dist * 1.0, 2), 8.00)
            else:
                be_trigger = 999999.0

            for k in range(i + 1, min(i + 72, n)):  # max 72h hold
                bars_held = k - i
                bar_h = highs[k]
                bar_l = lows[k]
                bar_c = closes[k]

                if triggered_side == "BUY":
                    # Check if break-even triggered during the bar
                    if not be_active and (bar_h - entry_price) >= be_trigger:
                        be_active = True
                        sl_price = entry_price + 0.35  # Scratch covers XM spread

                    # Partial scale-out at TP1 (1.2R)
                    if cfg_mode == "PARTIAL_SCALE" and not half_closed and bar_h >= tp1_price:
                        half_closed = True
                        part_lots = round(lots * 0.5, 2)
                        realized_pnl += (tp1_price - entry_price) * part_lots * 100.0
                        remaining_lots = max(0.01, round(lots - part_lots, 2))
                        # Automatically lock remaining runner to BE
                        be_active = True
                        sl_price = entry_price + 0.35

                    # Check SL
                    if bar_l <= sl_price:
                        trade_closed = True
                        exit_price = sl_price
                        exit_reason = "BREAKEVEN_SCRATCH" if be_active and sl_price >= entry_price else "STOP_LOSS"
                        break

                    # Check TP2 (or TP1 if no partial)
                    target_to_hit = tp2_price if cfg_mode != "NO_BE" else tp2_price
                    if bar_h >= target_to_hit:
                        trade_closed = True
                        exit_price = target_to_hit
                        exit_reason = "TAKE_PROFIT_2"
                        break

                else:  # SELL
                    eff_h = bar_h + spread_usd
                    eff_l = bar_l + spread_usd

                    if not be_active and (entry_price - bar_l) >= be_trigger:
                        be_active = True
                        sl_price = entry_price - 0.35

                    if cfg_mode == "PARTIAL_SCALE" and not half_closed and eff_l <= tp1_price:
                        half_closed = True
                        part_lots = round(lots * 0.5, 2)
                        realized_pnl += (entry_price - tp1_price) * part_lots * 100.0
                        remaining_lots = max(0.01, round(lots - part_lots, 2))
                        be_active = True
                        sl_price = entry_price - 0.35

                    if eff_h >= sl_price:
                        trade_closed = True
                        exit_price = sl_price
                        exit_reason = "BREAKEVEN_SCRATCH" if be_active and sl_price <= entry_price else "STOP_LOSS"
                        break

                    if eff_l <= tp2_price:
                        trade_closed = True
                        exit_price = tp2_price
                        exit_reason = "TAKE_PROFIT_2"
                        break

            if not trade_closed:
                exit_price = closes[min(i + 72, n - 1)]
                exit_reason = "TIME_EXPIRY"

            # Compute trade PnL
            if triggered_side == "BUY":
                runner_pnl = (exit_price - entry_price) * remaining_lots * 100.0
            else:
                runner_pnl = (entry_price - exit_price) * remaining_lots * 100.0

            total_trade_pnl = round(realized_pnl + runner_pnl, 2)
            equity += total_trade_pnl

            if equity > peak_equity:
                peak_equity = equity
            dd_usd = peak_equity - equity
            dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd_usd > max_drawdown_usd:
                max_drawdown_usd = dd_usd
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

            active_until = i + bars_held
            if total_trade_pnl < 0:
                loss_cooldown_until = active_until + 1  # 60m cooldown

            closed_trades.append({
                "entry_date": dt_str,
                "side": triggered_side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": total_trade_pnl,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "half_closed": half_closed,
            })

        # Calculate summary metrics
        t_count = len(closed_trades)
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] < 0]
        scratches = [t for t in closed_trades if t["pnl"] == 0]

        win_count = len(wins)
        loss_count = len(losses)
        scratch_count = len(scratches)

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        net_pnl = gross_profit - gross_loss
        net_return_pct = (net_pnl / initial_balance) * 100.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
        win_rate = (win_count / t_count * 100.0) if t_count > 0 else 0.0

        # Effective win rate includes scratch/profitable partials
        effective_win_rate = ((win_count + scratch_count) / t_count * 100.0) if t_count > 0 else 0.0
        avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        all_results[cfg_mode] = {
            "name": cfg_name,
            "trades": t_count,
            "wins": win_count,
            "losses": loss_count,
            "scratches": scratch_count,
            "win_rate": round(win_rate, 1),
            "effective_win_rate": round(effective_win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(net_return_pct, 1),
            "max_dd_usd": round(max_drawdown_usd, 2),
            "max_dd_pct": round(max_drawdown_pct, 1),
            "final_equity": round(equity, 2),
        }

    return all_results


def print_comparison_table(results: Dict[str, Any]):
    print("\n" + "=" * 105)
    print("🏆 6-MONTH XAUUSD BACKTEST COMPARISON: DONCHIAN 20-BAR BREAKOUT & BE MECHANICS")
    print("=" * 105)
    header = f"{'Configuration':<52} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Payoff':<6} | {'Net P&L ($)':<12} | {'Max DD %':<7}"
    print(header)
    print("-" * 105)

    for mode, data in results.items():
        name = data["name"]
        trades = data["trades"]
        wr = f"{data['win_rate']:.1f}%"
        pf = f"{data['profit_factor']:.2f}"
        payoff = f"{data['payoff_ratio']:.2f}:1"
        pnl = f"+${data['net_pnl']:,.2f}" if data['net_pnl'] >= 0 else f"-${abs(data['net_pnl']):,.2f}"
        dd = f"{data['max_dd_pct']:.1f}%"
        print(f"{name:<52} | {trades:<6} | {wr:<6} | {pf:<5} | {payoff:<6} | {pnl:>12} | {dd:>7}")

    print("=" * 105)


if __name__ == "__main__":
    results = run_donchian_backtest()
    print_comparison_table(results)
    
    # Save results to data folder
    out_path = os.path.expanduser("~/.gold_trading_data/donchian_backtest_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw backtest data to: {out_path}")
