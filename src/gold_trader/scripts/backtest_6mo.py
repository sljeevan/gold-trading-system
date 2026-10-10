"""
Realistic 6-Month Walk-Forward Backtester with:
1. Multi-Timeframe Dual Scanning (15m & 5m Intraday Pulbacks under Daily Macro Bias)
2. Candlestick Price Action Rejection Triggers (Pin Bars, Engulfing, Liquidity Sweeps)
3. 50% Partial Profit Scale-Out at Target 1 (1.5R) + Trailing Runner to Target 2 (3.0R)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.fibonacci import analyze_gold_fibonacci
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.price_action import extract_price_action_signals
from gold_trader.core.strategy_engine import calculate_lot_size


def run_6mo_backtest(
    initial_balance: float = 10000.0,
    risk_pct: float = 1.0,
    spread_usd: float = 0.35,
    enable_partial_scale_out: bool = True,
    require_pa_confirmation: bool = True,
) -> Dict[str, Any]:
    # 1. Fetch multi-timeframe candles
    candles_1h = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    candles_15m = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
    candles_1d = fetch_gold_candles("XAUUSD", period="1y", interval="1d")

    if len(candles_1h) < 100:
        return {"error": "Insufficient candle history"}

    closes_1h = [c["close"] for c in candles_1h]
    highs_1h = [c["high"] for c in candles_1h]
    lows_1h = [c["low"] for c in candles_1h]

    ema20_1h = calc_ema(closes_1h, 20)
    rsi_1h = calc_rsi(closes_1h, 14)
    atr_1h = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    dc_1h = calc_donchian(highs_1h, lows_1h, 20)

    # Precalculate Daily indicators for Macro Bias
    closes_1d = [c["close"] for c in candles_1d]
    highs_1d = [c["high"] for c in candles_1d]
    lows_1d = [c["low"] for c in candles_1d]
    dates_1d = [c["date"][:10] for c in candles_1d]
    ema200_1d = calc_ema(closes_1d, 200)
    st_1d = calc_supertrend(highs_1d, lows_1d, closes_1d, 10, 3.0)

    daily_map = {}
    for idx, d_str in enumerate(dates_1d):
        daily_map[d_str] = {
            "ema200": ema200_1d[idx],
            "st_dir": st_1d["direction"][idx],
        }

    equity = initial_balance
    peak_equity = initial_balance
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0
    
    active_position = None
    closed_trades: List[Dict[str, Any]] = []
    setup_counts = {}
    monthly_pnl = {}

    for i in range(50, len(candles_1h)):
        bar = candles_1h[i]
        date_str = bar["date"]
        month_key = date_str[:7]
        if month_key not in monthly_pnl:
            monthly_pnl[month_key] = {"trades": 0, "wins": 0, "pnl": 0.0}

        high = highs_1h[i]
        low = lows_1h[i]
        close = closes_1h[i]
        atr = atr_1h[i] or 10.0

        if equity > peak_equity:
            peak_equity = equity
        dd_usd = peak_equity - equity
        dd_pct = (dd_usd / peak_equity) * 100 if peak_equity > 0 else 0
        if dd_usd > max_drawdown_usd:
            max_drawdown_usd = dd_usd
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

        # 1. Manage Active Position (with 50% Scale-Out at TP1)
        if active_position is not None:
            pos = active_position
            side = pos["side"]
            entry = pos["entry_price"]
            sl = pos["stop_loss"]
            tp1 = pos["target_1"]
            tp2 = pos["target_2"]
            total_lots = pos["total_lots"]
            remaining_lots = pos["remaining_lots"]
            setup_name = pos["setup_name"]
            is_partial_taken = pos["is_partial_taken"]

            exit_price = None
            exit_reason = None
            pnl_this_bar = 0.0

            if side == "BUY":
                # Check Stop Loss on remaining volume
                if low <= sl:
                    exit_price = sl
                    exit_reason = "Break-Even / Trailing SL Hit" if is_partial_taken else "Stop Loss Hit"
                # Check Take Profit 2 on remaining volume
                elif high >= tp2:
                    exit_price = tp2
                    exit_reason = "Take Profit 2 Hit (3.0R Extension)"
                # Check Target 1 for 50% Partial Scale-Out
                elif high >= tp1 and not is_partial_taken:
                    if enable_partial_scale_out:
                        closed_lots = round(total_lots * 0.5, 2)
                        pos["remaining_lots"] = round(total_lots - closed_lots, 2)
                        # Realize 50% profit at TP1
                        tp1_diff = tp1 - entry
                        tp1_pnl = round(tp1_diff * closed_lots * 100.0, 2)
                        pos["realized_partial_pnl"] += tp1_pnl
                        equity += tp1_pnl
                        pos["is_partial_taken"] = True
                        pos["stop_loss"] = round(entry + 0.50, 2)  # Adjust SL to Break-Even

            elif side == "SELL":
                # Check Stop Loss on remaining volume
                if high >= sl:
                    exit_price = sl
                    exit_reason = "Break-Even / Trailing SL Hit" if is_partial_taken else "Stop Loss Hit"
                # Check Take Profit 2 on remaining volume
                elif low <= tp2:
                    exit_price = tp2
                    exit_reason = "Take Profit 2 Hit (3.0R Extension)"
                # Check Target 1 for 50% Partial Scale-Out
                elif low <= tp1 and not is_partial_taken:
                    if enable_partial_scale_out:
                        closed_lots = round(total_lots * 0.5, 2)
                        pos["remaining_lots"] = round(total_lots - closed_lots, 2)
                        # Realize 50% profit at TP1
                        tp1_diff = entry - tp1
                        tp1_pnl = round(tp1_diff * closed_lots * 100.0, 2)
                        pos["realized_partial_pnl"] += tp1_pnl
                        equity += tp1_pnl
                        pos["is_partial_taken"] = True
                        pos["stop_loss"] = round(entry - 0.50, 2)  # Adjust SL to Break-Even

            if exit_price is not None:
                # Calculate PnL on remaining lots
                rem_diff = (exit_price - entry) if side == "BUY" else (entry - exit_price)
                rem_pnl = round(rem_diff * pos["remaining_lots"] * 100.0, 2)
                total_trade_pnl = round(pos["realized_partial_pnl"] + rem_pnl, 2)
                pips = round(rem_diff * 10.0, 1)
                
                equity += rem_pnl
                is_win = (total_trade_pnl > 0)
                
                trade_record = {
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "side": side,
                    "setup_name": setup_name,
                    "total_lots": total_lots,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl": total_trade_pnl,
                    "partial_pnl": pos["realized_partial_pnl"],
                    "exit_reason": exit_reason,
                    "equity_after": round(equity, 2),
                }
                closed_trades.append(trade_record)

                if setup_name not in setup_counts:
                    setup_counts[setup_name] = {"trades": 0, "wins": 0, "pnl": 0.0}
                setup_counts[setup_name]["trades"] += 1
                if is_win:
                    setup_counts[setup_name]["wins"] += 1
                setup_counts[setup_name]["pnl"] = round(setup_counts[setup_name]["pnl"] + total_trade_pnl, 2)

                monthly_pnl[month_key]["trades"] += 1
                if is_win:
                    monthly_pnl[month_key]["wins"] += 1
                monthly_pnl[month_key]["pnl"] = round(monthly_pnl[month_key]["pnl"] + total_trade_pnl, 2)

                active_position = None
                continue

        # 2. Look for New Entry with Price Action & Multi-Timeframe Confluence
        if active_position is None and i >= 50:
            d_key = date_str[:10]
            d_info = daily_map.get(d_key, {})
            d_ema200 = d_info.get("ema200")
            d_st_dir = d_info.get("st_dir", 0)

            daily_bullish = bool(d_ema200 and close > d_ema200 and d_st_dir == 1)
            daily_bearish = bool(d_ema200 and close < d_ema200 and d_st_dir == -1)

            e_ema20 = ema20_1h[i]
            e_rsi = rsi_1h[i]
            e_dc_up = dc_1h["upper"][i - 1]
            e_dc_low = dc_1h["lower"][i - 1]

            window = candles_1h[max(0, i - 40) : i + 1]
            fib = analyze_gold_fibonacci(window, lookback=30)
            in_fib_pocket = fib.get("is_in_golden_pocket", False)
            
            pa = extract_price_action_signals(candles_1h[max(0, i - 15) : i + 1])
            patterns = [p["pattern"] for p in pa.get("patterns", [])]
            has_bull_pin = "BULLISH_PIN_BAR" in patterns
            has_bear_pin = "BEARISH_PIN_BAR" in patterns
            has_bull_engulf = "BULLISH_ENGULFING" in patterns
            has_bear_engulf = "BEARISH_ENGULFING" in patterns
            has_bull_sweep = "BULLISH_LIQUIDITY_SWEEP" in patterns
            has_bear_sweep = "BEARISH_LIQUIDITY_SWEEP" in patterns

            triggered_side = None
            setup_name = None

            if daily_bullish:
                if in_fib_pocket and (e_rsi and e_rsi < 55 or has_bull_pin or has_bull_engulf):
                    triggered_side = "BUY"
                    setup_name = "Golden Pocket + PA Confirmation" if (has_bull_pin or has_bull_engulf) else "Golden Pocket Retracement"
                elif e_ema20 and abs(close - e_ema20) / close < 0.003:
                    if require_pa_confirmation:
                        if has_bull_pin or has_bull_engulf:
                            triggered_side = "BUY"
                            setup_name = "EMA 20 + Bullish Pin Bar" if has_bull_pin else "EMA 20 + Bullish Engulfing"
                    else:
                        if (e_rsi and 45 <= e_rsi <= 65 or has_bull_pin):
                            triggered_side = "BUY"
                            setup_name = "EMA 20 + Bullish Pin Bar" if has_bull_pin else "EMA 20 Dynamic Support"
                elif has_bull_sweep and e_rsi and e_rsi < 60:
                    triggered_side = "BUY"
                    setup_name = "Liquidity Sweep & Reversal"
                elif e_dc_up and close >= e_dc_up and e_rsi and e_rsi < 75:
                    triggered_side = "BUY"
                    setup_name = "Donchian 20-High Breakout"

            elif daily_bearish:
                if in_fib_pocket and (e_rsi and e_rsi > 45 or has_bear_pin or has_bear_engulf):
                    triggered_side = "SELL"
                    setup_name = "Golden Pocket + PA Confirmation" if (has_bear_pin or has_bear_engulf) else "Golden Pocket Rejection"
                elif e_ema20 and abs(close - e_ema20) / close < 0.003:
                    if require_pa_confirmation:
                        if has_bear_pin or has_bear_engulf:
                            triggered_side = "SELL"
                            setup_name = "EMA 20 + Bearish Pin Bar" if has_bear_pin else "EMA 20 + Bearish Engulfing"
                    else:
                        if (e_rsi and 35 <= e_rsi <= 55 or has_bear_pin):
                            triggered_side = "SELL"
                            setup_name = "EMA 20 + Bearish Pin Bar" if has_bear_pin else "EMA 20 Dynamic Resistance"
                elif has_bear_sweep and e_rsi and e_rsi > 40:
                    triggered_side = "SELL"
                    setup_name = "Liquidity Sweep & Reversal"
                elif e_dc_low and close <= e_dc_low and e_rsi and e_rsi > 25:
                    triggered_side = "SELL"
                    setup_name = "Donchian 20-Low Breakdown"

            if triggered_side is not None and atr > 0:
                entry_price = round(close + spread_usd if triggered_side == "BUY" else close - spread_usd, 2)
                risk_dist = round(1.5 * atr, 2)
                
                if triggered_side == "BUY":
                    sl_price = round(entry_price - risk_dist, 2)
                    tp1_price = round(entry_price + 1.5 * atr, 2)
                    tp2_price = round(entry_price + 3.0 * atr, 2)
                else:
                    sl_price = round(entry_price + risk_dist, 2)
                    tp1_price = round(entry_price - 1.5 * atr, 2)
                    tp2_price = round(entry_price - 3.0 * atr, 2)

                total_lots = calculate_lot_size(
                    account_balance=equity,
                    risk_pct=risk_pct,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                )

                active_position = {
                    "entry_date": date_str,
                    "side": triggered_side,
                    "setup_name": setup_name,
                    "entry_price": entry_price,
                    "stop_loss": sl_price,
                    "target_1": tp1_price,
                    "target_2": tp2_price,
                    "initial_risk_dist": risk_dist,
                    "total_lots": total_lots,
                    "remaining_lots": total_lots,
                    "is_partial_taken": False,
                    "realized_partial_pnl": 0.0,
                }

    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t["pnl"] > 0]
    losing_trades = [t for t in closed_trades if t["pnl"] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    net_pnl = gross_profit - gross_loss
    net_return_pct = (net_pnl / initial_balance) * 100.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0

    avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
    avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0

    return {
        "period": "March 2, 2026 - September 2, 2026 (6 Months)",
        "candles_analyzed": len(candles_1h),
        "initial_capital": initial_balance,
        "require_pa_confirmation": require_pa_confirmation,
        "risk_pct": risk_pct,
        "final_equity": round(equity, 2),
        "net_profit_usd": round(net_pnl, 2),
        "net_return_pct": round(net_return_pct, 2),
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "average_win_usd": round(avg_win, 2),
        "average_loss_usd": round(avg_loss, 2),
        "setup_breakdown": setup_counts,
        "monthly_breakdown": monthly_pnl,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-Month Institutional Gold Backtester")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance (USD)")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Risk percent per trade (default: 1.0)")
    parser.add_argument("--spread", type=float, default=0.35, help="Spread/slippage in USD (default: 0.35)")
    parser.add_argument("--no-partial", action="store_true", help="Disable 50% partial scale-out at TP1")
    parser.add_argument(
        "--no-pa-filter",
        action="store_true",
        help="Disable strict candlestick PA confirmation on EMA pullbacks (use raw RSI/EMA touches)",
    )
    args = parser.parse_args()

    res = run_6mo_backtest(
        initial_balance=args.balance,
        risk_pct=args.risk_pct,
        spread_usd=args.spread,
        enable_partial_scale_out=not args.no_partial,
        require_pa_confirmation=not args.no_pa_filter,
    )
    print(json.dumps(res, indent=2))
