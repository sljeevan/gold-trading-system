"""
Quantitative Backtest Simulator for 30-Level Compounding Progression ($20 -> $52,384)
Comparing Baseline vs Latest Updates (Bar Lock + 15m Loss Cooldown).
Evaluates both Market Execution (Live Bot Default) and Limit Pullback Execution.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import numpy as np

from gold_trader.core.compounding_ladder import LADDER_MATRIX, get_ladder_level_info
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_supertrend, calc_rsi


def simulate_ladder_progression(
    candles_15m: List[Dict[str, Any]],
    use_pullback_limit: bool = False,
    enable_bar_lock: bool = True,
    loss_cooldown_bars: int = 1,
    spread: float = 0.35,
    pullback_discount: float = 0.80,
) -> Dict[str, Any]:
    n = len(candles_15m)
    highs = np.array([float(c["high"]) for c in candles_15m], dtype=float)
    lows = np.array([float(c["low"]) for c in candles_15m], dtype=float)
    closes = np.array([float(c["close"]) for c in candles_15m], dtype=float)
    opens = np.array([float(c["open"]) for c in candles_15m], dtype=float)
    dates = [c["date"] for c in candles_15m]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)

    level = 1
    balance = 20.0
    peak_level = 1
    peak_balance = 20.0
    completions = 0
    reloads = 0
    wins = 0
    losses = 0
    scratches = 0
    total_banked_profit = 0.0

    completion_history = []
    trade_log = []
    level_reach_count = {i: 0 for i in range(1, 31)}

    consecutive_losses = 0
    max_consecutive_losses = 0
    consecutive_wins = 0
    max_consecutive_wins = 0

    blocked_until_bar = -1
    last_trade_bar = -1

    for i in range(100, n - 36):
        # 1. Check if blocked by active trade or loss cooldown
        if i < blocked_until_bar:
            continue

        # 2. Bar Lock check
        if enable_bar_lock and i == last_trade_bar:
            continue

        dt = dates[i]
        hour = 12
        try:
            sep = "T" if "T" in dt else " "
            hour = int(dt.split(sep)[1].split(":")[0])
        except Exception:
            pass

        # Session filter: London & New York (07:00 to 19:00 UTC)
        if not (7 <= hour <= 19):
            continue

        e9, e21, e50 = ema9[i], ema21[i], ema50[i]
        st_d = st["direction"][i]
        r = rsi[i]
        if None in (e9, e21, e50, st_d, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
            dir_ = "BUY"
        elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        # Fill determination
        k_fill = i + 1
        if k_fill >= n:
            break

        if use_pullback_limit:
            limit_p = c - pullback_discount if dir_ == "BUY" else c + pullback_discount
            if dir_ == "BUY" and lows[k_fill] <= limit_p:
                fill_p = limit_p
            elif dir_ == "SELL" and highs[k_fill] >= limit_p:
                fill_p = limit_p
            else:
                continue
        else:
            # Market execution at open of next candle
            fill_p = opens[k_fill]

        last_trade_bar = i
        lvl_info = get_ladder_level_info(level)
        level_reach_count[level] += 1
        start_lvl = level

        eff_entry = fill_p + spread if dir_ == "BUY" else fill_p
        tp = eff_entry + 2.0 if dir_ == "BUY" else eff_entry - 2.0 - spread
        sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + spread
        be_trigger = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0

        won, lost, scratched = False, False, False
        is_be = False
        bars_held = 0
        exit_bar = k_fill

        for k in range(k_fill, min(k_fill + 32, n)):
            exit_bar = k
            bars_held = k - i
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
                    if h_k >= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= tp:
                        won = True
                        break
                else:
                    if h_k >= tp:
                        won = True
                        break
                    if h_k >= be_trigger:
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
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= tp:
                        won = True
                        break
                else:
                    if l_k <= tp:
                        won = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break

        # If trade timed out after 32 bars without hitting TP/SL, scratch at market
        if not (won or lost or scratched):
            scratched = True

        # Process trade outcome
        if won:
            wins += 1
            gain = lvl_info["profit_dollars"]
            balance += gain
            trade_outcome = "WIN"
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)

            # Next bar can trade immediately
            blocked_until_bar = exit_bar + 1

            if level == 30:
                completions += 1
                total_banked_profit += balance
                completion_history.append({
                    "date": dt,
                    "trade_no": wins + losses + scratches,
                    "final_balance": round(balance, 2),
                })
                # Reset to Level 1 with fresh $20 seed
                level = 1
                balance = 20.0
            else:
                level += 1
                peak_level = max(peak_level, level)

        elif scratched:
            scratches += 1
            gain = 0.0
            trade_outcome = "BREAKEVEN"
            # No cooldown for breakeven
            blocked_until_bar = exit_bar + 1

        elif lost:
            losses += 1
            gain = -lvl_info["risk_dollars"]
            balance = max(0.0, balance + gain)
            trade_outcome = "LOSS"
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            # Apply loss cooldown
            blocked_until_bar = exit_bar + 1 + loss_cooldown_bars

            if level > 1:
                level -= 1
            else:
                if balance < 5.0:
                    reloads += 1
                    balance = 20.0
                    level = 1

        peak_balance = max(peak_balance, balance)

        trade_log.append({
            "trade_no": wins + losses + scratches,
            "date": dt,
            "direction": dir_,
            "start_level": start_lvl,
            "outcome": trade_outcome,
            "pnl": gain,
            "new_level": level,
            "balance": round(balance, 2),
            "bars_held": bars_held,
        })

    total_trades = wins + losses + scratches
    decided_trades = wins + losses
    win_rate = (wins / decided_trades * 100) if decided_trades > 0 else 0.0

    return {
        "execution_mode": "LIMIT_PULLBACK" if use_pullback_limit else "MARKET_EXECUTION",
        "bar_lock_enabled": enable_bar_lock,
        "loss_cooldown_bars": loss_cooldown_bars,
        "total_trades": total_trades,
        "decided_trades": decided_trades,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_pct": round(win_rate, 2),
        "scratch_rate_pct": round(scratches / total_trades * 100, 2) if total_trades else 0.0,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "trades_per_day": round(total_trades / 42.0, 2),
        "peak_level": peak_level,
        "peak_balance": round(peak_balance, 2),
        "level_30_completions": completions,
        "reloads_count": reloads,
        "total_banked_profit": round(total_banked_profit + balance, 2),
        "ending_level": level,
        "ending_balance": round(balance, 2),
        "completion_milestones": completion_history,
        "level_distribution": level_reach_count,
        "recent_trades": trade_log[-10:],
    }


def simulate_1y_hourly_progression(
    candles_1h: List[Dict[str, Any]],
    enable_cooldown: bool = True,
    spread: float = 0.35,
    pullback_discount: float = 0.80,
) -> Dict[str, Any]:
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h], dtype=float)
    lows = np.array([float(c["low"]) for c in candles_1h], dtype=float)
    closes = np.array([float(c["close"]) for c in candles_1h], dtype=float)
    opens = np.array([float(c["open"]) for c in candles_1h], dtype=float)
    dates = [c["date"] for c in candles_1h]

    fast_ema = calc_ema(closes.tolist(), 9)
    slow_ema = calc_ema(closes.tolist(), 21)
    trend_ema = calc_ema(closes.tolist(), 100)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    rsi = calc_rsi(closes.tolist(), 14)

    level = 1
    balance = 20.0
    peak_level = 1
    peak_balance = 20.0
    completions = 0
    reloads = 0
    wins = 0
    losses = 0
    scratches = 0
    total_banked_profit = 0.0

    completion_history = []
    trade_log = []
    consecutive_losses = 0
    max_consecutive_losses = 0

    blocked_until_bar = -1

    for i in range(200, n - 24):
        if i < blocked_until_bar:
            continue

        dt = dates[i]
        hour = 12
        try:
            sep = "T" if "T" in dt else " "
            hour = int(dt.split(sep)[1].split(":")[0])
        except Exception:
            pass

        if not (7 <= hour <= 19):
            continue

        f, s, t = fast_ema[i], slow_ema[i], trend_ema[i]
        st_d = st["direction"][i]
        r = rsi[i]
        if None in (f, s, t, st_d, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        if c > t and f > s and (l <= f and c >= s) and c >= o and (45 <= r <= 68):
            dir_ = "BUY"
        elif c < t and f < s and (h >= f and c <= s) and c <= o and (32 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        k_fill = i + 1
        limit_p = c - pullback_discount if dir_ == "BUY" else c + pullback_discount
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
        be_trigger = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0

        won, lost, scratched = False, False, False
        is_be = False
        bars_held = 0
        exit_bar = k_fill

        for k in range(k_fill, min(k_fill + 20, n)):
            exit_bar = k
            bars_held = k - i
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
                    if h_k >= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= tp:
                        won = True
                        break
                else:
                    if h_k >= tp:
                        won = True
                        break
                    if h_k >= be_trigger:
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
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if l_k <= tp:
                        won = True
                        break
                else:
                    if l_k <= tp:
                        won = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    curr_sl = eff_entry if is_be else sl
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break

        if not (won or lost or scratched):
            scratched = True

        if won:
            wins += 1
            gain = lvl_info["profit_dollars"]
            balance += gain
            consecutive_losses = 0
            blocked_until_bar = exit_bar + 1

            if level == 30:
                completions += 1
                total_banked_profit += balance
                completion_history.append({
                    "date": dt,
                    "trade_no": wins + losses + scratches,
                    "final_balance": round(balance, 2),
                })
                level = 1
                balance = 20.0
            else:
                level += 1
                peak_level = max(peak_level, level)

        elif scratched:
            scratches += 1
            blocked_until_bar = exit_bar + 1

        elif lost:
            losses += 1
            gain = -lvl_info["risk_dollars"]
            balance = max(0.0, balance + gain)
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            # 1-hour loss cooldown: pause 1 candle bar after loss if enabled
            cooldown_bars = 1 if enable_cooldown else 0
            blocked_until_bar = exit_bar + 1 + cooldown_bars

            if level > 1:
                level -= 1
            else:
                if balance < 5.0:
                    reloads += 1
                    balance = 20.0
                    level = 1

        peak_balance = max(peak_balance, balance)

    total_trades = wins + losses + scratches
    decided_trades = wins + losses
    win_rate = (wins / decided_trades * 100) if decided_trades > 0 else 0.0

    return {
        "dataset": "1-Year Hourly (5,699 Candles)",
        "total_trades": total_trades,
        "decided_trades": decided_trades,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_pct": round(win_rate, 2),
        "scratch_rate_pct": round(scratches / total_trades * 100, 2) if total_trades else 0.0,
        "max_consecutive_losses": max_consecutive_losses,
        "trades_per_day": round(total_trades / 252.0, 2),
        "peak_level": peak_level,
        "peak_balance": round(peak_balance, 2),
        "level_30_completions": completions,
        "reloads_count": reloads,
        "total_banked_profit": round(total_banked_profit + balance, 2),
        "ending_level": level,
        "ending_balance": round(balance, 2),
        "completion_milestones": completion_history,
    }


def main():
    print("Fetching 15m Gold candles (60 days) and 1h Gold candles (1 year)...")
    candles_15m = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
    candles_1h = fetch_gold_candles("XAUUSD", period="1y", interval="1h")
    print(f"Retrieved {len(candles_15m)} 15m candles and {len(candles_1h)} 1h candles.")

    # 1. Market Execution - Baseline (No Cooldown, No Bar Lock)
    res_market_baseline = simulate_ladder_progression(
        candles_15m,
        use_pullback_limit=False,
        enable_bar_lock=False,
        loss_cooldown_bars=0,
    )

    # 2. Market Execution - UPDATED (Bar Lock + 15m Loss Cooldown)
    res_market_updated = simulate_ladder_progression(
        candles_15m,
        use_pullback_limit=False,
        enable_bar_lock=True,
        loss_cooldown_bars=1,
    )

    # 3. Pullback Limit - Baseline (No Cooldown)
    res_limit_baseline = simulate_ladder_progression(
        candles_15m,
        use_pullback_limit=True,
        enable_bar_lock=False,
        loss_cooldown_bars=0,
    )

    # 4. Pullback Limit - UPDATED (Bar Lock + 15m Loss Cooldown)
    res_limit_updated = simulate_ladder_progression(
        candles_15m,
        use_pullback_limit=True,
        enable_bar_lock=True,
        loss_cooldown_bars=1,
    )

    # 5. 1-Year Hourly Macro - Updated with Cooldown
    res_1y_updated = simulate_1y_hourly_progression(
        candles_1h,
        enable_cooldown=True,
    )

    results = {
        "dataset_info": {
            "timeframe": "15m",
            "period": "60 days",
            "total_candles": len(candles_15m),
            "start_date": candles_15m[0]["date"],
            "end_date": candles_15m[-1]["date"],
        },
        "market_execution": {
            "baseline_no_cooldown": res_market_baseline,
            "updated_with_cooldown": res_market_updated,
        },
        "limit_execution": {
            "baseline_no_cooldown": res_limit_baseline,
            "updated_with_cooldown": res_limit_updated,
        },
        "1year_hourly": res_1y_updated,
    }

    out_file = os.path.expanduser("~/.gold_trading_data/compounding_updated_backtest.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("🏆 30-LEVEL COMPOUNDING LADDER BACKTEST: BEFORE vs AFTER UPDATE")
    print("=" * 70)
    print("\n[A] LIVE AUTO-TRADER MODE: MARKET EXECUTION")
    print(f"{'Metric':<30} | {'Baseline (No Cooldown)':<22} | {'Updated (Bar Lock + Cooldown)':<25}")
    print("-" * 80)
    print(f"{'Total Trades':<30} | {res_market_baseline['total_trades']:<22} | {res_market_updated['total_trades']:<25}")
    print(f"{'Trades / Day':<30} | {res_market_baseline['trades_per_day']:<22} | {res_market_updated['trades_per_day']:<25}")
    print(f"{'Decided Win Rate':<30} | {res_market_baseline['win_rate_pct']}%{'':<16} | {res_market_updated['win_rate_pct']}%{'':<19}")
    print(f"{'Breakeven Rate':<30} | {res_market_baseline['scratch_rate_pct']}%{'':<16} | {res_market_updated['scratch_rate_pct']}%{'':<19}")
    print(f"{'Max Consecutive Losses':<30} | {res_market_baseline['max_consecutive_losses']:<22} | {res_market_updated['max_consecutive_losses']:<25}")
    print(f"{'Peak Level Reached':<30} | Level {res_market_baseline['peak_level']:<16} | Level {res_market_updated['peak_level']:<19}")
    print(f"{'Level 30 Completions':<30} | {res_market_baseline['level_30_completions']} times{'':<15} | {res_market_updated['level_30_completions']} times{'':<18}")
    print(f"{'L1 Reloads Needed':<30} | {res_market_baseline['reloads_count']:<22} | {res_market_updated['reloads_count']:<25}")
    print(f"{'Total Banked Wealth ($20 seed)':<30} | ${res_market_baseline['total_banked_profit']:,.2f}{'':<12} | ${res_market_updated['total_banked_profit']:,.2f}{'':<14}")

    print("\n[B] PULLBACK LIMIT EXECUTION (8-Pip Limit)")
    print(f"{'Metric':<30} | {'Baseline (No Cooldown)':<22} | {'Updated (Bar Lock + Cooldown)':<25}")
    print("-" * 80)
    print(f"{'Total Trades':<30} | {res_limit_baseline['total_trades']:<22} | {res_limit_updated['total_trades']:<25}")
    print(f"{'Trades / Day':<30} | {res_limit_baseline['trades_per_day']:<22} | {res_limit_updated['trades_per_day']:<25}")
    print(f"{'Decided Win Rate':<30} | {res_limit_baseline['win_rate_pct']}%{'':<16} | {res_limit_updated['win_rate_pct']}%{'':<19}")
    print(f"{'Breakeven Rate':<30} | {res_limit_baseline['scratch_rate_pct']}%{'':<16} | {res_limit_updated['scratch_rate_pct']}%{'':<19}")
    print(f"{'Max Consecutive Losses':<30} | {res_limit_baseline['max_consecutive_losses']:<22} | {res_limit_updated['max_consecutive_losses']:<25}")
    print(f"{'Peak Level Reached':<30} | Level {res_limit_baseline['peak_level']:<16} | Level {res_limit_updated['peak_level']:<19}")
    print(f"{'Level 30 Completions':<30} | {res_limit_baseline['level_30_completions']} times{'':<15} | {res_limit_updated['level_30_completions']} times{'':<18}")
    print(f"{'L1 Reloads Needed':<30} | {res_limit_baseline['reloads_count']:<22} | {res_limit_updated['reloads_count']:<25}")
    print(f"{'Total Banked Wealth':<30} | ${res_limit_baseline['total_banked_profit']:,.2f}{'':<12} | ${res_limit_updated['total_banked_profit']:,.2f}{'':<14}")

    if res_limit_updated["completion_milestones"]:
        print("\n🏆 Level 30 Milestones with Updated System (Limit Execution):")
        for m in res_limit_updated["completion_milestones"]:
            print(f"  • Date: {m['date']} | Trade #{m['trade_no']} | Payout: ${m['final_balance']:,.2f}")

    print("\n[C] 1-YEAR HOURLY MACRO ENGINE (5,699 Candles)")
    print(f"  Total Trades: {res_1y_updated['total_trades']} | Trades/Day: {res_1y_updated['trades_per_day']}")
    print(f"  Wins: {res_1y_updated['wins']} | Losses: {res_1y_updated['losses']} | Scratches: {res_1y_updated['scratches']}")
    print(f"  Win Rate: {res_1y_updated['win_rate_pct']}% | Breakeven Rate: {res_1y_updated['scratch_rate_pct']}%")
    print(f"  Level 30 Completions: {res_1y_updated['level_30_completions']} times ($52,384 each)")
    print(f"  Reloads at L1: {res_1y_updated['reloads_count']} | Max Consecutive Losses: {res_1y_updated['max_consecutive_losses']}")
    print(f"  Total Banked Wealth: ${res_1y_updated['total_banked_profit']:,.2f}")
    if res_1y_updated["completion_milestones"]:
        for m in res_1y_updated["completion_milestones"]:
            print(f"    ⭐ Level 30 Achieved on {m['date']} (Trade #{m['trade_no']}) with ${m['final_balance']:,.2f}")


if __name__ == "__main__":
    main()
