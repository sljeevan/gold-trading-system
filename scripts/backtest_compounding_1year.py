"""
High-Performance Quantitative 1-Year Backtest Simulator for the 30-Level Gold Compounding Challenge ($20 -> $52,384).
Precomputes vectorized features, ML inference, and uses standard institutional OHLC intrabar pathing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from gold_trader.core.compounding_ladder import (
    LADDER_MATRIX,
    find_closest_level_for_balance,
    get_ladder_level_info,
)
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.ml_filter import GoldMLPredictor, extract_candle_features


def generate_candidate_signals(
    candles: List[Dict[str, Any]],
    session_filter: bool = True,
) -> List[Dict[str, Any]]:
    n = len(candles)
    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    opens = [float(c["open"]) for c in candles]

    fast_ema = calc_ema(closes, 5)
    slow_ema = calc_ema(closes, 20)
    trend_ema = calc_ema(closes, 200)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    rsi = calc_rsi(closes, 14)

    signals = []

    for i in range(200, n - 25):
        c_close = closes[i]
        c_high = highs[i]
        c_low = lows[i]
        c_open = opens[i]

        f, s = fast_ema[i], slow_ema[i]
        fp, sp = fast_ema[i - 1], slow_ema[i - 1]
        t = trend_ema[i]
        st_dir = st["direction"][i]
        r = rsi[i]

        if None in (f, s, fp, sp, t, st_dir, r):
            continue

        if session_filter:
            dt_str = candles[i].get("date", "")
            hour = 12
            try:
                sep = "T" if "T" in dt_str else " "
                hour = int(dt_str.split(sep)[1].split(":")[0])
            except Exception:
                pass
            if not (7 <= hour <= 19):
                continue

        rng = max(0.01, c_high - c_low)
        lower_wick = min(c_open, c_close) - c_low
        upper_wick = c_high - max(c_open, c_close)

        is_bull_cross = (fp <= sp and f > s)
        is_bull_trend = (c_close > t and st_dir == 1)
        is_bull_rejection = (lower_wick / rng >= 0.40 and c_close > s)

        if is_bull_trend and (is_bull_cross or is_bull_rejection) and 42 <= r <= 68:
            signals.append({
                "bar_idx": i,
                "date": candles[i]["date"],
                "direction": "BUY",
                "entry_price": c_close,
            })

        is_bear_cross = (fp >= sp and f < s)
        is_bear_trend = (c_close < t and st_dir == -1)
        is_bear_rejection = (upper_wick / rng >= 0.40 and c_close < s)

        if is_bear_trend and (is_bear_cross or is_bear_rejection) and 32 <= r <= 58:
            signals.append({
                "bar_idx": i,
                "date": candles[i]["date"],
                "direction": "SELL",
                "entry_price": c_close,
            })

    return signals


def simulate_bracket_fast(
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
    max_bars: int = 24,
) -> Tuple[bool, float, int]:
    n = len(closes)
    end_bar = min(entry_idx + max_bars + 1, n)

    if direction == "BUY":
        eff_entry = entry_price + spread_pts
        tp_target = eff_entry + tp_points
        sl_target = eff_entry - sl_points

        for k in range(entry_idx + 1, end_bar):
            c_open = opens[k]
            c_high = highs[k]
            c_low = lows[k]
            c_close = closes[k]

            hit_tp = c_high >= tp_target
            hit_sl = c_low <= sl_target

            if hit_tp and not hit_sl:
                return True, tp_target, k - entry_idx
            if hit_sl and not hit_tp:
                return False, sl_target, k - entry_idx
            if hit_tp and hit_sl:
                if c_close >= c_open:
                    return False, sl_target, k - entry_idx
                else:
                    return True, tp_target, k - entry_idx

        last_close = closes[min(entry_idx + max_bars, n - 1)]
        return (last_close > eff_entry), last_close, max_bars

    else:
        eff_entry = entry_price
        tp_target = eff_entry - tp_points - spread_pts
        sl_target = eff_entry + sl_points + spread_pts

        for k in range(entry_idx + 1, end_bar):
            c_open = opens[k]
            c_high = highs[k]
            c_low = lows[k]
            c_close = closes[k]

            hit_tp = c_low <= tp_target
            hit_sl = c_high >= sl_target

            if hit_tp and not hit_sl:
                return True, tp_target, k - entry_idx
            if hit_sl and not hit_tp:
                return False, sl_target, k - entry_idx
            if hit_tp and hit_sl:
                if c_close <= c_open:
                    return False, sl_target, k - entry_idx
                else:
                    return True, tp_target, k - entry_idx

        last_close = closes[min(entry_idx + max_bars, n - 1)]
        return (eff_entry > last_close), last_close, max_bars


def run_compounding_backtest_fast(
    candles: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    ml_probabilities: Optional[np.ndarray] = None,
    min_ml_prob: float = 0.0,
    loss_policy: Literal["RESET_TO_LEVEL_1", "STEP_DOWN_BY_BALANCE", "PRESERVE_MILESTONES", "ALLOW_1_LOSS_BUFFER"] = "STEP_DOWN_BY_BALANCE",
    initial_balance: float = 20.0,
) -> Dict[str, Any]:
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)

    current_level = 1
    current_balance = initial_balance
    peak_level = 1
    peak_balance = initial_balance

    executed_trades = 0
    wins = 0
    losses = 0
    consecutive_wins = 0
    max_consecutive_wins = 0

    challenge_completions = 0
    completion_stats: List[Dict[str, Any]] = []
    level_milestones_reached = {1: 0, 5: 0, 10: 0, 15: 0, 20: 0, 25: 0, 30: 0}

    active_until_bar = -1
    trades_in_current_run = 0
    lives_remaining = 2 if loss_policy == "ALLOW_1_LOSS_BUFFER" else 1

    for idx, sig in enumerate(signals):
        bar_idx = sig["bar_idx"]
        if bar_idx <= active_until_bar:
            continue

        direction = sig["direction"]

        if ml_probabilities is not None and min_ml_prob > 0.0:
            p_win = ml_probabilities[idx]
            if np.isnan(p_win) or p_win < min_ml_prob:
                continue

        lvl_info = get_ladder_level_info(current_level)
        is_win, exit_price, bars_held = simulate_bracket_fast(
            highs=highs,
            lows=lows,
            closes=closes,
            opens=opens,
            entry_idx=bar_idx,
            direction=direction,
            entry_price=sig["entry_price"],
            tp_points=2.00,
            sl_points=lvl_info["sl_pips"] * 0.10,
        )

        active_until_bar = bar_idx + bars_held
        executed_trades += 1
        trades_in_current_run += 1

        if is_win:
            wins += 1
            consecutive_wins += 1
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            gain = lvl_info["profit_dollars"]
            current_balance = round(current_balance + gain, 2)

            for m in level_milestones_reached:
                if current_level >= m:
                    level_milestones_reached[m] += 1

            if current_level == 30:
                challenge_completions += 1
                completion_stats.append({
                    "completion_no": challenge_completions,
                    "total_trades_taken": trades_in_current_run,
                    "final_balance": current_balance,
                    "date": sig["date"],
                })
                current_level = 1
                current_balance = initial_balance
                trades_in_current_run = 0
                consecutive_wins = 0
                lives_remaining = 2 if loss_policy == "ALLOW_1_LOSS_BUFFER" else 1
            else:
                current_level += 1
                peak_level = max(peak_level, current_level)
        else:
            losses += 1
            consecutive_wins = 0
            loss_amt = lvl_info["risk_dollars"]
            current_balance = max(5.0, round(current_balance - loss_amt, 2))

            if loss_policy == "RESET_TO_LEVEL_1":
                current_level = 1
                current_balance = initial_balance
                trades_in_current_run = 0
            elif loss_policy == "STEP_DOWN_BY_BALANCE":
                current_level = find_closest_level_for_balance(current_balance)
            elif loss_policy == "PRESERVE_MILESTONES":
                if current_level >= 20:
                    current_level = 20
                elif current_level >= 15:
                    current_level = 15
                elif current_level >= 10:
                    current_level = 10
                elif current_level >= 5:
                    current_level = 5
                else:
                    current_level = 1
            elif loss_policy == "ALLOW_1_LOSS_BUFFER":
                lives_remaining -= 1
                if lives_remaining <= 0:
                    current_level = 1
                    current_balance = initial_balance
                    trades_in_current_run = 0
                    lives_remaining = 2

        peak_balance = max(peak_balance, current_balance)

    win_rate = round((wins / executed_trades) * 100, 1) if executed_trades > 0 else 0.0

    return {
        "policy": loss_policy,
        "min_ml_prob": min_ml_prob,
        "total_signals_detected": len(signals),
        "executed_trades": executed_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": win_rate,
        "max_consecutive_wins": max_consecutive_wins,
        "peak_level_reached": peak_level,
        "peak_balance": peak_balance,
        "final_balance": current_balance,
        "challenge_completions": challenge_completions,
        "milestones_count": level_milestones_reached,
        "completion_details": completion_stats,
    }


def main():
    print("=" * 78)
    print("🏆 30-LEVEL GOLD COMPOUNDING 1-YEAR BACKTEST ($20 -> $52,384)")
    print("=" * 78)

    t0 = time.time()
    candles = fetch_gold_candles("XAUUSD", period="1y", interval="1h")
    print(f"Loaded {len(candles)} hourly Gold candles across 1 full year ({time.time() - t0:.2f}s).")
    print(f"Date Range: {candles[0]['date']} to {candles[-1]['date']}\n")

    signals = generate_candidate_signals(candles, session_filter=True)
    print(f"Identified {len(signals)} high-liquidity session trade signals.\n")

    predictor = GoldMLPredictor()
    ml_probs = None
    if predictor.model is not None:
        print("Extracting stationary features & computing vectorized ML probabilities...")
        X_all, feature_names = extract_candle_features(candles)
        sig_features = []
        for idx, sig in enumerate(signals):
            b_idx = sig["bar_idx"]
            f_row = X_all[b_idx]
            dir_sign = 1.0 if sig["direction"] == "BUY" else -1.0
            vec = np.append(f_row, [dir_sign])
            sig_features.append(vec)

        sig_matrix = np.array(sig_features, dtype=float)
        col_means = np.nanmean(sig_matrix, axis=0)
        inds = np.where(np.isnan(sig_matrix))
        sig_matrix[inds] = np.take(col_means, inds[1])

        probs = predictor.model.predict_proba(sig_matrix)[:, 1]
        ml_probs = probs
        print(f"Computed ML probabilities. Mean P(Win): {np.mean(probs):.1%}, Max P(Win): {np.max(probs):.1%}\n")

    scenarios = [
        ("1. Strict Reset (No ML)", 0.0, "RESET_TO_LEVEL_1"),
        ("2. Dynamic Step-Down (No ML)", 0.0, "STEP_DOWN_BY_BALANCE"),
        ("3. ML Gated (P >= 0.60, Step-Down)", 0.60, "STEP_DOWN_BY_BALANCE"),
        ("4. ML Gated (P >= 0.65, Step-Down)", 0.65, "STEP_DOWN_BY_BALANCE"),
        ("5. ML Gated (P >= 0.70, Step-Down)", 0.70, "STEP_DOWN_BY_BALANCE"),
        ("6. Milestone Preservation (L5/L10/L15/L20)", 0.60, "PRESERVE_MILESTONES"),
        ("7. 1-Loss Buffer Per Level (Step-Down)", 0.60, "ALLOW_1_LOSS_BUFFER"),
    ]

    all_results = []
    for label, prob, policy in scenarios:
        t_start = time.time()
        res = run_compounding_backtest_fast(
            candles=candles,
            signals=signals,
            ml_probabilities=ml_probs,
            min_ml_prob=prob,
            loss_policy=policy,
        )
        res["label"] = label
        all_results.append(res)
        print(f"--- {label} --- ({time.time() - t_start:.3f}s)")
        print(f"  Executed Trades: {res['executed_trades']} | Win Rate: {res['win_rate_pct']}% | Max Streak: {res['max_consecutive_wins']}")
        print(f"  Peak Level Reached: Level {res['peak_level_reached']} | Peak Equity: ${res['peak_balance']:,.2f}")
        print(f"  Milestones: L5={res['milestones_count'][5]}, L10={res['milestones_count'][10]}, L15={res['milestones_count'][15]}, L20={res['milestones_count'][20]}")
        print(f"  🏆 30-Level Completions: {res['challenge_completions']}")
        if res["challenge_completions"] > 0:
            for c in res["completion_details"]:
                print(f"    ⭐ Achieved in {c['total_trades_taken']} trades on {c['date']} (Balance: ${c['final_balance']:,.2f})")
        print()

    out_file = os.path.expanduser("~/.gold_trading_data/compounding_1y_backtest_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved complete 1-year backtest results to {out_file}")


if __name__ == "__main__":
    main()
