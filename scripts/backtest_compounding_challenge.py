"""
Quantitative Backtest Simulator for the 30-Level Gold Compounding Plan ($20 -> $52,384).
Simulates realistic historical execution on XAUUSD hourly candles across 2 full years,
accounting for spreads ($0.35/oz), ML probability gating, and loss handling policies.
"""
from __future__ import annotations

import json
import logging
import os
import sys
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


def generate_candidate_signals(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate directional impulse candidates based on multi-timeframe trend,
    momentum crossovers, and candlestick rejection wicks.
    """
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

        rng = max(0.01, c_high - c_low)
        lower_wick = min(c_open, c_close) - c_low
        upper_wick = c_high - max(c_open, c_close)

        # Bullish signal:
        # Fast EMA cross or bounce + Trend filter + RSI healthy
        is_bull_cross = (fp <= sp and f > s)
        is_bull_trend = (c_close > t and st_dir == 1)
        is_bull_rejection = (lower_wick / rng >= 0.50 and c_close > s)

        if is_bull_trend and (is_bull_cross or is_bull_rejection) and 40 <= r <= 70:
            signals.append({
                "bar_idx": i,
                "date": candles[i]["date"],
                "direction": "BUY",
                "entry_price": c_close,
            })

        # Bearish signal:
        is_bear_cross = (fp >= sp and f < s)
        is_bear_trend = (c_close < t and st_dir == -1)
        is_bear_rejection = (upper_wick / rng >= 0.50 and c_close < s)

        if is_bear_trend and (is_bear_cross or is_bear_rejection) and 30 <= r <= 60:
            signals.append({
                "bar_idx": i,
                "date": candles[i]["date"],
                "direction": "SELL",
                "entry_price": c_close,
            })

    return signals


def simulate_bracket_outcome(
    candles: List[Dict[str, Any]],
    entry_idx: int,
    direction: str,
    entry_price: float,
    tp_points: float = 2.00,
    sl_points: float = 1.54,
    spread_pts: float = 0.35,  # $0.35 / oz XM MT5 Gold spread
    max_bars: int = 48,
) -> Tuple[bool, float, int]:
    """
    Evaluates forward candles to determine if TP was touched before SL.
    Returns: (is_win, exit_price, bars_held)
    """
    n = len(candles)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]

    if direction == "BUY":
        # Entry pays ask: entry + spread
        eff_entry = entry_price + spread_pts
        tp_target = eff_entry + tp_points
        sl_target = eff_entry - sl_points

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            # Check high for TP and low for SL
            hit_tp = highs[k] >= tp_target
            hit_sl = lows[k] <= sl_target

            if hit_tp and not hit_sl:
                return True, tp_target, k - entry_idx
            if hit_sl and not hit_tp:
                return False, sl_target, k - entry_idx
            if hit_tp and hit_sl:
                # Ambiguous bar: conservative assumption is loss
                return False, sl_target, k - entry_idx

        # Time barrier
        last_close = closes[min(entry_idx + max_bars, n - 1)]
        is_win = (last_close - eff_entry) > 0
        return is_win, last_close, max_bars

    else:  # SELL
        # Sell entry is bid. To close, must buy at ask (bid + spread).
        eff_entry = entry_price
        tp_target = eff_entry - tp_points - spread_pts
        sl_target = eff_entry + sl_points + spread_pts

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            hit_tp = lows[k] <= tp_target
            hit_sl = highs[k] >= sl_target

            if hit_tp and not hit_sl:
                return True, tp_target, k - entry_idx
            if hit_sl and not hit_tp:
                return False, sl_target, k - entry_idx
            if hit_tp and hit_sl:
                return False, sl_target, k - entry_idx

        last_close = closes[min(entry_idx + max_bars, n - 1)]
        is_win = (eff_entry - last_close) > 0
        return is_win, last_close, max_bars


def run_compounding_backtest(
    candles: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    ml_predictor: Optional[GoldMLPredictor] = None,
    min_ml_prob: float = 0.0,  # 0.0 = no ML filter
    loss_policy: Literal["RESET_TO_LEVEL_1", "STEP_DOWN_BY_BALANCE", "PRESERVE_MILESTONES"] = "STEP_DOWN_BY_BALANCE",
    initial_balance: float = 20.0,
) -> Dict[str, Any]:
    """
    Run chronological walk through signals simulating the 30-level ladder.
    """
    X_all, feature_names = extract_candle_features(candles) if ml_predictor else (None, None)

    current_level = 1
    current_balance = initial_balance
    peak_level = 1
    peak_balance = initial_balance

    total_signals = len(signals)
    executed_trades = 0
    wins = 0
    losses = 0
    consecutive_wins = 0
    max_consecutive_wins = 0

    challenge_completions = 0
    completion_stats: List[Dict[str, Any]] = []

    active_until_bar = -1
    trades_in_current_run = 0

    for sig in signals:
        bar_idx = sig["bar_idx"]

        # Prevent overlapping trades (one active trade at a time)
        if bar_idx <= active_until_bar:
            continue

        direction = sig["direction"]

        # ML Probability Filter
        if ml_predictor is not None and X_all is not None:
            latest_feat = X_all[bar_idx]
            if np.isnan(latest_feat).any():
                continue
            dir_sign = 1.0 if direction == "BUY" else -1.0
            vec = np.append(latest_feat, [dir_sign]).reshape(1, -1)
            prob_win = ml_predictor.model.predict_proba(vec)[0][1]

            if prob_win < min_ml_prob:
                continue  # Skip low-conviction trade

        # Execute trade
        lvl_info = get_ladder_level_info(current_level)
        lots = lvl_info["lot_size"]

        is_win, exit_price, bars_held = simulate_bracket_outcome(
            candles=candles,
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

            if current_level == 30:
                challenge_completions += 1
                completion_stats.append({
                    "completion_no": challenge_completions,
                    "total_trades_taken": trades_in_current_run,
                    "final_balance": current_balance,
                    "date": sig["date"],
                })
                # Reset for next challenge run
                current_level = 1
                current_balance = initial_balance
                trades_in_current_run = 0
                consecutive_wins = 0
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
                elif current_level >= 10:
                    current_level = 10
                else:
                    current_level = 1

        peak_balance = max(peak_balance, current_balance)

    win_rate = round((wins / executed_trades) * 100, 1) if executed_trades > 0 else 0.0

    return {
        "policy": loss_policy,
        "min_ml_prob": min_ml_prob,
        "total_signals_detected": total_signals,
        "executed_trades": executed_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": win_rate,
        "max_consecutive_wins": max_consecutive_wins,
        "peak_level_reached": peak_level,
        "peak_balance": peak_balance,
        "final_balance": current_balance,
        "challenge_completions": challenge_completions,
        "completion_details": completion_stats,
    }


def main():
    print("=" * 75)
    print("🏆 30-LEVEL GOLD COMPOUNDING BACKTEST ($20 -> $52,384)")
    print("=" * 75)

    candles = fetch_gold_candles("XAUUSD", period="2y", interval="1h")
    print(f"Loaded {len(candles)} hourly Gold candles (2 Years).")

    signals = generate_candidate_signals(candles)
    print(f"Identified {len(signals)} structural trend & momentum trade candidates.\n")

    predictor = GoldMLPredictor()
    if predictor.model is None:
        print("Training ML predictor...")
        predictor.train_model(period="1y", interval="1h")

    # Scenarios to test:
    # 1. No ML filter (Baseline) with Reset on Loss
    # 2. No ML filter with Step-Down by Balance
    # 3. ML Filter (P >= 0.65) with Step-Down by Balance
    # 4. ML Filter (P >= 0.70) with Step-Down by Balance
    # 5. ML Filter (P >= 0.70) with Milestone Preservation

    scenarios = [
        ("No ML Filter (Strict Reset)", 0.0, "RESET_TO_LEVEL_1"),
        ("No ML Filter (Step-Down)", 0.0, "STEP_DOWN_BY_BALANCE"),
        ("ML Gated (P >= 0.60, Step-Down)", 0.60, "STEP_DOWN_BY_BALANCE"),
        ("ML Gated (P >= 0.65, Step-Down)", 0.65, "STEP_DOWN_BY_BALANCE"),
        ("ML Gated (P >= 0.70, Step-Down)", 0.70, "STEP_DOWN_BY_BALANCE"),
        ("ML Gated (P >= 0.65, Milestones)", 0.65, "PRESERVE_MILESTONES"),
    ]

    results = []
    for label, prob, policy in scenarios:
        res = run_compounding_backtest(
            candles=candles,
            signals=signals,
            ml_predictor=predictor if prob > 0 else None,
            min_ml_prob=prob,
            loss_policy=policy,
        )
        res["label"] = label
        results.append(res)
        print(f"--- {label} ---")
        print(f"  Executed Trades: {res['executed_trades']} | Win Rate: {res['win_rate_pct']}% | Max Streak: {res['max_consecutive_wins']}")
        print(f"  Peak Level Reached: Level {res['peak_level_reached']} | Peak Equity: ${res['peak_balance']:,.2f}")
        print(f"  30-Level Completions: {res['challenge_completions']}")
        if res["challenge_completions"] > 0:
            for c in res["completion_details"]:
                print(f"    ⭐ Achieved in {c['total_trades_taken']} trades on {c['date']} (Balance: ${c['final_balance']:,.2f})")
        print()

    # Save summary
    out_path = os.path.expanduser("~/.gold_trading_data/compounding_backtest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete results to {out_path}")


if __name__ == "__main__":
    main()
