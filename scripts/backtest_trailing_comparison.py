#!/usr/bin/env python3
"""
Comprehensive Trailing Stop Loss Backtest Engine: 0.5R vs 1.0R Trailing Stop.
Evaluates Spot Gold (XAUUSD), Bitcoin (BTCUSD), and Ethereum (ETHUSD).

Compares:
A. Two-Stage Production Model (50% TP1 @ 1.0R, SL to BE):
   - Variant 1: Runner Trailing Stop Buffer = 0.5R (0.5 * sl_dist)
   - Variant 2: Runner Trailing Stop Buffer = 1.0R (1.0 * sl_dist)
   
B. Pure Trailing Stop Model (100% Position Trailed, No Partial Scale-Out):
   - Variant 1: Pure Trailing Stop = 0.5R behind peak
   - Variant 2: Pure Trailing Stop = 1.0R behind peak

C. Early-Lock Trailing Model (TP1 @ 0.5R with 0.5R Trail vs TP1 @ 1.0R with 1.0R Trail):
   - Variant 1: 0.5R TP1 + 0.5R Runner Trail
   - Variant 2: 1.0R TP1 + 1.0R Runner Trail

Evaluates institutional metrics:
- Trades, Win Rate %, Profit Factor, Net R-Multiple, Expectancy
- Net P&L ($75.44 Risk Flat Baseline, $1,000 Seed)
- Max Drawdown (% and $), Sharpe Ratio, Calmar Ratio
- Winning Months / 12, Average Winner vs Average Loser (Payoff Ratio)
- Trailing Exit efficiency: runner capture rate and stop-out distribution.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lightgbm as lgb
import numpy as np

from scripts.run_production_backtest import (
    DEFAULT_ASSET_COSTS,
    compute_deflated_sharpe_ratio,
    extract_features,
    load_market_data,
    purged_time_series_split,
)
from scripts.backtest_5pct_risk import compute_comprehensive_metrics, simulate_flat_risk


def simulate_trade_flexible(
    data: Dict[str, Any],
    entry_idx: int,
    action: str,
    entry_price: float,
    sl_dist: float,
    mode: str = "TWO_STAGE",  # "TWO_STAGE" or "PURE_TRAILING"
    scale_out_r: float = 1.0,
    runner_trail_buffer_r: float = 0.5,  # Expressed in R-multiples (e.g. 0.5 or 1.0)
    runner_t2_multiplier: float = 3.0,
    be_offset: float = 0.50,
    max_bars: int = 48,
    spread: float = 0.0,
    slippage: float = 0.0,
    swap_daily: float = 0.0,
) -> Dict[str, Any]:
    """
    Simulates trade execution with either:
    1. TWO_STAGE: 50% partial at scale_out_r, move SL to BE+be_offset, runner trailed by runner_trail_buffer_r * sl_dist to runner_t2_multiplier.
    2. PURE_TRAILING: 100% position trailed by runner_trail_buffer_r * sl_dist behind peak excursion up to runner_t2_multiplier.
    """
    highs = data["highs"]
    lows = data["lows"]
    closes = data["closes"]
    n = data["n"]

    tp1_dist = scale_out_r * sl_dist
    tp2_dist = runner_t2_multiplier * sl_dist
    actual_trail_buffer = runner_trail_buffer_r * sl_dist

    entry_friction = (spread / 2.0) + slippage
    exit_friction = (spread / 2.0) + slippage

    stage1_closed = False
    stage1_pts = 0.0

    if action == "BUY":
        eff_entry_fill = entry_price + entry_friction
        current_sl = entry_price - sl_dist
        tp1_price = entry_price + tp1_dist
        tp2_price = entry_price + tp2_dist
        peak_price = entry_price

        # Intrabar Check on Entry Candle
        if lows[entry_idx] <= current_sl:
            exit_fill = current_sl - exit_friction
            net_pts = exit_fill - eff_entry_fill
            net_r = net_pts / sl_dist
            return {
                "outcome": 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": 0,
                "exit_type": "SAME_BAR_SL",
                "exit_bar_idx": entry_idx,
            }

        if mode == "TWO_STAGE":
            if highs[entry_idx] >= tp1_price:
                stage1_closed = True
                stage1_exit_fill = tp1_price - exit_friction
                stage1_pts = 0.5 * (stage1_exit_fill - eff_entry_fill)
                current_sl = max(current_sl, entry_price + be_offset)
                peak_price = highs[entry_idx]

            for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
                c_h = highs[k]
                c_l = lows[k]
                peak_price = max(peak_price, c_h)
                hours_held = k - entry_idx
                swap_cost = (hours_held / 24.0) * swap_daily

                if not stage1_closed:
                    if c_l <= current_sl:
                        exit_fill = current_sl - exit_friction
                        net_pts = (exit_fill - eff_entry_fill) - swap_cost
                        net_r = net_pts / sl_dist
                        return {
                            "outcome": 1 if net_r > 0 else 0,
                            "pnl_r": net_r,
                            "pnl_pts": net_pts,
                            "bars": hours_held,
                            "exit_type": "INITIAL_SL",
                            "exit_bar_idx": k,
                        }
                    if c_h >= tp1_price:
                        stage1_closed = True
                        stage1_exit_fill = tp1_price - exit_friction
                        stage1_pts = 0.5 * (stage1_exit_fill - eff_entry_fill)
                        current_sl = max(current_sl, entry_price + be_offset)
                else:
                    trailed_sl = max(entry_price + be_offset, peak_price - actual_trail_buffer)
                    current_sl = max(current_sl, trailed_sl)

                    if c_l <= current_sl:
                        stage2_exit_fill = current_sl - exit_friction
                        stage2_pts = 0.5 * (stage2_exit_fill - eff_entry_fill)
                        net_pts = stage1_pts + stage2_pts - swap_cost
                        net_r = net_pts / sl_dist
                        return {
                            "outcome": 1 if net_r > 0 else 0,
                            "pnl_r": net_r,
                            "pnl_pts": net_pts,
                            "bars": hours_held,
                            "exit_type": "RUNNER_TRAIL",
                            "exit_bar_idx": k,
                        }
                    if c_h >= tp2_price:
                        stage2_exit_fill = tp2_price - exit_friction
                        stage2_pts = 0.5 * (stage2_exit_fill - eff_entry_fill)
                        net_pts = stage1_pts + stage2_pts - swap_cost
                        net_r = net_pts / sl_dist
                        return {
                            "outcome": 1 if net_r > 0 else 0,
                            "pnl_r": net_r,
                            "pnl_pts": net_pts,
                            "bars": hours_held,
                            "exit_type": "RUNNER_TP2",
                            "exit_bar_idx": k,
                        }

            # Time barrier
            final_idx = min(entry_idx + max_bars, n - 1)
            final_c = closes[final_idx]
            hours_held = max_bars
            swap_cost = (hours_held / 24.0) * swap_daily
            if not stage1_closed:
                exit_fill = final_c - exit_friction
                net_pts = (exit_fill - eff_entry_fill) - swap_cost
                net_r = net_pts / sl_dist
            else:
                stage2_exit_fill = final_c - exit_friction
                stage2_pts = 0.5 * (stage2_exit_fill - eff_entry_fill)
                net_pts = stage1_pts + stage2_pts - swap_cost
                net_r = net_pts / sl_dist
            return {
                "outcome": 1 if net_r > 0 else 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": max_bars,
                "exit_type": "TIME_BARRIER",
                "exit_bar_idx": final_idx,
            }

        else:  # PURE_TRAILING (100% position trailed)
            for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
                c_h = highs[k]
                c_l = lows[k]
                peak_price = max(peak_price, c_h)
                hours_held = k - entry_idx
                swap_cost = (hours_held / 24.0) * swap_daily

                trailed_sl = peak_price - actual_trail_buffer
                current_sl = max(current_sl, trailed_sl)

                if c_l <= current_sl:
                    exit_fill = current_sl - exit_friction
                    net_pts = (exit_fill - eff_entry_fill) - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0,
                        "pnl_r": net_r,
                        "pnl_pts": net_pts,
                        "bars": hours_held,
                        "exit_type": "TRAILING_STOP" if current_sl > (entry_price - sl_dist + 1e-4) else "INITIAL_SL",
                        "exit_bar_idx": k,
                    }
                if c_h >= tp2_price:
                    exit_fill = tp2_price - exit_friction
                    net_pts = (exit_fill - eff_entry_fill) - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0,
                        "pnl_r": net_r,
                        "pnl_pts": net_pts,
                        "bars": hours_held,
                        "exit_type": "MAX_TP2",
                        "exit_bar_idx": k,
                    }

            final_idx = min(entry_idx + max_bars, n - 1)
            final_c = closes[final_idx]
            swap_cost = (max_bars / 24.0) * swap_daily
            exit_fill = final_c - exit_friction
            net_pts = (exit_fill - eff_entry_fill) - swap_cost
            net_r = net_pts / sl_dist
            return {
                "outcome": 1 if net_r > 0 else 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": max_bars,
                "exit_type": "TIME_BARRIER",
                "exit_bar_idx": final_idx,
            }

    else:  # SELL
        eff_entry_fill = entry_price - entry_friction
        current_sl = entry_price + sl_dist
        tp1_price = entry_price - tp1_dist
        tp2_price = entry_price - tp2_dist
        trough_price = entry_price

        # Intrabar Check on Entry Candle
        if highs[entry_idx] >= current_sl:
            exit_fill = current_sl + exit_friction
            net_pts = eff_entry_fill - exit_fill
            net_r = net_pts / sl_dist
            return {
                "outcome": 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": 0,
                "exit_type": "SAME_BAR_SL",
                "exit_bar_idx": entry_idx,
            }

        if mode == "TWO_STAGE":
            if lows[entry_idx] <= tp1_price:
                stage1_closed = True
                stage1_exit_fill = tp1_price + exit_friction
                stage1_pts = 0.5 * (eff_entry_fill - stage1_exit_fill)
                current_sl = min(current_sl, entry_price - be_offset)
                trough_price = lows[entry_idx]

            for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
                c_h = highs[k]
                c_l = lows[k]
                trough_price = min(trough_price, c_l)
                hours_held = k - entry_idx
                swap_cost = (hours_held / 24.0) * swap_daily

                if not stage1_closed:
                    if c_h >= current_sl:
                        exit_fill = current_sl + exit_friction
                        net_pts = (eff_entry_fill - exit_fill) - swap_cost
                        net_r = net_pts / sl_dist
                        return {
                            "outcome": 1 if net_r > 0 else 0,
                            "pnl_r": net_r,
                            "pnl_pts": net_pts,
                            "bars": hours_held,
                            "exit_type": "INITIAL_SL",
                            "exit_bar_idx": k,
                        }
                    if c_l <= tp1_price:
                        stage1_closed = True
                        stage1_exit_fill = tp1_price + exit_friction
                        stage1_pts = 0.5 * (eff_entry_fill - stage1_exit_fill)
                        current_sl = min(current_sl, entry_price - be_offset)
                else:
                    trailed_sl = min(entry_price - be_offset, trough_price + actual_trail_buffer)
                    current_sl = min(current_sl, trailed_sl)

                    if c_h >= current_sl:
                        stage2_exit_fill = current_sl + exit_friction
                        stage2_pts = 0.5 * (eff_entry_fill - stage2_exit_fill)
                        net_pts = stage1_pts + stage2_pts - swap_cost
                        net_r = net_pts / sl_dist
                        return {
                            "outcome": 1 if net_r > 0 else 0,
                            "pnl_r": net_r,
                            "pnl_pts": net_pts,
                            "bars": hours_held,
                            "exit_type": "RUNNER_TRAIL",
                            "exit_bar_idx": k,
                        }
                    if c_l <= tp2_price:
                        stage2_exit_fill = tp2_price + exit_friction
                        stage2_pts = 0.5 * (eff_entry_fill - stage2_exit_fill)
                        net_pts = stage1_pts + stage2_pts - swap_cost
                        net_r = net_pts / sl_dist
                        return {
                            "outcome": 1 if net_r > 0 else 0,
                            "pnl_r": net_r,
                            "pnl_pts": net_pts,
                            "bars": hours_held,
                            "exit_type": "RUNNER_TP2",
                            "exit_bar_idx": k,
                        }

            final_idx = min(entry_idx + max_bars, n - 1)
            final_c = closes[final_idx]
            hours_held = max_bars
            swap_cost = (hours_held / 24.0) * swap_daily
            if not stage1_closed:
                exit_fill = final_c + exit_friction
                net_pts = (eff_entry_fill - exit_fill) - swap_cost
                net_r = net_pts / sl_dist
            else:
                stage2_exit_fill = final_c + exit_friction
                stage2_pts = 0.5 * (eff_entry_fill - stage2_exit_fill)
                net_pts = stage1_pts + stage2_pts - swap_cost
                net_r = net_pts / sl_dist
            return {
                "outcome": 1 if net_r > 0 else 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": max_bars,
                "exit_type": "TIME_BARRIER",
                "exit_bar_idx": final_idx,
            }

        else:  # PURE_TRAILING (SELL)
            for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
                c_h = highs[k]
                c_l = lows[k]
                trough_price = min(trough_price, c_l)
                hours_held = k - entry_idx
                swap_cost = (hours_held / 24.0) * swap_daily

                trailed_sl = trough_price + actual_trail_buffer
                current_sl = min(current_sl, trailed_sl)

                if c_h >= current_sl:
                    exit_fill = current_sl + exit_friction
                    net_pts = (eff_entry_fill - exit_fill) - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0,
                        "pnl_r": net_r,
                        "pnl_pts": net_pts,
                        "bars": hours_held,
                        "exit_type": "TRAILING_STOP" if current_sl < (entry_price + sl_dist - 1e-4) else "INITIAL_SL",
                        "exit_bar_idx": k,
                    }
                if c_l <= tp2_price:
                    exit_fill = tp2_price + exit_friction
                    net_pts = (eff_entry_fill - exit_fill) - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0,
                        "pnl_r": net_r,
                        "pnl_pts": net_pts,
                        "bars": hours_held,
                        "exit_type": "MAX_TP2",
                        "exit_bar_idx": k,
                    }

            final_idx = min(entry_idx + max_bars, n - 1)
            final_c = closes[final_idx]
            swap_cost = (max_bars / 24.0) * swap_daily
            exit_fill = final_c + exit_friction
            net_pts = (eff_entry_fill - exit_fill) - swap_cost
            net_r = net_pts / sl_dist
            return {
                "outcome": 1 if net_r > 0 else 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": max_bars,
                "exit_type": "TIME_BARRIER",
                "exit_bar_idx": final_idx,
            }


def run_experiment_for_asset(
    symbol: str,
    sl_atr_mult: float,
    min_sl_dist: float,
    be_offset: float,
    session_filter: bool,
    mode: str,
    scale_out_r: float,
    runner_trail_buffer_r: float,
    tp2_mult: float = 3.0,
    data_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    costs = DEFAULT_ASSET_COSTS.get(symbol.upper(), {"spread": 0.35, "slippage": 0.15, "swap_daily": 0.05})
    data = data_cache if data_cache else load_market_data(symbol, period="2y", interval="1h")

    n = data["n"]
    closes, highs, lows, opens, dates = data["closes"], data["highs"], data["lows"], data["opens"], data["dates"]
    st, ema50, dc10, macd, atr14 = data["st"], data["ema50"], data["dc10"], data["macd"], data["atr14"]

    trades = []
    last_trade_bar = -100

    for i in range(200, n - 48):
        if (
            ema50[i] is None
            or atr14[i] is None
            or dc10["upper"][i - 1] is None
            or dc10["lower"][i - 1] is None
            or st["direction"][i] is None
        ):
            continue

        if i - last_trade_bar < 1:
            continue

        c, o, h, l = closes[i], opens[i], highs[i], lows[i]
        prev_d_up = dc10["upper"][i - 1]
        prev_d_low = dc10["lower"][i - 1]
        st_dir = st["direction"][i]
        trend_val = ema50[i]

        dt_str = dates[i]
        try:
            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = 12

        if session_filter and not (7 <= hour <= 19):
            continue

        macro_bullish = c > trend_val
        macro_bearish = c < trend_val

        action = None
        eff_entry = c

        if h >= prev_d_up and st_dir == 1 and macro_bullish:
            action = "BUY"
            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
        elif l <= prev_d_low and st_dir == -1 and macro_bearish:
            action = "SELL"
            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c

        if not action and macd["histogram"][i] is not None and macd["histogram"][i - 1] is not None:
            m_curr, m_sig_curr = macd["macd"][i], macd["signal"][i]
            m_prev, m_sig_prev = macd["macd"][i - 1], macd["signal"][i - 1]
            if st_dir == 1 and macro_bullish and (m_prev <= m_sig_prev and m_curr > m_sig_curr) and c > o:
                action = "BUY"
                eff_entry = c
            elif st_dir == -1 and macro_bearish and (m_prev >= m_sig_prev and m_curr < m_sig_curr) and c < o:
                action = "SELL"
                eff_entry = c

        if not action:
            continue

        atr_val = atr14[i]
        sl_dist = max(sl_atr_mult * atr_val, min_sl_dist)

        trade_res = simulate_trade_flexible(
            data=data,
            entry_idx=i,
            action=action,
            entry_price=eff_entry,
            sl_dist=sl_dist,
            mode=mode,
            scale_out_r=scale_out_r,
            runner_trail_buffer_r=runner_trail_buffer_r,
            runner_t2_multiplier=tp2_mult,
            be_offset=be_offset,
            max_bars=48,
            spread=costs["spread"],
            slippage=costs["slippage"],
            swap_daily=costs["swap_daily"],
        )

        features = extract_features(data, i - 1, action)
        if features is None:
            continue

        trades.append({
            "bar_idx": i,
            "exit_bar_idx": trade_res["exit_bar_idx"],
            "date": dt_str,
            "action": action,
            "entry_price": eff_entry,
            "sl_dist": sl_dist,
            "outcome": trade_res["outcome"],
            "pnl_r": trade_res["pnl_r"],
            "pnl_pts": trade_res["pnl_pts"],
            "bars": trade_res["bars"],
            "exit_type": trade_res["exit_type"],
            "features": features,
        })
        last_trade_bar = trade_res["exit_bar_idx"]

    # Walk-Forward Purged ML Meta-Labeling
    splits = purged_time_series_split(trades, n_splits=5)
    X = np.array([t["features"] for t in trades])
    y = np.array([t["outcome"] for t in trades])
    oos_trades = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.80,
            colsample_bytree=0.80,
            random_state=42 + fold_idx,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]

        for idx, p in zip(val_idx, probs):
            tr = dict(trades[idx])
            tr["ml_prob"] = float(p)
            oos_trades.append(tr)

    return {
        "symbol": symbol,
        "all_trades": trades,
        "oos_trades": oos_trades,
    }


def main():
    print("=" * 115)
    print("🔬 COMPREHENSIVE QUANTITATIVE AUDIT: 0.5R VS 1.0R TRAILING STOP-LOSS ACROSS SPOT GOLD, BITCOIN & ETHEREUM")
    print("   Realistic XM Broker Spreads, Slippage, Overnight Swap Financing & Purged ML Meta-Labeling")
    print("=" * 115)

    asset_configs = [
        ("Spot Gold (XAU/USD)", "XAUUSD", 1.5, 8.0, 0.5, True),
        ("Bitcoin (BTC/USD)", "BTCUSD", 1.5, 500.0, 30.0, False),
        ("Ethereum (ETH/USD)", "ETHUSD", 1.5, 20.0, 2.0, True),
    ]

    print("\n[+] Preloading Market Data for All Three Assets...")
    cached_market_data = {}
    for label, sym, mult, min_sl, be, sess in asset_configs:
        cached_market_data[sym] = load_market_data(sym, period="2y", interval="1h")

    results = {}

    for label, sym, mult, min_sl, be, sess in asset_configs:
        data = cached_market_data[sym]
        print(f"\n{'='*70}\n[+] Processing Walk-Forward Models for {label} ({sym})...\n{'='*70}")

        # Scenario 1A: Two-Stage, Runner Trail Buffer = 0.5R
        res_2s_05r = run_experiment_for_asset(
            sym, mult, min_sl, be, sess,
            mode="TWO_STAGE", scale_out_r=1.0, runner_trail_buffer_r=0.5, data_cache=data
        )
        
        # Scenario 1B: Two-Stage, Runner Trail Buffer = 1.0R
        res_2s_10r = run_experiment_for_asset(
            sym, mult, min_sl, be, sess,
            mode="TWO_STAGE", scale_out_r=1.0, runner_trail_buffer_r=1.0, data_cache=data
        )

        # Scenario 2A: Pure Trailing Stop = 0.5R
        res_pure_05r = run_experiment_for_asset(
            sym, mult, min_sl, be, sess,
            mode="PURE_TRAILING", scale_out_r=1.0, runner_trail_buffer_r=0.5, data_cache=data
        )

        # Scenario 2B: Pure Trailing Stop = 1.0R
        res_pure_10r = run_experiment_for_asset(
            sym, mult, min_sl, be, sess,
            mode="PURE_TRAILING", scale_out_r=1.0, runner_trail_buffer_r=1.0, data_cache=data
        )

        results[sym] = {
            "label": label,
            "2s_05r": res_2s_05r["oos_trades"],
            "2s_10r": res_2s_10r["oos_trades"],
            "pure_05r": res_pure_05r["oos_trades"],
            "pure_10r": res_pure_10r["oos_trades"],
        }

    # =========================================================================
    # DISPLAY SECTION 1: TWO-STAGE PRODUCTION SYSTEM (0.5R VS 1.0R RUNNER TRAIL)
    # =========================================================================
    print("\n" + "=" * 165)
    print("📊 COMPARISON 1: TWO-STAGE SYSTEM — RUNNER TRAILING BUFFER = 0.5R VS 1.0R (TP1 BANKED @ +1.0R, 50% RUNNER)")
    print("   Fixed Risk: $75.44 / Trade ($1,000 Seed Capital Baseline) | ML Meta-Labeling Threshold P >= 0.44")
    print("=" * 165)

    hdr = f"{'Asset / Trailing Buffer':<42} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Payoff':<6} | {'Sharpe':<6} | {'Calmar':<6} | {'Max DD %':<8} | {'Max DD ($)':<11} | {'Net P&L ($)':<14} | {'Runner Exits'}"
    print(hdr)
    print("-" * 165)

    def format_row(name, m, trades):
        tp2_cnt = sum(1 for t in trades if t.get("exit_type") == "RUNNER_TP2")
        trail_cnt = sum(1 for t in trades if t.get("exit_type") in ("RUNNER_TRAIL", "TRAILING_STOP"))
        be_sl_cnt = sum(1 for t in trades if t.get("exit_type") == "INITIAL_SL")
        sb_cnt = sum(1 for t in trades if t.get("exit_type") == "SAME_BAR_SL")
        runner_summary = f"Trail: {trail_cnt} | TP2: {tp2_cnt} | SL: {be_sl_cnt + sb_cnt}"
        return (
            f"{name:<42} | {m['total_trades']:<6} | {m['win_rate_pct']:>5.1f}% | {m['profit_factor']:>5.2f} | "
            f"{m['net_r']:>+7.1f}R | {m['payoff_ratio']:>5.2f}:1 | {m['sharpe_ratio']:>6.2f} | {m['calmar_ratio']:>6.2f} | "
            f"{m['max_dd_pct']:>7.1f}% | ${m['max_dd_usd']:>9,.2f} | ${m['net_pnl']:>+12,.2f} | {runner_summary}"
        )

    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        lbl = results[sym]["label"]
        t_05 = [t for t in results[sym]["2s_05r"] if t.get("ml_prob", 0) >= 0.44]
        t_10 = [t for t in results[sym]["2s_10r"] if t.get("ml_prob", 0) >= 0.44]

        m_05 = compute_comprehensive_metrics(sym, t_05, ml_threshold=None, initial_capital=1000.0, risk_pct=0.07544)
        m_10 = compute_comprehensive_metrics(sym, t_10, ml_threshold=None, initial_capital=1000.0, risk_pct=0.07544)

        print(format_row(f"{lbl} — 0.5R Runner Trail", m_05, t_05))
        print(format_row(f"{lbl} — 1.0R Runner Trail", m_10, t_10))

        delta_r = m_10['net_r'] - m_05['net_r']
        delta_pnl = m_10['net_pnl'] - m_05['net_pnl']
        delta_pf = m_10['profit_factor'] - m_05['profit_factor']
        delta_dd = m_10['max_dd_pct'] - m_05['max_dd_pct']
        print(f"  👉 DELTA (1.0R vs 0.5R): Net R: {delta_r:+.1f}R | P&L: ${delta_pnl:+,.2f} | PF: {delta_pf:+.2f} | Max DD Delta: {delta_dd:+.1f}%")
        print("-" * 165)

    # =========================================================================
    # DISPLAY SECTION 2: PURE TRAILING STOP SYSTEM (0.5R VS 1.0R NO PARTIAL)
    # =========================================================================
    print("\n" + "=" * 165)
    print("📊 COMPARISON 2: PURE TRAILING STOP SYSTEM — 100% POSITION TRAILED (NO PARTIAL SCALE-OUT)")
    print("   Fixed Risk: $75.44 / Trade ($1,000 Seed Capital Baseline) | ML Meta-Labeling Threshold P >= 0.44")
    print("=" * 165)
    print(hdr)
    print("-" * 165)

    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        lbl = results[sym]["label"]
        t_05 = [t for t in results[sym]["pure_05r"] if t.get("ml_prob", 0) >= 0.44]
        t_10 = [t for t in results[sym]["pure_10r"] if t.get("ml_prob", 0) >= 0.44]

        m_05 = compute_comprehensive_metrics(sym, t_05, ml_threshold=None, initial_capital=1000.0, risk_pct=0.07544)
        m_10 = compute_comprehensive_metrics(sym, t_10, ml_threshold=None, initial_capital=1000.0, risk_pct=0.07544)

        print(format_row(f"{lbl} — Pure 0.5R Trail", m_05, t_05))
        print(format_row(f"{lbl} — Pure 1.0R Trail", m_10, t_10))

        delta_r = m_10['net_r'] - m_05['net_r']
        delta_pnl = m_10['net_pnl'] - m_05['net_pnl']
        delta_pf = m_10['profit_factor'] - m_05['profit_factor']
        delta_dd = m_10['max_dd_pct'] - m_05['max_dd_pct']
        print(f"  👉 DELTA (1.0R vs 0.5R): Net R: {delta_r:+.1f}R | P&L: ${delta_pnl:+,.2f} | PF: {delta_pf:+.2f} | Max DD Delta: {delta_dd:+.1f}%")
        print("-" * 165)

    # Save complete JSON data for deeper inspection
    summary_export = {}
    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        summary_export[sym] = {
            "label": results[sym]["label"],
            "2s_05r": compute_comprehensive_metrics(sym, [t for t in results[sym]["2s_05r"] if t.get("ml_prob", 0) >= 0.44], ml_threshold=None),
            "2s_10r": compute_comprehensive_metrics(sym, [t for t in results[sym]["2s_10r"] if t.get("ml_prob", 0) >= 0.44], ml_threshold=None),
            "pure_05r": compute_comprehensive_metrics(sym, [t for t in results[sym]["pure_05r"] if t.get("ml_prob", 0) >= 0.44], ml_threshold=None),
            "pure_10r": compute_comprehensive_metrics(sym, [t for t in results[sym]["pure_10r"] if t.get("ml_prob", 0) >= 0.44], ml_threshold=None),
        }
    with open("/tmp/trailing_stop_comparison_results.json", "w") as f:
        json.dump(summary_export, f, indent=2)
    print("\n✅ Results successfully exported to /tmp/trailing_stop_comparison_results.json")


if __name__ == "__main__":
    main()
