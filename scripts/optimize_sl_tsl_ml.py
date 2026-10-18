"""
Stop Loss (SL) and Trailing Stop Loss (TSL) Machine Learning Optimization Engine.
Tests multiple configurations of SL, TSL buffers, TP1 scale-outs, and TP2 runner targets
for the current production strategy (COMPOUNDING_TOUCH: Donchian 10 Intrabar Touch + 50 EMA + SuperTrend).

Evaluates:
- Raw Win Rate %, Profit Factor, Net R, Net PnL (pts)
- Walk-Forward Machine Learning Meta-Labeling (LightGBM) probabilities
- ML-Filtered metrics (P >= 0.50, P >= 0.52, P >= 0.55)
- Noise-Adaptive Compounding Ladder simulation ($1,000 baseline seed with progressive ratchet floors)
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from scripts.optimize_strategy_combinations import (
    extract_features_for_bar,
    load_candles_and_indicators,
    simulate_compounding_ladder,
)


def simulate_touch_trade(
    data: Dict[str, Any],
    entry_idx: int,
    action: str,
    entry_price: float,
    sl_dist: float,
    scale_out_r: float = 1.0,
    two_stage: bool = True,
    runner_trail_buffer: float = 6.0,
    runner_t2_multiplier: float = 2.5,
    be_offset: float = 0.50,
    max_bars: int = 48,
) -> Dict[str, Any]:
    """
    Simulates intrabar touch trade execution with two-stage management:
    - Stage 1: 50% scale-out at scale_out_r
    - Stage 2: Move SL to Break-Even + be_offset
    - Stage 3: Dynamic runner trailing stop behind peak excursion up to runner_t2_multiplier
    """
    highs = data["highs"]
    lows = data["lows"]
    closes = data["closes"]
    n = data["n"]

    tp1_dist = scale_out_r * sl_dist
    tp2_dist = runner_t2_multiplier * sl_dist

    stage1_closed = False
    stage1_pnl = 0.0

    if action == "BUY":
        current_sl = entry_price - sl_dist
        tp1_price = entry_price + tp1_dist
        tp2_price = entry_price + tp2_dist
        peak_price = entry_price

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            c_h = highs[k]
            c_l = lows[k]
            peak_price = max(peak_price, c_h)

            if not two_stage:
                if c_h >= tp1_price:
                    return {"outcome": 1, "pnl_r": scale_out_r, "pnl_pts": tp1_dist, "bars": k - entry_idx}
                if c_l <= current_sl:
                    return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
            else:
                if not stage1_closed:
                    # Check SL first to maintain conservative execution in wide bars
                    if c_l <= current_sl:
                        return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
                    if c_h >= tp1_price:
                        stage1_closed = True
                        stage1_pnl = 0.5 * scale_out_r  # 50% banked
                        current_sl = max(current_sl, entry_price + be_offset)
                else:
                    # Runner management
                    trailed_sl = max(entry_price + be_offset, peak_price - runner_trail_buffer)
                    current_sl = max(current_sl, trailed_sl)

                    if c_h >= tp2_price:
                        stage2_pnl = 0.5 * runner_t2_multiplier
                        total_r = stage1_pnl + stage2_pnl
                        return {"outcome": 1, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx}
                    if c_l <= current_sl:
                        exit_p = current_sl
                        stage2_pnl = 0.5 * ((exit_p - entry_price) / sl_dist)
                        total_r = stage1_pnl + stage2_pnl
                        return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx}

        # Time barrier expiration
        final_c = closes[min(entry_idx + max_bars, n - 1)]
        if not two_stage:
            pnl_pts = final_c - entry_price
            pnl_r = pnl_pts / sl_dist
            return {"outcome": 1 if pnl_r > 0 else 0, "pnl_r": pnl_r, "pnl_pts": pnl_pts, "bars": max_bars}
        else:
            if not stage1_closed:
                pnl_pts = final_c - entry_price
                pnl_r = pnl_pts / sl_dist
                return {"outcome": 1 if pnl_r > 0 else 0, "pnl_r": pnl_r, "pnl_pts": pnl_pts, "bars": max_bars}
            else:
                stage2_pnl = 0.5 * ((final_c - entry_price) / sl_dist)
                total_r = stage1_pnl + stage2_pnl
                return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": max_bars}

    else:  # SELL
        current_sl = entry_price + sl_dist
        tp1_price = entry_price - tp1_dist
        tp2_price = entry_price - tp2_dist
        trough_price = entry_price

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            c_h = highs[k]
            c_l = lows[k]
            trough_price = min(trough_price, c_l)

            if not two_stage:
                if c_l <= tp1_price:
                    return {"outcome": 1, "pnl_r": scale_out_r, "pnl_pts": tp1_dist, "bars": k - entry_idx}
                if c_h >= current_sl:
                    return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
            else:
                if not stage1_closed:
                    if c_h >= current_sl:
                        return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
                    if c_l <= tp1_price:
                        stage1_closed = True
                        stage1_pnl = 0.5 * scale_out_r
                        current_sl = min(current_sl, entry_price - be_offset)
                else:
                    trailed_sl = min(entry_price - be_offset, trough_price + runner_trail_buffer)
                    current_sl = min(current_sl, trailed_sl)

                    if c_l <= tp2_price:
                        stage2_pnl = 0.5 * runner_t2_multiplier
                        total_r = stage1_pnl + stage2_pnl
                        return {"outcome": 1, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx}
                    if c_h >= current_sl:
                        exit_p = current_sl
                        stage2_pnl = 0.5 * ((entry_price - exit_p) / sl_dist)
                        total_r = stage1_pnl + stage2_pnl
                        return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx}

        final_c = closes[min(entry_idx + max_bars, n - 1)]
        if not two_stage:
            pnl_pts = entry_price - final_c
            pnl_r = pnl_pts / sl_dist
            return {"outcome": 1 if pnl_r > 0 else 0, "pnl_r": pnl_r, "pnl_pts": pnl_pts, "bars": max_bars}
        else:
            if not stage1_closed:
                pnl_pts = entry_price - final_c
                pnl_r = pnl_pts / sl_dist
                return {"outcome": 1 if pnl_r > 0 else 0, "pnl_r": pnl_r, "pnl_pts": pnl_pts, "bars": max_bars}
            else:
                stage2_pnl = 0.5 * ((entry_price - final_c) / sl_dist)
                total_r = stage1_pnl + stage2_pnl
                return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": max_bars}


def generate_trades_for_params(
    data: Dict[str, Any],
    sl_atr_mult: float = 1.5,
    sl_min_dist: float = 8.0,
    tsl_buffer_mode: str = "FIXED",  # "FIXED" or "ATR"
    tsl_buffer_val: float = 6.0,     # $6.00 if FIXED, or mult * ATR if ATR
    scale_out_r: float = 1.0,
    runner_t2_mult: float = 2.5,
    be_offset: float = 0.50,
) -> List[Dict[str, Any]]:
    """Generates trades across historical data for specified SL and TSL parameters."""
    n = data["n"]
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    opens = data["opens"]
    dates = data["dates"]
    st = data["st"]
    ema50 = data["ema50"]
    dc10 = data["dc10"]
    macd = data["macd"]
    atr14 = data["atr14"]

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

        # Session filter: 07:00 to 19:00 UTC
        if not (7 <= hour <= 19):
            continue

        macro_bullish = c > trend_val
        macro_bearish = c < trend_val

        action = None
        eff_entry = c

        # Intrabar Donchian Touch Trigger
        if h >= prev_d_up and st_dir == 1 and macro_bullish:
            action = "BUY"
            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
        elif l <= prev_d_low and st_dir == -1 and macro_bearish:
            action = "SELL"
            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c

        # MACD continuation secondary trigger
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
        sl_dist = max(sl_atr_mult * atr_val, sl_min_dist)

        # Runner trailing buffer
        if tsl_buffer_mode == "ATR":
            actual_trail_buffer = max(3.0, tsl_buffer_val * atr_val)
        else:
            actual_trail_buffer = tsl_buffer_val

        trade_res = simulate_touch_trade(
            data=data,
            entry_idx=i,
            action=action,
            entry_price=eff_entry,
            sl_dist=sl_dist,
            scale_out_r=scale_out_r,
            two_stage=True,
            runner_trail_buffer=actual_trail_buffer,
            runner_t2_multiplier=runner_t2_mult,
            be_offset=be_offset,
            max_bars=48,
        )

        features = extract_features_for_bar(data, i, action)
        if features is None:
            continue

        trades.append({
            "bar_idx": i,
            "date": dt_str,
            "action": action,
            "entry": eff_entry,
            "sl_dist": sl_dist,
            "outcome": trade_res["outcome"],
            "pnl_r": trade_res["pnl_r"],
            "pnl_pts": trade_res["pnl_pts"],
            "bars": trade_res["bars"],
            "features": features,
        })
        last_trade_bar = i

    return trades


def run_walk_forward_ml(trades: List[Dict[str, Any]]) -> np.ndarray:
    """Train Walk-Forward LightGBM Meta-Labeler and return out-of-sample win probabilities."""
    X = np.array([t["features"] for t in trades])
    y = np.array([t["outcome"] for t in trades])
    n = len(trades)
    probs = np.full(n, np.nan)

    tscv = TimeSeriesSplit(n_splits=5)
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.80,
        colsample_bytree=0.80,
        random_state=42,
        verbose=-1,
    )

    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        clf.fit(X_tr, y_tr)
        probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    return probs


def compute_metrics(
    label: str,
    trades: List[Dict[str, Any]],
    ml_threshold: Optional[float] = None,
    starting_balance: float = 1000.0,
) -> Dict[str, Any]:
    selected = trades
    if ml_threshold is not None:
        selected = [t for t in trades if t.get("ml_prob") is not None and t["ml_prob"] >= ml_threshold]

    total_trades = len(selected)
    if total_trades == 0:
        return {
            "label": label,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "net_pnl_pts": 0.0,
            "final_balance": starting_balance,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = [t for t in selected if t["pnl_r"] > 0]
    losses = [t for t in selected if t["pnl_r"] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0

    gross_profit_pts = sum(t["pnl_pts"] for t in wins)
    gross_loss_pts = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_profit_pts / gross_loss_pts if gross_loss_pts > 0 else 999.0
    net_pnl_pts = gross_profit_pts - gross_loss_pts
    total_r = sum(t["pnl_r"] for t in selected)

    ladder_sim = simulate_compounding_ladder(selected, starting_balance=starting_balance)

    return {
        "label": label,
        "trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_r": round(total_r, 1),
        "net_pnl_pts": round(net_pnl_pts, 1),
        "final_balance": ladder_sim["final_balance"],
        "total_return_pct": ladder_sim["total_return_pct"],
        "max_drawdown_pct": ladder_sim["max_drawdown_pct"],
    }


def run_experiment_suite():
    print("=" * 90)
    print("🔬 STOP LOSS (SL) & TRAILING STOP LOSS (TSL) MACHINE LEARNING OPTIMIZATION ENGINE")
    print("=" * 90)

    data = load_candles_and_indicators("XAUUSD", period="2y", interval="1h")
    print(f"Data Loaded: {data['n']} candles.\n")

    # Storage for all test results
    all_experiments = {}

    # =========================================================================
    # EXPERIMENT 1: Initial Stop Loss (SL) Sensitivity
    # =========================================================================
    print("-" * 90)
    print("📊 EXPERIMENT 1: Stop Loss (SL) Exploration (TSL=$6.00, TP1=1.0R, TP2=2.5R)")
    print("-" * 90)

    sl_configs = [
        {"id": "SL-1.0x-min$5", "mult": 1.0, "min_dist": 5.0, "desc": "Ultra-Tight (1.0x ATR, min $5)"},
        {"id": "SL-1.25x-min$6.5", "mult": 1.25, "min_dist": 6.5, "desc": "Tight (1.25x ATR, min $6.5)"},
        {"id": "SL-1.5x-min$8(Base)", "mult": 1.50, "min_dist": 8.0, "desc": "Production Baseline (1.5x ATR, min $8)"},
        {"id": "SL-1.75x-min$9", "mult": 1.75, "min_dist": 9.0, "desc": "Moderate-Wide (1.75x ATR, min $9)"},
        {"id": "SL-2.0x-min$10", "mult": 2.00, "min_dist": 10.0, "desc": "Wide Noise-Immune (2.0x ATR, min $10)"},
        {"id": "SL-2.5x-min$12", "mult": 2.50, "min_dist": 12.0, "desc": "Deep Macro Breathing (2.5x ATR, min $12)"},
        {"id": "SL-PureATR-1.5x", "mult": 1.50, "min_dist": 0.0, "desc": "Pure ATR (1.5x ATR, no floor)"},
        {"id": "SL-PureFloor-$8", "mult": 0.0, "min_dist": 8.0, "desc": "Pure Fixed Floor ($8.00 flat)"},
    ]

    sl_results = []
    sl_trade_sets = {}

    for cfg in sl_configs:
        trades = generate_trades_for_params(
            data,
            sl_atr_mult=cfg["mult"],
            sl_min_dist=cfg["min_dist"],
            tsl_buffer_mode="FIXED",
            tsl_buffer_val=6.0,
            scale_out_r=1.0,
            runner_t2_mult=2.5,
        )
        probs = run_walk_forward_ml(trades)
        for idx, p in enumerate(probs):
            trades[idx]["ml_prob"] = p
        
        # Valid out-of-sample trades
        valid_trades = [t for t in trades if not np.isnan(t.get("ml_prob", np.nan))]
        sl_trade_sets[cfg["id"]] = valid_trades

        # Eval Raw and ML P>=0.52
        m_raw = compute_metrics(f"{cfg['desc']} [Raw]", valid_trades)
        m_ml = compute_metrics(f"{cfg['desc']} [ML P>=0.52]", valid_trades, ml_threshold=0.52)
        sl_results.append({"cfg": cfg, "raw": m_raw, "ml": m_ml})

    print(f"{'SL Configuration':<38} | {'Raw WR':<7} | {'Raw PF':<7} | {'Raw Net R':<10} | {'ML WR':<7} | {'ML PF':<7} | {'ML Net R':<10} | {'ML Compounding $':<18}")
    print("-" * 125)
    for res in sl_results:
        raw = res["raw"]
        ml = res["ml"]
        print(f"{res['cfg']['desc']:<38} | {raw['win_rate_pct']:>5.1f}% | {raw['profit_factor']:>5.2f} | {raw['net_r']:>+7.1f}R  | {ml['win_rate_pct']:>5.1f}% | {ml['profit_factor']:>5.2f} | {ml['net_r']:>+7.1f}R  | ${ml['final_balance']:>12,.2f}")
    all_experiments["sl_exploration"] = sl_results

    # =========================================================================
    # EXPERIMENT 2: Trailing Stop Loss (TSL) Runner Buffer Exploration
    # =========================================================================
    print("\n" + "-" * 90)
    print("📊 EXPERIMENT 2: Trailing Stop Loss (TSL) Runner Buffer Exploration (SL=1.5x ATR min $8)")
    print("-" * 90)

    tsl_configs = [
        {"id": "TSL-$3", "mode": "FIXED", "val": 3.0, "desc": "Fixed $3.00 Ultra-Tight Trail"},
        {"id": "TSL-$4.5", "mode": "FIXED", "val": 4.5, "desc": "Fixed $4.50 Tight Trail"},
        {"id": "TSL-$6(Base)", "mode": "FIXED", "val": 6.0, "desc": "Fixed $6.00 Production Baseline"},
        {"id": "TSL-$8", "mode": "FIXED", "val": 8.0, "desc": "Fixed $8.00 Medium-Wide Trail"},
        {"id": "TSL-$10", "mode": "FIXED", "val": 10.0, "desc": "Fixed $10.00 Wide Trail"},
        {"id": "TSL-$12", "mode": "FIXED", "val": 12.0, "desc": "Fixed $12.00 Deep Swing Trail"},
        {"id": "TSL-0.75xATR", "mode": "ATR", "val": 0.75, "desc": "Dynamic ATR 0.75x Trail"},
        {"id": "TSL-1.0xATR", "mode": "ATR", "val": 1.00, "desc": "Dynamic ATR 1.00x Trail"},
        {"id": "TSL-1.25xATR", "mode": "ATR", "val": 1.25, "desc": "Dynamic ATR 1.25x Trail"},
        {"id": "TSL-1.5xATR", "mode": "ATR", "val": 1.50, "desc": "Dynamic ATR 1.50x Trail"},
    ]

    tsl_results = []
    tsl_trade_sets = {}

    for cfg in tsl_configs:
        trades = generate_trades_for_params(
            data,
            sl_atr_mult=1.5,
            sl_min_dist=8.0,
            tsl_buffer_mode=cfg["mode"],
            tsl_buffer_val=cfg["val"],
            scale_out_r=1.0,
            runner_t2_mult=2.5,
        )
        probs = run_walk_forward_ml(trades)
        for idx, p in enumerate(probs):
            trades[idx]["ml_prob"] = p
        
        valid_trades = [t for t in trades if not np.isnan(t.get("ml_prob", np.nan))]
        tsl_trade_sets[cfg["id"]] = valid_trades

        m_raw = compute_metrics(f"{cfg['desc']} [Raw]", valid_trades)
        m_ml = compute_metrics(f"{cfg['desc']} [ML P>=0.52]", valid_trades, ml_threshold=0.52)
        tsl_results.append({"cfg": cfg, "raw": m_raw, "ml": m_ml})

    print(f"{'TSL Configuration':<38} | {'Raw WR':<7} | {'Raw PF':<7} | {'Raw Net R':<10} | {'ML WR':<7} | {'ML PF':<7} | {'ML Net R':<10} | {'ML Compounding $':<18}")
    print("-" * 125)
    for res in tsl_results:
        raw = res["raw"]
        ml = res["ml"]
        print(f"{res['cfg']['desc']:<38} | {raw['win_rate_pct']:>5.1f}% | {raw['profit_factor']:>5.2f} | {raw['net_r']:>+7.1f}R  | {ml['win_rate_pct']:>5.1f}% | {ml['profit_factor']:>5.2f} | {ml['net_r']:>+7.1f}R  | ${ml['final_balance']:>12,.2f}")
    all_experiments["tsl_exploration"] = tsl_results

    # =========================================================================
    # EXPERIMENT 3: Runner Target 2 (TP2) Multiplier & Open-Ended Runners
    # =========================================================================
    print("\n" + "-" * 90)
    print("📊 EXPERIMENT 3: Runner Target 2 (TP2) Multiplier (SL=1.5x ATR min $8, TSL=$6.00)")
    print("-" * 90)

    tp2_configs = [
        {"id": "TP2-2.0R", "mult": 2.0, "desc": "Conservative Runner (2.0R)"},
        {"id": "TP2-2.5R(Base)", "mult": 2.5, "desc": "Production Baseline Runner (2.5R)"},
        {"id": "TP2-3.0R", "mult": 3.0, "desc": "Ambitious Runner (3.0R)"},
        {"id": "TP2-3.5R", "mult": 3.5, "desc": "Extended Runner (3.5R)"},
        {"id": "TP2-5.0R-Open", "mult": 5.0, "desc": "Pure Trailing Open Runner (5.0R cap)"},
    ]

    tp2_results = []
    for cfg in tp2_configs:
        trades = generate_trades_for_params(
            data,
            sl_atr_mult=1.5,
            sl_min_dist=8.0,
            tsl_buffer_mode="FIXED",
            tsl_buffer_val=6.0,
            scale_out_r=1.0,
            runner_t2_mult=cfg["mult"],
        )
        probs = run_walk_forward_ml(trades)
        for idx, p in enumerate(probs):
            trades[idx]["ml_prob"] = p
        
        valid_trades = [t for t in trades if not np.isnan(t.get("ml_prob", np.nan))]
        m_raw = compute_metrics(f"{cfg['desc']} [Raw]", valid_trades)
        m_ml = compute_metrics(f"{cfg['desc']} [ML P>=0.52]", valid_trades, ml_threshold=0.52)
        tp2_results.append({"cfg": cfg, "raw": m_raw, "ml": m_ml})

    print(f"{'Runner TP2 Target':<38} | {'Raw WR':<7} | {'Raw PF':<7} | {'Raw Net R':<10} | {'ML WR':<7} | {'ML PF':<7} | {'ML Net R':<10} | {'ML Compounding $':<18}")
    print("-" * 125)
    for res in tp2_results:
        raw = res["raw"]
        ml = res["ml"]
        print(f"{res['cfg']['desc']:<38} | {raw['win_rate_pct']:>5.1f}% | {raw['profit_factor']:>5.2f} | {raw['net_r']:>+7.1f}R  | {ml['win_rate_pct']:>5.1f}% | {ml['profit_factor']:>5.2f} | {ml['net_r']:>+7.1f}R  | ${ml['final_balance']:>12,.2f}")
    all_experiments["tp2_exploration"] = tp2_results

    # =========================================================================
    # EXPERIMENT 4: Stage 1 Scale-out Target (TP1) & Breakeven Trigger
    # =========================================================================
    print("\n" + "-" * 90)
    print("📊 EXPERIMENT 4: Stage 1 Scale-out Target (TP1) & Breakeven Lock Trigger")
    print("-" * 90)

    tp1_configs = [
        {"id": "TP1-0.8R", "scale_r": 0.8, "desc": "Fast De-risk (50% @ 0.8R)"},
        {"id": "TP1-1.0R(Base)", "scale_r": 1.0, "desc": "Production Baseline (50% @ 1.0R)"},
        {"id": "TP1-1.25R", "scale_r": 1.25, "desc": "Patient Scale-Out (50% @ 1.25R)"},
    ]

    tp1_results = []
    for cfg in tp1_configs:
        trades = generate_trades_for_params(
            data,
            sl_atr_mult=1.5,
            sl_min_dist=8.0,
            tsl_buffer_mode="FIXED",
            tsl_buffer_val=6.0,
            scale_out_r=cfg["scale_r"],
            runner_t2_mult=2.5,
        )
        probs = run_walk_forward_ml(trades)
        for idx, p in enumerate(probs):
            trades[idx]["ml_prob"] = p
        
        valid_trades = [t for t in trades if not np.isnan(t.get("ml_prob", np.nan))]
        m_raw = compute_metrics(f"{cfg['desc']} [Raw]", valid_trades)
        m_ml = compute_metrics(f"{cfg['desc']} [ML P>=0.52]", valid_trades, ml_threshold=0.52)
        tp1_results.append({"cfg": cfg, "raw": m_raw, "ml": m_ml})

    print(f"{'Scale-Out TP1 Target':<38} | {'Raw WR':<7} | {'Raw PF':<7} | {'Raw Net R':<10} | {'ML WR':<7} | {'ML PF':<7} | {'ML Net R':<10} | {'ML Compounding $':<18}")
    print("-" * 125)
    for res in tp1_results:
        raw = res["raw"]
        ml = res["ml"]
        print(f"{res['cfg']['desc']:<38} | {raw['win_rate_pct']:>5.1f}% | {raw['profit_factor']:>5.2f} | {raw['net_r']:>+7.1f}R  | {ml['win_rate_pct']:>5.1f}% | {ml['profit_factor']:>5.2f} | {ml['net_r']:>+7.1f}R  | ${ml['final_balance']:>12,.2f}")
    all_experiments["tp1_exploration"] = tp1_results

    # =========================================================================
    # EXPERIMENT 5: Best Combinatorial Synergies & Cross-Grid Leaderboard
    # =========================================================================
    print("\n" + "=" * 90)
    print("🏆 EXPERIMENT 5: Combinatorial Synergies Grid (Top SL x Top TSL x ML Filter Levels)")
    print("=" * 90)

    synergy_candidates = [
        {"name": "1. Production Baseline", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "FIXED", "tsl_val": 6.0, "tp1": 1.0, "tp2": 2.5},
        {"name": "2. Tighter SL (1.25x ATR min $6.5) + Base TSL", "sl_mult": 1.25, "sl_min": 6.5, "tsl_mode": "FIXED", "tsl_val": 6.0, "tp1": 1.0, "tp2": 2.5},
        {"name": "3. Wider SL (1.75x ATR min $9) + Base TSL", "sl_mult": 1.75, "sl_min": 9.0, "tsl_mode": "FIXED", "tsl_val": 6.0, "tp1": 1.0, "tp2": 2.5},
        {"name": "4. Base SL + Dynamic ATR TSL (1.0x ATR)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "ATR", "tsl_val": 1.0, "tp1": 1.0, "tp2": 2.5},
        {"name": "5. Base SL + Dynamic ATR TSL (1.25x ATR)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "ATR", "tsl_val": 1.25, "tp1": 1.0, "tp2": 2.5},
        {"name": "6. Base SL + Wider TSL ($8.00)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "FIXED", "tsl_val": 8.0, "tp1": 1.0, "tp2": 2.5},
        {"name": "7. Base SL + Tighter TSL ($4.50)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "FIXED", "tsl_val": 4.5, "tp1": 1.0, "tp2": 2.5},
        {"name": "8. Wider SL (1.75x ATR) + Dynamic ATR TSL (1.0x ATR)", "sl_mult": 1.75, "sl_min": 9.0, "tsl_mode": "ATR", "tsl_val": 1.0, "tp1": 1.0, "tp2": 2.5},
        {"name": "9. Base SL + Base TSL + Extended Runner (3.0R)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "FIXED", "tsl_val": 6.0, "tp1": 1.0, "tp2": 3.0},
        {"name": "10. Base SL + Dynamic ATR TSL (1.0x ATR) + Extended Runner (3.0R)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "ATR", "tsl_val": 1.0, "tp1": 1.0, "tp2": 3.0},
        {"name": "11. Base SL + Wide TSL ($8.00) + Extended Runner (3.0R)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "FIXED", "tsl_val": 8.0, "tp1": 1.0, "tp2": 3.0},
        {"name": "12. Base SL + Wide TSL ($8.00) + Open Trail (5.0R)", "sl_mult": 1.50, "sl_min": 8.0, "tsl_mode": "FIXED", "tsl_val": 8.0, "tp1": 1.0, "tp2": 5.0},
    ]

    synergy_results = []
    for cand in synergy_candidates:
        trades = generate_trades_for_params(
            data,
            sl_atr_mult=cand["sl_mult"],
            sl_min_dist=cand["sl_min"],
            tsl_buffer_mode=cand["tsl_mode"],
            tsl_buffer_val=cand["tsl_val"],
            scale_out_r=cand["tp1"],
            runner_t2_mult=cand["tp2"],
        )
        probs = run_walk_forward_ml(trades)
        for idx, p in enumerate(probs):
            trades[idx]["ml_prob"] = p
        
        valid_trades = [t for t in trades if not np.isnan(t.get("ml_prob", np.nan))]

        m_raw = compute_metrics(f"{cand['name']} [Raw]", valid_trades)
        m_ml_50 = compute_metrics(f"{cand['name']} [ML P>=0.50]", valid_trades, ml_threshold=0.50)
        m_ml_52 = compute_metrics(f"{cand['name']} [ML P>=0.52]", valid_trades, ml_threshold=0.52)
        m_ml_55 = compute_metrics(f"{cand['name']} [ML P>=0.55]", valid_trades, ml_threshold=0.55)

        synergy_results.append({
            "candidate": cand,
            "raw": m_raw,
            "ml_50": m_ml_50,
            "ml_52": m_ml_52,
            "ml_55": m_ml_55,
        })

    # Sort candidates by ML P>=0.52 Compounding Final Balance
    sorted_synergy = sorted(synergy_results, key=lambda x: x["ml_52"]["final_balance"], reverse=True)

    print("\n" + "=" * 135)
    print(f"{'Synergy Configuration':<48} | {'Trades':<6} | {'ML WR':<7} | {'ML PF':<6} | {'ML Net R':<9} | {'Max DD':<7} | {'Compounding $ (1k seed)':<22}")
    print("=" * 135)
    for s in sorted_synergy:
        cand = s["candidate"]
        ml = s["ml_52"]
        print(f"{cand['name']:<48} | {ml['trades']:<6} | {ml['win_rate_pct']:>5.1f}% | {ml['profit_factor']:>5.2f}  | {ml['net_r']:>+7.1f}R | {ml['max_drawdown_pct']:>5.1f}% | ${ml['final_balance']:>13,.2f} ({ml['total_return_pct']:>+6.1f}%)")
    print("=" * 135)

    all_experiments["synergy_grid"] = sorted_synergy

    # Save to JSON
    output_path = os.path.expanduser("~/.gold_trading_data/sl_tsl_ml_optimization_results.json")
    with open(output_path, "w") as f:
        # Convert to serializable format
        def sanitize(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj
        json.dump(sanitize(all_experiments), f, indent=2)
    print(f"\n✅ All Optimization Results Stored At: {output_path}")

    return all_experiments


if __name__ == "__main__":
    run_experiment_suite()
