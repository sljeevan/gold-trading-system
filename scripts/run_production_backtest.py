"""
Comprehensive Production Backtest Engine for Gold (XAU/USD) & Bitcoin (BTC/USD).
Simulates the live production system with newly deployed parameters:
- COMPOUNDING_TOUCH (1H Donchian 10 Touch + 50 EMA + SuperTrend)
- Two-Stage Execution (50% scale-out @ +1.0R, BE shift, Runner Trailing Stop to 3.0R)
- Walk-Forward LightGBM Meta-Labeling Gate (P >= 0.52)
- 37-Level Noise-Adaptive Compounding Ladder ($1,000 starting seed with progressive ratchet floors)
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
)


def load_market_data(symbol: str, period: str = "2y", interval: str = "1h") -> Dict[str, Any]:
    print(f"Loading {period} {interval} candle data for {symbol}...")
    candles = fetch_gold_candles(symbol, period=period, interval=interval)
    n = len(candles)
    print(f"  Loaded {n} candles ({candles[0]['date']} to {candles[-1]['date']}).")

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    opens = [float(c["open"]) for c in candles]
    dates = [str(c["date"]) for c in candles]

    print(f"  Calculating technical indicators for {symbol}...")
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    ema100 = calc_ema(closes, 100)
    ema200 = calc_ema(closes, 200)
    atr14 = calc_atr(highs, lows, closes, 14)
    atr50 = calc_atr(highs, lows, closes, 50)
    rsi14 = calc_rsi(closes, 14)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    dc10 = calc_donchian(highs, lows, 10)
    macd = calc_macd(closes, 12, 26, 9)

    return {
        "symbol": symbol,
        "candles": candles,
        "n": n,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "opens": opens,
        "dates": dates,
        "ema20": ema20,
        "ema50": ema50,
        "ema100": ema100,
        "ema200": ema200,
        "atr14": atr14,
        "atr50": atr50,
        "rsi14": rsi14,
        "st": st,
        "dc10": dc10,
        "macd": macd,
    }


def extract_features(data: Dict[str, Any], i: int, action: str) -> Optional[List[float]]:
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    opens = data["opens"]
    dates = data["dates"]
    ema20 = data["ema20"]
    ema50 = data["ema50"]
    ema200 = data["ema200"]
    atr14 = data["atr14"]
    atr50 = data["atr50"]
    rsi14 = data["rsi14"]
    dc10 = data["dc10"]
    macd = data["macd"]

    if (
        i < 200
        or ema200[i] is None
        or atr14[i] is None
        or atr50[i] is None
        or rsi14[i] is None
        or dc10["upper"][i] is None
        or macd["histogram"][i] is None
    ):
        return None

    c, o, h, l = closes[i], opens[i], highs[i], lows[i]
    rng = max(0.01, h - l)
    atr_val = max(0.10, atr14[i])
    atr50_val = max(0.10, atr50[i])

    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    dist_ema20 = (c - ema20[i]) / atr_val
    dist_ema50 = (c - ema50[i]) / atr_val
    dist_ema200 = (c - ema200[i]) / atr_val
    macro_div = (ema50[i] - ema200[i]) / atr_val

    rsi_val = rsi14[i]
    rsi_prev = rsi14[i - 3] if i >= 3 and rsi14[i - 3] is not None else rsi_val
    rsi_delta = (rsi_val - rsi_prev)

    dc10_range = max(0.01, (dc10["upper"][i] or h) - (dc10["lower"][i] or l))
    dc10_pos = (c - (dc10["lower"][i] or l)) / dc10_range
    dc_width_atr = dc10_range / atr_val

    macd_hist_atr = macd["histogram"][i] / atr_val
    atr_ratio = atr_val / atr50_val

    dt_str = dates[i]
    hour = 12
    try:
        time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
        hour = int(time_part.split(":")[0])
    except Exception:
        hour = 12

    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    is_london = 1.0 if 7 <= hour <= 15 else 0.0
    is_ny = 1.0 if 12 <= hour <= 20 else 0.0
    is_overlap = 1.0 if 12 <= hour <= 16 else 0.0
    dir_sign = 1.0 if action == "BUY" else -1.0

    return [
        body / rng,
        lower_wick / rng,
        upper_wick / rng,
        rng / atr_val,
        dist_ema20,
        dist_ema50,
        dist_ema200,
        macro_div,
        rsi_val,
        rsi_delta,
        dc10_pos,
        dc_width_atr,
        macd_hist_atr,
        atr_ratio,
        hour_sin,
        hour_cos,
        is_london,
        is_ny,
        is_overlap,
        dir_sign,
    ]


def simulate_trade(
    data: Dict[str, Any],
    entry_idx: int,
    action: str,
    entry_price: float,
    sl_dist: float,
    scale_out_r: float = 1.0,
    runner_trail_buffer: float = 4.50,
    runner_t2_multiplier: float = 3.0,
    be_offset: float = 0.50,
    max_bars: int = 48,
) -> Dict[str, Any]:
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

            if not stage1_closed:
                # Conservative order execution: check adverse stop excursion first
                if c_l <= current_sl:
                    return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx, "exit_type": "INITIAL_SL"}
                if c_h >= tp1_price:
                    stage1_closed = True
                    stage1_pnl = 0.5 * scale_out_r  # 50% cash realized
                    current_sl = max(current_sl, entry_price + be_offset)
            else:
                # Trailing runner execution
                trailed_sl = max(entry_price + be_offset, peak_price - runner_trail_buffer)
                current_sl = max(current_sl, trailed_sl)

                if c_h >= tp2_price:
                    stage2_pnl = 0.5 * runner_t2_multiplier
                    total_r = stage1_pnl + stage2_pnl
                    return {"outcome": 1, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx, "exit_type": "RUNNER_TP2"}
                if c_l <= current_sl:
                    exit_p = current_sl
                    stage2_pnl = 0.5 * ((exit_p - entry_price) / sl_dist)
                    total_r = stage1_pnl + stage2_pnl
                    return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx, "exit_type": "RUNNER_TRAIL"}

        # Time barrier expiration
        final_c = closes[min(entry_idx + max_bars, n - 1)]
        if not stage1_closed:
            pnl_pts = final_c - entry_price
            pnl_r = pnl_pts / sl_dist
            return {"outcome": 1 if pnl_r > 0 else 0, "pnl_r": pnl_r, "pnl_pts": pnl_pts, "bars": max_bars, "exit_type": "TIME_BARRIER"}
        else:
            stage2_pnl = 0.5 * ((final_c - entry_price) / sl_dist)
            total_r = stage1_pnl + stage2_pnl
            return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": max_bars, "exit_type": "TIME_BARRIER"}

    else:  # SELL
        current_sl = entry_price + sl_dist
        tp1_price = entry_price - tp1_dist
        tp2_price = entry_price - tp2_dist
        trough_price = entry_price

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            c_h = highs[k]
            c_l = lows[k]
            trough_price = min(trough_price, c_l)

            if not stage1_closed:
                if c_h >= current_sl:
                    return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx, "exit_type": "INITIAL_SL"}
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
                    return {"outcome": 1, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx, "exit_type": "RUNNER_TP2"}
                if c_h >= current_sl:
                    exit_p = current_sl
                    stage2_pnl = 0.5 * ((entry_price - exit_p) / sl_dist)
                    total_r = stage1_pnl + stage2_pnl
                    return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": k - entry_idx, "exit_type": "RUNNER_TRAIL"}

        final_c = closes[min(entry_idx + max_bars, n - 1)]
        if not stage1_closed:
            pnl_pts = entry_price - final_c
            pnl_r = pnl_pts / sl_dist
            return {"outcome": 1 if pnl_r > 0 else 0, "pnl_r": pnl_r, "pnl_pts": pnl_pts, "bars": max_bars, "exit_type": "TIME_BARRIER"}
        else:
            stage2_pnl = 0.5 * ((entry_price - final_c) / sl_dist)
            total_r = stage1_pnl + stage2_pnl
            return {"outcome": 1 if total_r > 0 else 0, "pnl_r": total_r, "pnl_pts": total_r * sl_dist, "bars": max_bars, "exit_type": "TIME_BARRIER"}


def run_backtest_for_asset(
    symbol: str,
    sl_atr_mult: float,
    min_sl_dist: float,
    tsl_buffer: float,
    tp2_mult: float,
    be_offset: float,
    session_filter: bool,
) -> Dict[str, Any]:
    data = load_market_data(symbol, period="2y", interval="1h")
    n = data["n"]
    closes, highs, lows, opens, dates = data["closes"], data["highs"], data["lows"], data["opens"], data["dates"]
    st, ema50, dc10, macd, atr14 = data["st"], data["ema50"], data["dc10"], data["macd"], data["atr14"]

    trades = []
    last_trade_bar = -100

    print(f"  Scanning {symbol} bars for COMPOUNDING_TOUCH setups...")
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

        # Intrabar Donchian Touch Trigger
        if h >= prev_d_up and st_dir == 1 and macro_bullish:
            action = "BUY"
            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
        elif l <= prev_d_low and st_dir == -1 and macro_bearish:
            action = "SELL"
            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c

        # Momentum continuation secondary trigger
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

        trade_res = simulate_trade(
            data=data,
            entry_idx=i,
            action=action,
            entry_price=eff_entry,
            sl_dist=sl_dist,
            scale_out_r=1.0,
            runner_trail_buffer=tsl_buffer,
            runner_t2_multiplier=tp2_mult,
            be_offset=be_offset,
            max_bars=48,
        )

        features = extract_features(data, i, action)
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
            "exit_type": trade_res["exit_type"],
            "features": features,
        })
        last_trade_bar = i

    print(f"  Generated {len(trades)} candidate setups for {symbol}.")

    # Run Walk-Forward Machine Learning Meta-Labeler (5-Fold TimeSeriesSplit)
    print(f"  Training 5-Fold Walk-Forward LightGBM Meta-Labeler on {symbol}...")
    X = np.array([t["features"] for t in trades])
    y = np.array([t["outcome"] for t in trades])
    n_trades = len(trades)
    probs = np.full(n_trades, np.nan)

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

    for idx, p in enumerate(probs):
        trades[idx]["ml_prob"] = p

    valid_trades = [t for t in trades if not np.isnan(t.get("ml_prob", np.nan))]
    print(f"  Out-of-sample evaluated trades: {len(valid_trades)}")

    return {
        "symbol": symbol,
        "all_trades": valid_trades,
    }


def compute_performance(
    symbol: str,
    trades: List[Dict[str, Any]],
    ml_threshold: Optional[float] = None,
    starting_balance: float = 1000.0,
    base_risk_pct: float = 8.0,
) -> Dict[str, Any]:
    selected = trades
    if ml_threshold is not None:
        selected = [t for t in trades if t.get("ml_prob") is not None and t["ml_prob"] >= ml_threshold]

    total_trades = len(selected)
    if total_trades == 0:
        return {"total_trades": 0, "final_balance": starting_balance}

    wins = [t for t in selected if t["pnl_r"] > 0]
    losses = [t for t in selected if t["pnl_r"] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0

    gross_profit_pts = sum(t["pnl_pts"] for t in wins)
    gross_loss_pts = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_profit_pts / gross_loss_pts if gross_loss_pts > 0 else 999.0
    net_pnl_pts = gross_profit_pts - gross_loss_pts
    total_r = sum(t["pnl_r"] for t in selected)

    avg_win_r = np.mean([t["pnl_r"] for t in wins]) if wins else 0.0
    avg_loss_r = abs(np.mean([t["pnl_r"] for t in losses])) if losses else 1.0
    payoff_ratio = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0.0
    avg_duration_bars = np.mean([t["bars"] for t in selected])

    exit_counts = {}
    for t in selected:
        et = t.get("exit_type", "UNKNOWN")
        exit_counts[et] = exit_counts.get(et, 0) + 1

    # Compounding Ladder Simulation
    balance = starting_balance
    peak_balance = starting_balance
    max_dd_pct = 0.0
    ratchet_floor = 700.0

    for t in selected:
        if balance >= 4000.0:
            ratchet_floor = max(ratchet_floor, 3000.0)
        elif balance >= 2500.0:
            ratchet_floor = max(ratchet_floor, 1800.0)
        elif balance >= 1000.0:
            ratchet_floor = max(ratchet_floor, 700.0)
        elif balance >= 500.0:
            ratchet_floor = max(ratchet_floor, 350.0)
        elif balance >= 250.0:
            ratchet_floor = max(ratchet_floor, 150.0)

        riskable = max(10.0, balance - ratchet_floor)
        risk_usd = min(balance * (base_risk_pct / 100.0), riskable)
        trade_pnl_usd = risk_usd * t["pnl_r"]
        balance += trade_pnl_usd
        balance = max(balance, 10.0)

        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance * 100.0
        max_dd_pct = max(max_dd_pct, dd)

    total_return_pct = ((balance - starting_balance) / starting_balance) * 100.0

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_r": round(total_r, 1),
        "net_pnl_pts": round(net_pnl_pts, 1),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "avg_duration_hours": round(avg_duration_bars, 1),
        "exit_counts": exit_counts,
        "final_balance": round(balance, 2),
        "total_return_pct": round(total_return_pct, 1),
        "peak_balance": round(peak_balance, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
    }


def main():
    print("=" * 90)
    print("🚀 QUANTITATIVE PRODUCTION BACKTEST: GOLD (XAU/USD) & BITCOIN (BTC/USD)")
    print("=" * 90)

    # 1. Backtest Gold (XAUUSD)
    print("\n[1/2] Backtesting Spot Gold (XAU/USD)...")
    gold_data = run_backtest_for_asset(
        symbol="XAUUSD",
        sl_atr_mult=1.5,
        min_sl_dist=8.00,
        tsl_buffer=4.50,
        tp2_mult=3.00,
        be_offset=0.50,
        session_filter=True,
    )

    gold_raw = compute_performance("XAUUSD", gold_data["all_trades"], ml_threshold=None)
    gold_ml = compute_performance("XAUUSD", gold_data["all_trades"], ml_threshold=0.52)

    # Also compute Gold baseline for side-by-side comparison (TSL $6, TP2 2.5R)
    print("  Calculating Gold baseline comparison...")
    gold_base_data = run_backtest_for_asset(
        symbol="XAUUSD",
        sl_atr_mult=1.5,
        min_sl_dist=8.00,
        tsl_buffer=6.00,
        tp2_mult=2.50,
        be_offset=0.50,
        session_filter=True,
    )
    gold_base_ml = compute_performance("XAUUSD", gold_base_data["all_trades"], ml_threshold=0.52)

    # 2. Backtest Bitcoin (BTCUSD)
    print("\n[2/2] Backtesting Bitcoin (BTC/USD)...")
    btc_data = run_backtest_for_asset(
        symbol="BTCUSD",
        sl_atr_mult=1.5,
        min_sl_dist=500.00,
        tsl_buffer=350.00,
        tp2_mult=3.00,
        be_offset=30.00,
        session_filter=False,
    )

    btc_raw = compute_performance("BTCUSD", btc_data["all_trades"], ml_threshold=None)
    btc_ml = compute_performance("BTCUSD", btc_data["all_trades"], ml_threshold=0.52)

    print("  Calculating Bitcoin baseline comparison...")
    btc_base_data = run_backtest_for_asset(
        symbol="BTCUSD",
        sl_atr_mult=1.5,
        min_sl_dist=500.00,
        tsl_buffer=500.00,
        tp2_mult=2.50,
        be_offset=30.00,
        session_filter=False,
    )
    btc_base_ml = compute_performance("BTCUSD", btc_base_data["all_trades"], ml_threshold=0.52)

    # 3. Print Results
    print("\n" + "=" * 125)
    print("🏆 FINAL PRODUCTION PERFORMANCE LEADERBOARD (2 YEARS HISTORICAL WALK-FORWARD)")
    print("=" * 125)
    print(f"{'Asset / Configuration':<42} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Payoff':<6} | {'Max DD':<7} | {'Compounding ($1k Seed)':<22}")
    print("-" * 125)

    print(f"{'Gold Live Production (TSL $4.50, T2 3.0R, ML)':<42} | {gold_ml['total_trades']:<6} | {gold_ml['win_rate_pct']:>5.1f}% | {gold_ml['profit_factor']:>5.2f} | {gold_ml['net_r']:>+7.1f}R | {gold_ml['payoff_ratio']:>5.2f}:1 | {gold_ml['max_drawdown_pct']:>5.1f}% | ${gold_ml['final_balance']:>13,.2f} ({gold_ml['total_return_pct']:>+6.1f}%)")
    print(f"{'Gold Previous Baseline (TSL $6.00, T2 2.5R, ML)':<42} | {gold_base_ml['total_trades']:<6} | {gold_base_ml['win_rate_pct']:>5.1f}% | {gold_base_ml['profit_factor']:>5.2f} | {gold_base_ml['net_r']:>+7.1f}R | {gold_base_ml['payoff_ratio']:>5.2f}:1 | {gold_base_ml['max_drawdown_pct']:>5.1f}% | ${gold_base_ml['final_balance']:>13,.2f} ({gold_base_ml['total_return_pct']:>+6.1f}%)")
    print(f"{'Gold Raw (No ML Filter)':<42} | {gold_raw['total_trades']:<6} | {gold_raw['win_rate_pct']:>5.1f}% | {gold_raw['profit_factor']:>5.2f} | {gold_raw['net_r']:>+7.1f}R | {gold_raw['payoff_ratio']:>5.2f}:1 | {gold_raw['max_drawdown_pct']:>5.1f}% | ${gold_raw['final_balance']:>13,.2f} ({gold_raw['total_return_pct']:>+6.1f}%)")

    print("-" * 125)
    print(f"{'BTC Live Production (TSL $350, T2 3.0R, ML)':<42} | {btc_ml['total_trades']:<6} | {btc_ml['win_rate_pct']:>5.1f}% | {btc_ml['profit_factor']:>5.2f} | {btc_ml['net_r']:>+7.1f}R | {btc_ml['payoff_ratio']:>5.2f}:1 | {btc_ml['max_drawdown_pct']:>5.1f}% | ${btc_ml['final_balance']:>13,.2f} ({btc_ml['total_return_pct']:>+6.1f}%)")
    print(f"{'BTC Previous Baseline (TSL $500, T2 2.5R, ML)':<42} | {btc_base_ml['total_trades']:<6} | {btc_base_ml['win_rate_pct']:>5.1f}% | {btc_base_ml['profit_factor']:>5.2f} | {btc_base_ml['net_r']:>+7.1f}R | {btc_base_ml['payoff_ratio']:>5.2f}:1 | {btc_base_ml['max_drawdown_pct']:>5.1f}% | ${btc_base_ml['final_balance']:>13,.2f} ({btc_base_ml['total_return_pct']:>+6.1f}%)")
    print(f"{'BTC Raw (No ML Filter)':<42} | {btc_raw['total_trades']:<6} | {btc_raw['win_rate_pct']:>5.1f}% | {btc_raw['profit_factor']:>5.2f} | {btc_raw['net_r']:>+7.1f}R | {btc_raw['payoff_ratio']:>5.2f}:1 | {btc_raw['max_drawdown_pct']:>5.1f}% | ${btc_raw['final_balance']:>13,.2f} ({btc_raw['total_return_pct']:>+6.1f}%)")
    print("=" * 125)

    # Save summary report to JSON
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "gold": {
            "live_optimized": gold_ml,
            "previous_baseline": gold_base_ml,
            "raw_no_ml": gold_raw,
        },
        "bitcoin": {
            "live_optimized": btc_ml,
            "previous_baseline": btc_base_ml,
            "raw_no_ml": btc_raw,
        },
    }

    output_path = os.path.expanduser("~/.gold_trading_data/live_production_backtest_results.json")
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

    with open(output_path, "w") as f:
        json.dump(sanitize(report), f, indent=2)

    print(f"\n✅ Production Backtest Results Stored At: {output_path}")


if __name__ == "__main__":
    main()
