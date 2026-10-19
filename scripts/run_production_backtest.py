"""
Comprehensive Production Backtest Engine for Gold (XAU/USD), Bitcoin (BTC/USD) & Ethereum (ETH/USD).
Simulates the live production system with newly deployed parameters:
- COMPOUNDING_TOUCH (1H Donchian 10 Touch + 50 EMA + SuperTrend)
- Two-Stage Execution (50% scale-out @ +1.0R, BE shift, Runner Trailing Stop to 3.0R)
- Walk-Forward LightGBM Meta-Labeling Gate (P >= 0.52)
- Execution Mode: FLAT RISK ($75.44 fixed risk on $1,000.00 capital baseline, Compounding Ladder Removed)
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

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

        # Strictly extract features from bar i-1 (closed candle prior to entry) to eliminate all intrabar lookahead
        features = extract_features(data, i - 1, action)
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
    planned_risk_usd: float = 75.44,
) -> Dict[str, Any]:
    selected = trades
    if ml_threshold is not None:
        selected = [t for t in trades if t.get("ml_prob") is not None and t["ml_prob"] >= ml_threshold]

    total_trades = len(selected)
    if total_trades == 0:
        return {
            "symbol": symbol,
            "total_trades": 0,
            "starting_balance": starting_balance,
            "final_balance": starting_balance,
            "net_pnl_usd": 0.0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "payoff_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "sharpe_ratio": 0.0,
        }

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

    # Standard Realistic Flat Risk Execution Model ($75.44 fixed risk on $1,000 baseline)
    balance = starting_balance
    peak_balance = starting_balance
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    trade_pnls = []

    for t in selected:
        trade_pnl = planned_risk_usd * t["pnl_r"]
        trade_pnls.append(trade_pnl)
        balance += trade_pnl
        balance = max(balance, 0.0)
        peak_balance = max(peak_balance, balance)
        dd_usd = peak_balance - balance
        dd_pct = (dd_usd / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_dd_usd = max(max_dd_usd, dd_usd)
        max_dd_pct = max(max_dd_pct, dd_pct)

    net_pnl_usd = balance - starting_balance
    total_return_pct = (net_pnl_usd / starting_balance) * 100.0

    # Sharpe ratio estimate (annualized over 2-year horizon)
    sharpe = 0.0
    if len(trade_pnls) > 1:
        mean_pnl = np.mean(trade_pnls)
        std_pnl = np.std(trade_pnls)
        if std_pnl > 0:
            trades_per_year = len(trade_pnls) / 2.0
            sharpe = (mean_pnl / std_pnl) * math.sqrt(max(1.0, trades_per_year))

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
        "starting_balance": starting_balance,
        "planned_risk_usd": planned_risk_usd,
        "final_balance": round(balance, 2),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "total_return_pct": round(total_return_pct, 1),
        "peak_balance": round(peak_balance, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "sharpe_ratio": round(sharpe, 2),
    }


def main():
    print("=" * 95)
    print("🚀 QUANTITATIVE PRODUCTION BACKTEST: LIVE DEPLOYED CONFIGURATION (2 YEARS)")
    print("=" * 95)

    # 1. Backtest Gold (XAUUSD)
    print("\n[1/3] Backtesting Spot Gold (XAU/USD)...")
    gold_data = run_backtest_for_asset(
        symbol="XAUUSD",
        sl_atr_mult=1.5,
        min_sl_dist=8.00,
        tsl_buffer=4.50,
        tp2_mult=3.00,
        be_offset=0.50,
        session_filter=True,
    )

    gold_live = compute_performance("XAUUSD", gold_data["all_trades"], ml_threshold=0.44)
    gold_strict = compute_performance("XAUUSD", gold_data["all_trades"], ml_threshold=0.52)
    gold_raw = compute_performance("XAUUSD", gold_data["all_trades"], ml_threshold=None)

    # 2. Backtest Bitcoin (BTCUSD)
    print("\n[2/3] Backtesting Bitcoin (BTC/USD)...")
    btc_data = run_backtest_for_asset(
        symbol="BTCUSD",
        sl_atr_mult=1.5,
        min_sl_dist=500.00,
        tsl_buffer=350.00,
        tp2_mult=3.00,
        be_offset=30.00,
        session_filter=False,
    )

    btc_live = compute_performance("BTCUSD", btc_data["all_trades"], ml_threshold=0.44)
    btc_strict = compute_performance("BTCUSD", btc_data["all_trades"], ml_threshold=0.52)
    btc_raw = compute_performance("BTCUSD", btc_data["all_trades"], ml_threshold=None)

    # 3. Backtest Ethereum (ETHUSD)
    print("\n[3/3] Backtesting Ethereum (ETH/USD)...")
    # 3a. 24/7 Full Crypto Schedule
    eth_data_247 = run_backtest_for_asset(
        symbol="ETHUSD",
        sl_atr_mult=1.5,
        min_sl_dist=20.00,
        tsl_buffer=10.00,
        tp2_mult=3.00,
        be_offset=2.00,
        session_filter=False,
    )
    eth_live_247 = compute_performance("ETHUSD", eth_data_247["all_trades"], ml_threshold=0.44)
    eth_strict_247 = compute_performance("ETHUSD", eth_data_247["all_trades"], ml_threshold=0.52)
    eth_raw_247 = compute_performance("ETHUSD", eth_data_247["all_trades"], ml_threshold=None)

    # 3b. London / NY Core Session Filter (07:00-19:00 UTC)
    print("  Evaluating Ethereum with London/NY Core Session Filter (07:00-19:00 UTC)...")
    eth_data_session = run_backtest_for_asset(
        symbol="ETHUSD",
        sl_atr_mult=1.5,
        min_sl_dist=20.00,
        tsl_buffer=10.00,
        tp2_mult=3.00,
        be_offset=2.00,
        session_filter=True,
    )
    eth_live_session = compute_performance("ETHUSD", eth_data_session["all_trades"], ml_threshold=0.44)
    eth_strict_session = compute_performance("ETHUSD", eth_data_session["all_trades"], ml_threshold=0.52)
    eth_raw_session = compute_performance("ETHUSD", eth_data_session["all_trades"], ml_threshold=None)

    # 4. Print Results Leaderboard (Flat Risk Baseline)
    print("\n" + "=" * 148)
    print("🏆 FINAL PRODUCTION PERFORMANCE LEADERBOARD (2 YEARS WALK-FORWARD — FLAT RISK $75.44 ON $1,000 BASELINE)")
    print("=" * 148)
    header = f"{'Asset / Configuration':<44} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Payoff':<6} | {'Sharpe':<6} | {'Max DD %':<8} | {'Max DD ($)':<11} | {'Net P&L ($)':<14} | {'Final Balance':<14}"
    print(header)
    print("-" * 148)

    def row(name, m):
        return f"{name:<44} | {m['total_trades']:<6} | {m['win_rate_pct']:>5.1f}% | {m['profit_factor']:>5.2f} | {m['net_r']:>+7.1f}R | {m['payoff_ratio']:>5.2f}:1 | {m['sharpe_ratio']:>6.2f} | {m['max_drawdown_pct']:>7.1f}% | ${m['max_drawdown_usd']:>9,.2f} | ${m['net_pnl_usd']:>+12,.2f} | ${m['final_balance']:>12,.2f}"

    print(row("Gold: Live Config (TSL $4.50, T2 3.0R, ML 0.44)", gold_live))
    print(row("Gold: Strict ML Gate (ML P >= 0.52)", gold_strict))
    print(row("Gold: Raw Baseline (No ML Filter)", gold_raw))

    print("-" * 148)
    print(row("BTC: Live Config (TSL $350, T2 3.0R, ML 0.44)", btc_live))
    print(row("BTC: Strict ML Gate (ML P >= 0.52)", btc_strict))
    print(row("BTC: Raw Baseline (No ML Filter)", btc_raw))

    print("-" * 148)
    print(row("ETH: Live Config 24/7 (TSL $10, ML 0.44)", eth_live_247))
    print(row("ETH: Strict ML 24/7 (ML P >= 0.52)", eth_strict_247))
    print(row("ETH: Raw Baseline 24/7 (No ML Filter)", eth_raw_247))
    print(row("ETH: London/NY Session (ML P >= 0.44)", eth_live_session))
    print(row("ETH: London/NY Session (ML P >= 0.52)", eth_strict_session))
    print("=" * 148)

    # Save summary report to JSON
    from datetime import timezone
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gold": {
            "live_config_ml_044": gold_live,
            "strict_ml_052": gold_strict,
            "raw_no_ml": gold_raw,
        },
        "bitcoin": {
            "live_config_ml_044": btc_live,
            "strict_ml_052": btc_strict,
            "raw_no_ml": btc_raw,
        },
        "ethereum_247": {
            "live_config_ml_044": eth_live_247,
            "strict_ml_052": eth_strict_247,
            "raw_no_ml": eth_raw_247,
        },
        "ethereum_session_filtered": {
            "live_config_ml_044": eth_live_session,
            "strict_ml_052": eth_strict_session,
            "raw_no_ml": eth_raw_session,
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
