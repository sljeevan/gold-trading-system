"""
Institutional Strategy Combinations & Machine Learning Optimization Engine.
Tests and benchmarks 12 distinct quantitative combinations across 2 years of historical data.
Evaluates:
- Win Rate %
- Profit Factor
- Net Profit (Points & $)
- Maximum Drawdown
- Compounding Ladder Final Account Equity (Starting from $1,000.00 baseline)
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
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


def load_candles_and_indicators(symbol: str = "XAUUSD", period: str = "2y", interval: str = "1h") -> Dict[str, Any]:
    print(f"Loading {period} {interval} candle data for {symbol}...")
    candles = fetch_gold_candles(symbol, period=period, interval=interval)
    n = len(candles)
    print(f"Loaded {n} candles ({candles[0]['date']} to {candles[-1]['date']}).")

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    opens = [float(c["open"]) for c in candles]
    dates = [c["date"] for c in candles]

    print("Computing technical indicators...")
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    ema100 = calc_ema(closes, 100)
    ema200 = calc_ema(closes, 200)
    atr14 = calc_atr(highs, lows, closes, 14)
    atr50 = calc_atr(highs, lows, closes, 50)
    rsi14 = calc_rsi(closes, 14)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    dc10 = calc_donchian(highs, lows, 10)
    dc15 = calc_donchian(highs, lows, 15)
    dc20 = calc_donchian(highs, lows, 20)
    macd = calc_macd(closes, 12, 26, 9)

    return {
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
        "dc15": dc15,
        "dc20": dc20,
        "macd": macd,
    }


def extract_features_for_bar(data: Dict[str, Any], i: int, action: str) -> Optional[List[float]]:
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


def simulate_trade_execution(
    data: Dict[str, Any],
    entry_idx: int,
    action: str,
    entry_price: float,
    sl_dist: float,
    rr_multiplier: float,
    two_stage: bool = False,
    scale_out_r: float = 1.0,
    runner_trail_buffer: float = 6.0,
    runner_t2_multiplier: float = 2.5,
    max_bars: int = 48,
) -> Dict[str, Any]:
    highs = data["highs"]
    lows = data["lows"]
    closes = data["closes"]
    dates = data["dates"]
    n = data["n"]

    tp1_dist = rr_multiplier * sl_dist if not two_stage else scale_out_r * sl_dist
    tp2_dist = runner_t2_multiplier * sl_dist

    stage1_closed = False
    stage1_pnl = 0.0
    stage2_pnl = 0.0

    if action == "BUY":
        current_sl = entry_price - sl_dist
        tp1_price = entry_price + tp1_dist
        tp2_price = entry_price + tp2_dist
        peak_price = entry_price

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            c_h = highs[k]
            c_l = lows[k]
            c_c = closes[k]
            peak_price = max(peak_price, c_h)

            if not two_stage:
                if c_h >= tp1_price:
                    return {"outcome": 1, "pnl_r": rr_multiplier, "pnl_pts": tp1_dist, "bars": k - entry_idx}
                if c_l <= current_sl:
                    return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
            else:
                # Two-Stage Execution
                if not stage1_closed:
                    if c_h >= tp1_price:
                        stage1_closed = True
                        stage1_pnl = 0.5 * scale_out_r  # 50% closed at +1.0R
                        current_sl = max(current_sl, entry_price + 0.50)  # Move SL to BE+$0.50
                    elif c_l <= current_sl:
                        return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
                else:
                    # Runner management: trail behind peak
                    trailed_sl = max(entry_price + 0.50, peak_price - runner_trail_buffer)
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

        # Expiration at time barrier
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
            c_c = closes[k]
            trough_price = min(trough_price, c_l)

            if not two_stage:
                if c_l <= tp1_price:
                    return {"outcome": 1, "pnl_r": rr_multiplier, "pnl_pts": tp1_dist, "bars": k - entry_idx}
                if c_h >= current_sl:
                    return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
            else:
                if not stage1_closed:
                    if c_l <= tp1_price:
                        stage1_closed = True
                        stage1_pnl = 0.5 * scale_out_r
                        current_sl = min(current_sl, entry_price - 0.50)
                    elif c_h >= current_sl:
                        return {"outcome": 0, "pnl_r": -1.0, "pnl_pts": -sl_dist, "bars": k - entry_idx}
                else:
                    trailed_sl = min(entry_price - 0.50, trough_price + runner_trail_buffer)
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


def simulate_compounding_ladder(
    trades: List[Dict[str, Any]],
    starting_balance: float = 1000.0,
    base_risk_pct: float = 8.0,
    ratchet_floor: float = 700.0,
) -> Dict[str, Any]:
    balance = starting_balance
    peak_balance = starting_balance
    max_dd_pct = 0.0
    equity_curve = [balance]

    # Progressive Ratchet Tier Floors
    # $250 -> $150, $500 -> $350, $1000 -> $700, $2500 -> $1800, $4000 -> $3000
    for t in trades:
        # Update dynamic ratchet floor
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

        # Risk amount: base_risk_pct of current balance, but never risk below floor
        riskable = max(10.0, balance - ratchet_floor)
        risk_usd = min(balance * (base_risk_pct / 100.0), riskable)

        # Conviction weight if present
        weight = t.get("ml_weight", 1.0)
        effective_risk = risk_usd * weight

        pnl_r = t["pnl_r"]
        trade_pnl_usd = effective_risk * pnl_r
        balance += trade_pnl_usd
        balance = max(balance, 10.0)  # Micro-seed minimum

        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance * 100.0
        max_dd_pct = max(max_dd_pct, dd)
        equity_curve.append(balance)

    return {
        "final_balance": round(balance, 2),
        "total_return_pct": round(((balance - starting_balance) / starting_balance) * 100.0, 1),
        "peak_balance": round(peak_balance, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
    }


def run_strategy_backtest(
    data: Dict[str, Any],
    donchian_period: int = 10,
    trend_ema: int = 50,
    dual_ema_200: bool = False,
    trigger_mode: str = "TOUCH",
    rr_multiplier: float = 1.25,
    two_stage: bool = False,
    include_macd_continuation: bool = True,
    session_filter: bool = True,
) -> List[Dict[str, Any]]:
    n = data["n"]
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    opens = data["opens"]
    dates = data["dates"]
    st = data["st"]
    ema = data[f"ema{trend_ema}"]
    ema200 = data["ema200"]
    dc = data[f"dc{donchian_period}"]
    macd = data["macd"]
    atr14 = data["atr14"]

    trades = []
    last_trade_bar = -100

    for i in range(200, n - 48):
        if (
            ema[i] is None
            or atr14[i] is None
            or dc["upper"][i - 1] is None
            or dc["lower"][i - 1] is None
            or st["direction"][i] is None
        ):
            continue

        # Prevent duplicate entries on same bar
        if i - last_trade_bar < 1:
            continue

        c, o, h, l = closes[i], opens[i], highs[i], lows[i]
        prev_d_up = dc["upper"][i - 1]
        prev_d_low = dc["lower"][i - 1]
        st_dir = st["direction"][i]
        trend_val = ema[i]
        macro_200_val = ema200[i]

        dt_str = dates[i]
        hour = 12
        try:
            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = 12

        if session_filter and not (7 <= hour <= 19):
            continue

        bullish_macro = c > trend_val and (not dual_ema_200 or (macro_200_val and c > macro_200_val))
        bearish_macro = c < trend_val and (not dual_ema_200 or (macro_200_val and c < macro_200_val))

        action = None
        eff_entry = c

        # Trigger 1: Donchian Touch or Close
        if trigger_mode == "TOUCH":
            if h >= prev_d_up and st_dir == 1 and bullish_macro:
                action = "BUY"
                eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
            elif l <= prev_d_low and st_dir == -1 and bearish_macro:
                action = "SELL"
                eff_entry = min(o, prev_d_low) if o >= prev_d_low else c
        else:  # CLOSE
            if c > prev_d_up and st_dir == 1 and bullish_macro:
                action = "BUY"
                eff_entry = c
            elif c < prev_d_low and st_dir == -1 and bearish_macro:
                action = "SELL"
                eff_entry = c

        # Trigger 2: MACD Continuation (if enabled and no Donchian trigger)
        if not action and include_macd_continuation and macd["histogram"][i] is not None and macd["histogram"][i - 1] is not None:
            m_curr, m_sig_curr = macd["macd"][i], macd["signal"][i]
            m_prev, m_sig_prev = macd["macd"][i - 1], macd["signal"][i - 1]
            if st_dir == 1 and bullish_macro and (m_prev <= m_sig_prev and m_curr > m_sig_curr) and c > o:
                action = "BUY"
                eff_entry = c
            elif st_dir == -1 and bearish_macro and (m_prev >= m_sig_prev and m_curr < m_sig_curr) and c < o:
                action = "SELL"
                eff_entry = c

        if not action:
            continue

        atr_val = atr14[i]
        sl_dist = max(1.5 * atr_val, 8.0)

        # Simulate trade execution
        trade_res = simulate_trade_execution(
            data=data,
            entry_idx=i,
            action=action,
            entry_price=eff_entry,
            sl_dist=sl_dist,
            rr_multiplier=rr_multiplier,
            two_stage=two_stage,
            scale_out_r=1.0,
            runner_trail_buffer=6.0,
            runner_t2_multiplier=2.5,
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


def run_walk_forward_ml_eval(trades: List[Dict[str, Any]]) -> np.ndarray:
    """Trains a walk-forward LightGBM meta-labeler and returns out-of-sample probabilities."""
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


def evaluate_combination_metrics(
    name: str,
    trades: List[Dict[str, Any]],
    ml_filter_threshold: Optional[float] = None,
    dynamic_ml_sizing: bool = False,
    starting_balance: float = 1000.0,
) -> Dict[str, Any]:
    selected = trades
    if ml_filter_threshold is not None:
        selected = [t for t in trades if t.get("ml_prob") is not None and t["ml_prob"] >= ml_filter_threshold]
    elif dynamic_ml_sizing:
        selected = []
        for t in trades:
            prob = t.get("ml_prob")
            if prob is None or np.isnan(prob):
                continue
            if prob < 0.48:
                weight = 0.40  # De-risk
            elif prob < 0.58:
                weight = 1.00  # Base size
            else:
                weight = 1.35  # High-conviction boost
            t_copy = dict(t)
            t_copy["ml_weight"] = weight
            selected.append(t_copy)

    total_trades = len(selected)
    if total_trades == 0:
        return {"name": name, "trades": 0, "win_rate": 0, "profit_factor": 0, "net_pnl_pts": 0, "final_balance": starting_balance}

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
        "name": name,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_r": round(total_r, 1),
        "net_pnl_pts": round(net_pnl_pts, 1),
        "final_balance": ladder_sim["final_balance"],
        "total_return_pct": ladder_sim["total_return_pct"],
        "max_drawdown_pct": ladder_sim["max_drawdown_pct"],
    }


def main():
    print("=" * 80)
    print("🚀 QUANTITATIVE GOLD TRADING STRATEGY COMBINATIONS & ML OPTIMIZATION ENGINE")
    print("=" * 80)

    data = load_candles_and_indicators("XAUUSD", period="2y", interval="1h")

    # 1. Base Strategy Setups
    print("\n[1/3] Generating trade setups for each core technical configuration...")

    # Config A: Donchian 10 Touch + 50 EMA, 1.25R
    trades_d10_touch_125r = run_strategy_backtest(data, donchian_period=10, trend_ema=50, trigger_mode="TOUCH", rr_multiplier=1.25)
    # Config B: Donchian 15 Breakout Close + 200 EMA, 1.8R
    trades_d15_close_180r = run_strategy_backtest(data, donchian_period=15, trend_ema=200, trigger_mode="CLOSE", rr_multiplier=1.80)
    # Config C: Donchian 10 Touch + 50 EMA, 1.20R Fast Target
    trades_d10_touch_120r = run_strategy_backtest(data, donchian_period=10, trend_ema=50, trigger_mode="TOUCH", rr_multiplier=1.20)
    # Config D: Donchian 10 Touch + 50 EMA + Two-Stage Execution (50% @ 1.0R, Runner 2.5R)
    trades_d10_touch_twostage = run_strategy_backtest(data, donchian_period=10, trend_ema=50, trigger_mode="TOUCH", two_stage=True)
    # Config E: Donchian 15 Breakout Close + 50 EMA, 1.25R
    trades_d15_close_125r = run_strategy_backtest(data, donchian_period=15, trend_ema=50, trigger_mode="CLOSE", rr_multiplier=1.25)
    # Config F: Donchian 10 Touch + Dual EMA (50 & 200 EMA trend alignment), 1.25R
    trades_d10_dual_ema_125r = run_strategy_backtest(data, donchian_period=10, trend_ema=50, dual_ema_200=True, trigger_mode="TOUCH", rr_multiplier=1.25)

    print(f"Generated setups:")
    print(f"  Config A (D10 Touch 1.25R): {len(trades_d10_touch_125r)} trades")
    print(f"  Config B (D15 Close 1.80R): {len(trades_d15_close_180r)} trades")
    print(f"  Config C (D10 Touch 1.20R): {len(trades_d10_touch_120r)} trades")
    print(f"  Config D (D10 Touch Two-Stage): {len(trades_d10_touch_twostage)} trades")
    print(f"  Config E (D15 Close 1.25R): {len(trades_d15_close_125r)} trades")
    print(f"  Config F (D10 Dual EMA 1.25R): {len(trades_d10_dual_ema_125r)} trades")

    # 2. Run Machine Learning Meta-Labeling Walk-Forward
    print("\n[2/3] Training Walk-Forward Out-Of-Sample ML Meta-Labelers for each configuration...")
    probs_a = run_walk_forward_ml_eval(trades_d10_touch_125r)
    for i, p in enumerate(probs_a):
        trades_d10_touch_125r[i]["ml_prob"] = p

    probs_d = run_walk_forward_ml_eval(trades_d10_touch_twostage)
    for i, p in enumerate(probs_d):
        trades_d10_touch_twostage[i]["ml_prob"] = p

    probs_f = run_walk_forward_ml_eval(trades_d10_dual_ema_125r)
    for i, p in enumerate(probs_f):
        trades_d10_dual_ema_125r[i]["ml_prob"] = p

    probs_e = run_walk_forward_ml_eval(trades_d15_close_125r)
    for i, p in enumerate(probs_e):
        trades_d15_close_125r[i]["ml_prob"] = p

    # Filter to out-of-sample evaluated trades only for consistent comparison
    eval_a = [t for t in trades_d10_touch_125r if not np.isnan(t.get("ml_prob", np.nan))]
    eval_d = [t for t in trades_d10_touch_twostage if not np.isnan(t.get("ml_prob", np.nan))]
    eval_f = [t for t in trades_d10_dual_ema_125r if not np.isnan(t.get("ml_prob", np.nan))]
    eval_e = [t for t in trades_d15_close_125r if not np.isnan(t.get("ml_prob", np.nan))]

    # 3. Evaluate 12 Combinations
    print("\n[3/3] Benchmarking 12 Quantitative Strategy Combinations...")
    results = []

    # Comb 1: Baseline Touch (No ML)
    results.append(evaluate_combination_metrics("1. Baseline Donchian 10 Touch (50 EMA, 1.25R, No ML)", eval_a))
    # Comb 2: Donchian 15 Breakout Close (200 EMA, 1.8R, No ML)
    results.append(evaluate_combination_metrics("2. Compounding Hybrid Breakout (200 EMA, 1.80R, No ML)", trades_d15_close_180r[-len(eval_a):]))
    # Comb 3: Donchian 10 Touch (50 EMA, 1.20R Fast Target, No ML)
    results.append(evaluate_combination_metrics("3. Donchian 10 Touch Fast TP (50 EMA, 1.20R, No ML)", trades_d10_touch_120r[-len(eval_a):]))
    # Comb 4: Donchian 10 Touch + Two-Stage Execution (No ML)
    results.append(evaluate_combination_metrics("4. Donchian 10 Two-Stage (50% @ 1.0R, Trail 2.5R, No ML)", eval_d))
    # Comb 5: Donchian 15 Breakout Close (50 EMA, 1.25R, No ML)
    results.append(evaluate_combination_metrics("5. Donchian 15 Breakout Close (50 EMA, 1.25R, No ML)", eval_e))
    # Comb 6: Donchian 10 Touch + ML Filter P >= 0.50
    results.append(evaluate_combination_metrics("6. Donchian 10 Touch (1.25R) + ML Filter (P >= 0.50)", eval_a, ml_filter_threshold=0.50))
    # Comb 7: Donchian 10 Touch + ML Filter P >= 0.55
    results.append(evaluate_combination_metrics("7. Donchian 10 Touch (1.25R) + ML Filter (P >= 0.55)", eval_a, ml_filter_threshold=0.55))
    # Comb 8: Donchian 10 Touch + ML Filter P >= 0.58
    results.append(evaluate_combination_metrics("8. Donchian 10 Touch (1.25R) + ML Filter (P >= 0.58)", eval_a, ml_filter_threshold=0.58))
    # Comb 9: Donchian 10 Touch + Two-Stage + ML Filter P >= 0.52
    results.append(evaluate_combination_metrics("9. Donchian 10 Two-Stage + ML Filter (P >= 0.52)", eval_d, ml_filter_threshold=0.52))
    # Comb 10: Donchian 10 Touch + Dual EMA (50 & 200) + ML Filter P >= 0.52
    results.append(evaluate_combination_metrics("10. Donchian 10 Dual-EMA + ML Filter (P >= 0.52)", eval_f, ml_filter_threshold=0.52))
    # Comb 11: Donchian 15 Close + ML Filter P >= 0.52
    results.append(evaluate_combination_metrics("11. Donchian 15 Close (1.25R) + ML Filter (P >= 0.52)", eval_e, ml_filter_threshold=0.52))
    # Comb 12: Donchian 10 Touch + Dynamic ML Sizing (0.4x de-risk, 1.35x boost)
    results.append(evaluate_combination_metrics("12. Donchian 10 Touch + Dynamic ML Confidence Sizing", eval_a, dynamic_ml_sizing=True))

    print("\n" + "=" * 125)
    print(f"{'Strategy Combination':<52} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<7} | {'Drawdown':<8} | {'Final Equity ($1k seed)':<18}")
    print("=" * 125)

    # Sort combinations by final compounded equity
    sorted_res = sorted(results, key=lambda x: x["final_balance"], reverse=True)
    for r in sorted_res:
        name = r["name"]
        trades = r["total_trades"]
        wr = r["win_rate_pct"]
        pf = r["profit_factor"]
        net_r = r["net_r"]
        dd = r["max_drawdown_pct"]
        bal = r["final_balance"]
        ret = r["total_return_pct"]
        print(f"{name:<52} | {trades:<6} | {wr:>5.1f}% | {pf:>5.2f} | {net_r:>+6.1f}R | {dd:>7.1f}% | ${bal:>10,.2f} ({ret:>+6.1f}%)")
    print("=" * 125)

    # Save summary json
    output_path = os.path.expanduser("~/.gold_trading_data/strategy_combinations_benchmark.json")
    import json
    with open(output_path, "w") as f:
        json.dump(sorted_res, f, indent=2)
    print(f"\n✅ Benchmark completed and stored at: {output_path}")


if __name__ == "__main__":
    main()
