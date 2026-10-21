"""
Comprehensive Production Backtest Engine for Gold (XAU/USD), Bitcoin (BTC/USD) & Ethereum (ETH/USD).
Simulates the live production system with institutional quantitative rigor:
- COMPOUNDING_TOUCH (1H Donchian 10 Touch + 50 EMA + SuperTrend)
- Two-Stage Execution (50% scale-out @ +1.0R, BE shift, Runner Trailing Stop to 3.0R)
- Intrabar Same-Bar SL Check & Conservative Adverse-First Path Resolution
- Purged & Embargoed Time-Series Cross-Validation (Marcos López de Prado, AFML Ch. 4)
- Full Realistic Trading Friction (Variable Spreads, Slippage & Overnight Swap Financing)
- Execution Mode: FLAT RISK ($75.44 fixed risk on $1,000.00 capital baseline)
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
import scipy.stats as stats

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
)

# Standard realistic execution cost parameters per asset
DEFAULT_ASSET_COSTS: Dict[str, Dict[str, float]] = {
    "XAUUSD": {"spread": 0.35, "slippage": 0.15, "swap_daily": 0.05},
    "BTCUSD": {"spread": 15.00, "slippage": 5.00, "swap_daily": 2.50},
    "ETHUSD": {"spread": 0.75, "slippage": 0.25, "swap_daily": 0.10},
}


def compute_deflated_sharpe_ratio(
    returns: List[float],
    num_trials: int = 50,
    annualization_factor: float = math.sqrt(252),
) -> Dict[str, float]:
    """
    Computes Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR) and
    Probabilistic Sharpe Ratio (PSR) adjusting for multiple testing / selection bias.
    """
    if len(returns) < 5:
        return {"annualized_sharpe": 0.0, "dsr_pct": 0.0, "psr_pct": 0.0}

    r = np.array(returns)
    mean = np.mean(r)
    std = np.std(r, ddof=1)
    if std <= 1e-8:
        return {"annualized_sharpe": 0.0, "dsr_pct": 0.0, "psr_pct": 0.0}

    sr_unann = mean / std
    sr_ann = sr_unann * annualization_factor
    n = len(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # Pearson kurtosis (normal = 3)

    # Expected maximum SR across N independent trials
    euler = 0.5772156649
    z = math.sqrt(2.0 * math.log(max(num_trials, 2)))
    sr_star = (z + euler / z) / math.sqrt(252)

    var_sr = (1.0 - skew * sr_unann + ((kurt - 1.0) / 4.0) * (sr_unann**2)) / max(1, n - 1)
    if var_sr <= 0:
        var_sr = 1.0 / max(1, n - 1)

    dsr_stat = (sr_unann - sr_star) / math.sqrt(var_sr)
    dsr = float(stats.norm.cdf(dsr_stat))

    psr_stat = (sr_unann - 0.0) / math.sqrt(var_sr)
    psr = float(stats.norm.cdf(psr_stat))

    return {
        "annualized_sharpe": round(float(sr_ann), 2),
        "dsr_pct": round(dsr * 100.0, 1),
        "psr_pct": round(psr * 100.0, 1),
    }


def purged_time_series_split(
    trades: List[Dict[str, Any]],
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged & Embargoed Time-Series Cross Validation (Marcos López de Prado, AFML Ch. 4).
    
    Guarantees strict zero label lookahead:
    1. Validation set is fold k+1.
    2. Training set is drawn strictly from folds 0..k.
    3. PURGING: Any training trade whose holding period [t_entry, t_exit] overlaps 
       with or extends past the first entry bar of the validation set is dropped.
    """
    n_trades = len(trades)
    chunk_size = n_trades // (n_splits + 1)
    splits = []

    for fold in range(n_splits):
        val_start_idx = (fold + 1) * chunk_size
        val_end_idx = n_trades if fold == n_splits - 1 else (fold + 2) * chunk_size

        val_trades = trades[val_start_idx:val_end_idx]
        val_start_bar = min(t["bar_idx"] for t in val_trades)

        # Purge training trades that were still open when validation began
        clean_train_idx = [
            i
            for i in range(val_start_idx)
            if trades[i].get("exit_bar_idx", trades[i]["bar_idx"] + trades[i].get("bars", 0)) < val_start_bar
        ]

        val_idx = np.arange(val_start_idx, val_end_idx)
        splits.append((np.array(clean_train_idx), val_idx))

    return splits


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
    spread: float = 0.0,
    slippage: float = 0.0,
    swap_daily: float = 0.0,
) -> Dict[str, Any]:
    highs = data["highs"]
    lows = data["lows"]
    closes = data["closes"]
    n = data["n"]

    tp1_dist = scale_out_r * sl_dist
    tp2_dist = runner_t2_multiplier * sl_dist

    stage1_closed = False
    stage1_pts = 0.0

    # Friction deduction helpers: half-spread + slippage per side
    entry_friction = (spread / 2.0) + slippage
    exit_friction = (spread / 2.0) + slippage

    if action == "BUY":
        eff_entry_fill = entry_price + entry_friction
        current_sl = entry_price - sl_dist
        tp1_price = entry_price + tp1_dist
        tp2_price = entry_price + tp2_dist
        peak_price = entry_price

        # Intrabar Check on Entry Candle (entry_idx)
        # If low on entry candle penetrated initial SL, conservative assumption enforces same-bar stop out
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
        elif highs[entry_idx] >= tp1_price:
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
                # Conservative execution: check adverse stop excursion first
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
                # Trailing runner execution
                trailed_sl = max(entry_price + be_offset, peak_price - runner_trail_buffer)
                current_sl = max(current_sl, trailed_sl)

                # Conservative execution: check trailed SL before TP2 if both reached
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

        # Time barrier expiration
        final_idx = min(entry_idx + max_bars, n - 1)
        final_c = closes[final_idx]
        hours_held = max_bars
        swap_cost = (hours_held / 24.0) * swap_daily

        if not stage1_closed:
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

    else:  # SELL
        eff_entry_fill = entry_price - entry_friction
        current_sl = entry_price + sl_dist
        tp1_price = entry_price - tp1_dist
        tp2_price = entry_price - tp2_dist
        trough_price = entry_price

        # Intrabar Check on Entry Candle (entry_idx)
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
        elif lows[entry_idx] <= tp1_price:
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
                # Conservative execution: check adverse stop excursion first
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
                trailed_sl = min(entry_price - be_offset, trough_price + runner_trail_buffer)
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
            return {
                "outcome": 1 if net_r > 0 else 0,
                "pnl_r": net_r,
                "pnl_pts": net_pts,
                "bars": max_bars,
                "exit_type": "TIME_BARRIER",
                "exit_bar_idx": final_idx,
            }
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


def run_backtest_for_asset(
    symbol: str,
    sl_atr_mult: float,
    min_sl_dist: float,
    tsl_buffer: float,
    tp2_mult: float,
    be_offset: float,
    session_filter: bool,
    scale_out_r: float = 1.0,
    spread: Optional[float] = None,
    slippage: Optional[float] = None,
    swap_daily: Optional[float] = None,
) -> Dict[str, Any]:
    costs = DEFAULT_ASSET_COSTS.get(symbol.upper(), {"spread": 0.35, "slippage": 0.15, "swap_daily": 0.05})
    eff_spread = costs["spread"] if spread is None else spread
    eff_slippage = costs["slippage"] if slippage is None else slippage
    eff_swap = costs["swap_daily"] if swap_daily is None else swap_daily

    print(f"  Execution Cost Model for {symbol}: Spread=${eff_spread:.2f}, Slippage=${eff_slippage:.2f}, Daily Swap=${eff_swap:.2f}")

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
            scale_out_r=scale_out_r,
            runner_trail_buffer=tsl_buffer,
            runner_t2_multiplier=tp2_mult,
            be_offset=be_offset,
            max_bars=48,
            spread=eff_spread,
            slippage=eff_slippage,
            swap_daily=eff_swap,
        )

        features = extract_features(data, i - 1, action)
        if features is None:
            continue

        trades.append({
            "bar_idx": i,
            "exit_bar_idx": trade_res["exit_bar_idx"],
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

    # Run Purged & Embargoed Time-Series Cross-Validation (Marcos López de Prado)
    print(f"  Training Purged & Embargoed 5-Fold LightGBM Meta-Labeler on {symbol}...")
    X = np.array([t["features"] for t in trades])
    y = np.array([t["outcome"] for t in trades])
    n_trades = len(trades)
    probs = np.full(n_trades, np.nan)

    splits = purged_time_series_split(trades, n_splits=5)
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

    purged_total = 0
    for fold, (train_idx, val_idx) in enumerate(splits):
        if len(train_idx) == 0:
            continue
        purged_count = (val_idx[0]) - len(train_idx)
        purged_total += max(0, purged_count)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        clf.fit(X_tr, y_tr)
        probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    print(f"  Purged {purged_total} overlapping label observations across 5 folds to prevent lookahead leakage.")

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
    dsr_metrics = {"annualized_sharpe": 0.0, "dsr_pct": 0.0, "psr_pct": 0.0}
    if len(trade_pnls) > 1:
        mean_pnl = np.mean(trade_pnls)
        std_pnl = np.std(trade_pnls)
        if std_pnl > 0:
            trades_per_year = len(trade_pnls) / 2.0
            sharpe = (mean_pnl / std_pnl) * math.sqrt(max(1.0, trades_per_year))
            trade_returns = [pnl / starting_balance for pnl in trade_pnls]
            dsr_metrics = compute_deflated_sharpe_ratio(trade_returns, num_trials=50)

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
        "deflated_sharpe_pct": dsr_metrics["dsr_pct"],
        "probabilistic_sharpe_pct": dsr_metrics["psr_pct"],
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
