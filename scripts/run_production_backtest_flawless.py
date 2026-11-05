#!/usr/bin/env python3
"""
Rigorous, Zero-Flaw 2-Year Production Backtest Engine (Nov 2024 - Nov 2026).

Eliminates all three backtest flaws:
1. ZERO INTRA-BAR LOOKAHEAD: Filters (SuperTrend, 50 EMA, Slope, ADX, Wick) evaluated strictly on closed bar i-1.
   Entry triggers at exact wholesale touch (prev_d_up / prev_d_low) during bar i with zero knowledge of bar i close.
2. CONSERVATIVE SAME-BAR EXECUTION: Adverse stop excursion takes priority on entry bar. No optimistic same-bar TP1 fills.
3. ZERO IN-SAMPLE ML MEMORIZATION: All ML probabilities are strictly Out-Of-Sample (OOF) via 5-Fold Purged & Embargoed Time-Series Cross-Validation.
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_production_backtest import (
    load_market_data, purged_time_series_split,
    compute_deflated_sharpe_ratio, DEFAULT_ASSET_COSTS
)
from test_realistic_backtest import simulate_trade_realistic
from gold_trader.core.indicators import calc_adx
from gold_trader.core.ml_filter import extract_candle_features
import lightgbm as lgb

def generate_flawless_candidates(
    symbol: str,
    sl_atr_mult: float,
    min_sl_dist: float,
    tsl_buffer: float,
    tp2_mult: float,
    be_offset: float,
    scale_out_r: float,
    session_filter: bool,
    require_slope_alignment: bool = True,
    require_prev_wick_filter: bool = True,
    adx_min_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    costs = DEFAULT_ASSET_COSTS.get(symbol.upper(), {"spread": 0.35, "slippage": 0.15, "swap_daily": 0.05})
    spread = costs["spread"]
    slippage = costs["slippage"]
    swap_daily = costs["swap_daily"]

    data = load_market_data(symbol, period="2y", interval="1h")
    n = data["n"]
    closes, highs, lows, opens, dates = data["closes"], data["highs"], data["lows"], data["opens"], data["dates"]
    st, ema50, dc10, atr14 = data["st"], data["ema50"], data["dc10"], data["atr14"]

    adx_dict = calc_adx(highs, lows, closes, 14)
    adx_vals = adx_dict.get("adx", [20.0] * n)

    ema50_slopes = [0.0] * n
    for i in range(6, n):
        if ema50[i] is not None and ema50[i - 5] is not None and ema50[i - 5] > 0:
            ema50_slopes[i] = ((ema50[i] - ema50[i - 5]) / ema50[i - 5]) * 100.0

    candles = [{"open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": 100} for i in range(n)]
    X_all, feature_names = extract_candle_features(candles)

    candidates = []
    last_trade_bar = -100

    for i in range(200, n - 48):
        # 1. Closed Bar Indicators (evaluated strictly at i - 1)
        prev_i = i - 1
        if (
            ema50[prev_i] is None
            or atr14[prev_i] is None
            or dc10["upper"][prev_i] is None
            or dc10["lower"][prev_i] is None
            or st["direction"][prev_i] is None
        ):
            continue

        if i - last_trade_bar < 1:
            continue

        prev_d_up = dc10["upper"][prev_i]
        prev_d_low = dc10["lower"][prev_i]
        st_dir = st["direction"][prev_i]
        trend_val = ema50[prev_i]
        atr_val = atr14[prev_i]
        slope_val = ema50_slopes[prev_i]
        adx_val = adx_vals[prev_i] if (prev_i < len(adx_vals) and adx_vals[prev_i] is not None) else 20.0

        c_prev, o_prev, h_prev, l_prev = closes[prev_i], opens[prev_i], highs[prev_i], lows[prev_i]

        dt_str = dates[i]
        try:
            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = 12

        # 2. Session Filter
        if session_filter and not (7 <= hour <= 19):
            continue

        # 3. Macro Trend & SuperTrend Alignment (from closed bar i - 1)
        macro_bull = c_prev > trend_val
        macro_bear = c_prev < trend_val

        # 4. Slope Alignment (from closed bar i - 1)
        if require_slope_alignment:
            if macro_bull and slope_val < 0.0:
                macro_bull = False
            if macro_bear and slope_val > 0.0:
                macro_bear = False

        # 5. ADX Regime Guard (from closed bar i - 1)
        if adx_min_threshold is not None and adx_val < adx_min_threshold:
            continue

        # 6. Previous Candle Rejection Wick Check (does setup candle show violent rejection?)
        min_body = max(1.0, 0.15 * atr_val)
        if require_prev_wick_filter:
            prev_body_bull = max(c_prev - o_prev, 0.0)
            prev_upper_wick = h_prev - max(c_prev, o_prev)
            if prev_upper_wick > 1.8 * max(prev_body_bull, min_body):
                macro_bull = False

            prev_body_bear = max(o_prev - c_prev, 0.0)
            prev_lower_wick = min(c_prev, o_prev) - l_prev
            if prev_lower_wick > 1.8 * max(prev_body_bear, min_body):
                macro_bear = False

        # 7. Wholesale Touch Trigger on Current Bar i (zero lookahead into bar i close)
        h_curr = highs[i]
        l_curr = lows[i]
        o_curr = opens[i]

        is_bull_trigger = (h_curr >= prev_d_up) and (st_dir == 1) and macro_bull
        is_bear_trigger = (l_curr <= prev_d_low) and (st_dir == -1) and macro_bear

        if not (is_bull_trigger or is_bear_trigger):
            continue

        action = None
        eff_entry = 0.0

        if is_bull_trigger:
            action = "BUY"
            eff_entry = max(o_curr, prev_d_up)  # Enters at wholesale boundary (or open if gapped above)
        elif is_bear_trigger:
            action = "SELL"
            eff_entry = min(o_curr, prev_d_low)

        sl_dist = max(sl_atr_mult * atr_val, min_sl_dist)

        trade_res = simulate_trade_realistic(
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
            spread=spread,
            slippage=slippage,
            swap_daily=swap_daily,
        )

        dir_sign = 1.0 if action == "BUY" else -1.0
        feat_vec = np.append(X_all[prev_i], [dir_sign])
        if np.isnan(feat_vec).any():
            continue

        candidates.append({
            "symbol": symbol,
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
            "features": feat_vec,
        })
        last_trade_bar = i

    # Train Purged 5-fold Out-Of-Sample LightGBM with 23 features (Strict Zero Lookahead)
    X = np.array([c["features"] for c in candidates])
    y = np.array([c["outcome"] for c in candidates])
    n_cand = len(candidates)
    oof_probs = np.full(n_cand, np.nan)
    splits = purged_time_series_split(candidates, n_splits=5)

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.80,
        colsample_bytree=0.80,
        random_state=42,
        verbose=-1,
    )
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    for idx, p in enumerate(oof_probs):
        candidates[idx]["oof_prob"] = p

    return candidates

def simulate_selection(
    candidates: List[Dict[str, Any]],
    ml_threshold: float,
    loss_cooldown_bars: int = 1,
    max_daily_trades: int = 6,
    starting_balance: float = 1000.0,
    planned_risk_usd: float = 75.44,
) -> Dict[str, Any]:
    selected = []
    last_exit = -100
    last_loss = False
    daily = {}

    for c in candidates:
        prob = c.get("oof_prob", 0.0)
        # For No ML (0.00), pass all valid candidates
        if ml_threshold > 0.0:
            if np.isnan(prob) or prob < ml_threshold:
                continue

        b_idx = c["bar_idx"]
        if last_loss and (b_idx - last_exit < loss_cooldown_bars):
            continue

        day = c["date"].split(" ")[0] if " " in c["date"] else c["date"].split("T")[0]
        if daily.get(day, 0) >= max_daily_trades:
            continue

        selected.append(c)
        daily[day] = daily.get(day, 0) + 1
        last_exit = c["exit_bar_idx"]
        last_loss = (c["pnl_r"] <= 0)

    total_trades = len(selected)
    if total_trades == 0:
        return {
            "ml_threshold": ml_threshold,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "net_pnl_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_trades_per_week": 0.0,
            "selected_trades": []
        }

    wins = [t for t in selected if t["pnl_r"] > 0]
    losses = [t for t in selected if t["pnl_r"] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0

    gross_profit_pts = sum(t["pnl_pts"] for t in wins)
    gross_loss_pts = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_profit_pts / gross_loss_pts if gross_loss_pts > 0 else 999.0
    total_r = sum(t["pnl_r"] for t in selected)

    balance = starting_balance
    peak_balance = starting_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    r_multiples = []

    for t in selected:
        trade_pnl = t["pnl_r"] * planned_risk_usd
        balance += trade_pnl
        peak_balance = max(peak_balance, balance)
        dd_usd = peak_balance - balance
        dd_pct = (dd_usd / peak_balance) * 100.0 if peak_balance > 0 else 0.0
        max_dd_usd = max(max_dd_usd, dd_usd)
        max_dd_pct = max(max_dd_pct, dd_pct)
        r_multiples.append(t["pnl_r"])

    net_pnl_usd = balance - starting_balance
    dsr_metrics = compute_deflated_sharpe_ratio(r_multiples, num_trials=50)
    sharpe = dsr_metrics["annualized_sharpe"]

    return {
        "ml_threshold": ml_threshold,
        "total_trades": total_trades,
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_r": round(total_r, 1),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe_ratio": round(sharpe, 2),
        "avg_trades_per_week": round(total_trades / 104.0, 1),
        "selected_trades": selected,
    }

def main():
    print("=" * 135)
    print("🛡️ RIGOROUS ZERO-FLAW 2-YEAR PRODUCTION BACKTEST (NOV 2024 - NOV 2026)")
    print("   ZERO LOOKAHEAD | ZERO SAME-BAR OPTIMISM | ZERO IN-SAMPLE MEMORIZATION (STRICT OOF CV)")
    print("=" * 135)

    configs = [
        {"symbol": "XAUUSD", "sl_atr_mult": 1.5, "min_sl_dist": 8.00, "tsl_buffer": 4.50, "tp2_mult": 3.00, "be_offset": 0.50, "scale_out_r": 1.25, "session_filter": True, "adx_min_threshold": None},
        {"symbol": "BTCUSD", "sl_atr_mult": 1.5, "min_sl_dist": 500.00, "tsl_buffer": 350.00, "tp2_mult": 3.00, "be_offset": 30.00, "scale_out_r": 1.00, "session_filter": False, "adx_min_threshold": None},
        {"symbol": "ETHUSD", "sl_atr_mult": 1.5, "min_sl_dist": 20.00, "tsl_buffer": 10.00, "tp2_mult": 3.00, "be_offset": 2.00, "scale_out_r": 1.00, "session_filter": False, "adx_min_threshold": 22.0},
    ]

    all_candidates = {}
    for cfg in configs:
        sym = cfg["symbol"]
        print(f"\nProcessing {sym} (Zero-Lookahead Candidate Extraction)...")
        cands = generate_flawless_candidates(
            symbol=sym,
            sl_atr_mult=cfg["sl_atr_mult"],
            min_sl_dist=cfg["min_sl_dist"],
            tsl_buffer=cfg["tsl_buffer"],
            tp2_mult=cfg["tp2_mult"],
            be_offset=cfg["be_offset"],
            scale_out_r=cfg["scale_out_r"],
            session_filter=cfg["session_filter"],
            adx_min_threshold=cfg["adx_min_threshold"],
        )
        all_candidates[sym] = cands
        print(f"  {sym}: Generated {len(cands)} verified candidate trades.")

    thresholds = [0.00, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55]
    portfolio_sweep = []
    asset_sweep = {s: [] for s in all_candidates}

    for th in thresholds:
        combined_selected = []
        for sym, cands in all_candidates.items():
            res = simulate_selection(cands, ml_threshold=th)
            asset_sweep[sym].append(res)
            combined_selected.extend(res["selected_trades"])

        combined_selected.sort(key=lambda x: x["date"])
        tot_trades = len(combined_selected)

        if tot_trades > 0:
            wins = [t for t in combined_selected if t["pnl_r"] > 0]
            wr = (len(wins) / tot_trades) * 100.0
            tot_r = sum(t["pnl_r"] for t in combined_selected)
            tot_pnl = sum(t["pnl_r"] * 75.44 for t in combined_selected)

            # Portfolio Drawdown
            bal = 3000.0
            pk = bal
            m_dd = 0.0
            r_mults = []
            for t in combined_selected:
                bal += t["pnl_r"] * 75.44
                pk = max(pk, bal)
                dd = pk - bal
                m_dd = max(m_dd, dd)
                r_mults.append(t["pnl_r"])

            dsr = compute_deflated_sharpe_ratio(r_mults, num_trials=50)
            sh = dsr["annualized_sharpe"]

            gross_win = sum(t["pnl_r"] * 75.44 for t in wins)
            loss_t = [t for t in combined_selected if t["pnl_r"] <= 0]
            gross_loss = abs(sum(t["pnl_r"] * 75.44 for t in loss_t))
            pf = gross_win / gross_loss if gross_loss > 0 else 999.0

            portfolio_sweep.append({
                "threshold": th,
                "total_trades": tot_trades,
                "trades_per_week": round(tot_trades / 104.0, 1),
                "win_rate_pct": round(wr, 1),
                "profit_factor": round(pf, 2),
                "net_r": round(tot_r, 1),
                "net_pnl_usd": round(tot_pnl, 2),
                "avg_win_usd": round(tot_pnl / tot_trades, 2),
                "max_drawdown_usd": round(m_dd, 2),
                "sharpe_ratio": round(sh, 2),
            })

    # Save to JSON
    out_file = "/home/mcpuser/.gold_trading_data/flawless_production_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "portfolio_sweep": portfolio_sweep,
            "asset_sweep": {
                s: [{k: v for k, v in r.items() if k != "selected_trades"} for r in res_list]
                for s, res_list in asset_sweep.items()
            }
        }, f, indent=2)

    print(f"\n✅ Zero-Flaw Results successfully archived to {out_file}")

if __name__ == "__main__":
    main()
