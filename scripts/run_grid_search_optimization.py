"""
Comprehensive Grid Search Optimization Engine for Multi-Asset Autonomous Trading.
Systematically evaluates permutations of:
1. Seed Capital ($500, $1,000, $2,500)
2. Fixed Dollar Risk per Trade ($25, $35, $50, $75)
3. Trailing Stop Loss (TSL) Buffers (Tight, Medium/Baseline, Wide)
4. Machine Learning Meta-Labeling Gates (Raw No-ML, ML >= 0.44, ML >= 0.48, ML >= 0.52)
5. Concurrency Modes (Uncapped vs Max 1 Position per Asset)

Generates an institutional Efficiency Leaderboard ranked by Calmar-Sharpe Efficiency Score.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import lightgbm as lgb
import numpy as np

from scripts.run_production_backtest_5years import (
    ASSET_CONFIG,
    compute_deflated_sharpe_ratio,
    extract_features,
    filter_concurrency,
    load_5y_market_data,
    purged_time_series_split,
    simulate_trade,
)

# Trailing Stop Loss configurations across assets
TSL_REGIMES = {
    "Tight": {
        "XAUUSD": 3.00,
        "BTCUSD": 200.00,
        "ETHUSD": 6.00,
    },
    "Medium": {
        "XAUUSD": 4.50,   # Production Baseline
        "BTCUSD": 350.00,  # Production Baseline
        "ETHUSD": 10.00,   # Production Baseline
    },
    "Wide": {
        "XAUUSD": 6.50,
        "BTCUSD": 550.00,
        "ETHUSD": 16.00,
    },
}

# Grid parameters
SEED_CAPITALS = [500.0, 1000.0, 2500.0]
RISK_LEVELS = [25.0, 35.0, 50.0, 75.0]
ML_THRESHOLDS = [
    ("Raw (No ML)", None),
    ("ML >= 0.44", 0.44),
    ("ML >= 0.48", 0.48),
    ("ML >= 0.52", 0.52),
]
CONCURRENCY_MODES = ["Uncapped", "Max 1 Position"]


def scan_and_train_asset_tsl(
    symbol: str,
    market_data: Dict[str, Any],
    tsl_buffer: float,
    session_filter: bool,
) -> List[Dict[str, Any]]:
    """
    Scans 5-year bars for FLAT_RISK_TOUCH triggers with specified TSL buffer
    and evaluates out-of-sample trades via Purged 5-Fold Cross-Validation LightGBM.
    """
    cfg = ASSET_CONFIG[symbol]
    n = market_data["n"]
    closes, highs, lows, opens, dates = (
        market_data["closes"],
        market_data["highs"],
        market_data["lows"],
        market_data["opens"],
        market_data["dates"],
    )
    st, ema50, dc10, macd, atr14 = (
        market_data["st"],
        market_data["ema50"],
        market_data["dc10"],
        market_data["macd"],
        market_data["atr14"],
    )

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

        # Primary Trigger: 1H Donchian 10 Intrabar Touch
        if h >= prev_d_up and st_dir == 1 and macro_bullish:
            action = "BUY"
            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
        elif l <= prev_d_low and st_dir == -1 and macro_bearish:
            action = "SELL"
            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c

        # Secondary Trigger: MACD momentum continuation
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
        sl_dist = max(cfg["sl_atr_mult"] * atr_val, cfg["min_sl"])

        trade_res = simulate_trade(
            data=market_data,
            entry_idx=i,
            action=action,
            entry_price=eff_entry,
            sl_dist=sl_dist,
            scale_out_r=cfg["tp1_rr"],
            runner_trail_buffer=tsl_buffer,
            runner_t2_multiplier=cfg["tp2_mult"],
            be_offset=cfg["be_offset"],
            max_bars=48,
            spread=cfg["spread"],
            slippage=cfg["slippage"],
            swap_daily=cfg["swap_daily"],
        )

        features = extract_features(market_data, i - 1, action)
        if features is None:
            continue

        trades.append({
            "symbol": symbol,
            "bar_idx": i,
            "exit_bar_idx": trade_res["exit_bar_idx"],
            "entry_date": dt_str,
            "exit_date": trade_res["exit_date"],
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

    # Purged & Embargoed Cross-Validation (5 Folds) with LightGBM
    X = np.array([t["features"] for t in trades])
    y = np.array([t["outcome"] for t in trades])
    n_trades = len(trades)
    probs = np.full(n_trades, np.nan)

    splits = purged_time_series_split(trades, n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if len(train_idx) < 50 or len(val_idx) == 0:
            continue
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 20,
            "verbose": -1,
            "random_state": 42 + fold_idx,
            "n_jobs": 2,
        }
        bst = lgb.train(params, train_data, num_boost_round=60)
        p_val = bst.predict(X_val)
        probs[val_idx] = p_val

    nan_mask = np.isnan(probs)
    if np.any(nan_mask):
        probs[nan_mask] = 0.50

    for i, t in enumerate(trades):
        t["ml_prob"] = float(probs[i])

    return trades


def simulate_equity_curve(
    trades: List[Dict[str, Any]],
    starting_balance: float,
    planned_risk_usd: float,
    years: float = 5.0,
) -> Dict[str, Any]:
    """
    Computes path-dependent equity curve and quantitative metrics.
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "net_pnl_usd": 0.0,
            "final_balance": starting_balance,
            "roi_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "efficiency_score": 0.0,
        }

    wins = [t for t in trades if t["pnl_r"] > 0]
    losses = [t for t in trades if t["pnl_r"] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0

    gross_profit_r = sum(t["pnl_r"] for t in wins)
    gross_loss_r = abs(sum(t["pnl_r"] for t in losses))
    pf = gross_profit_r / gross_loss_r if gross_loss_r > 0 else 999.0
    net_r = gross_profit_r - gross_loss_r

    balance = starting_balance
    peak_balance = starting_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    trade_pnls = []

    for t in trades:
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
    roi_pct = (net_pnl_usd / starting_balance) * 100.0
    ann_return_pct = roi_pct / years
    calmar = ann_return_pct / max(1.0, max_dd_pct) if max_dd_pct > 0 else 0.0

    sharpe = 0.0
    if len(trade_pnls) > 1:
        mean_pnl = np.mean(trade_pnls)
        std_pnl = np.std(trade_pnls)
        if std_pnl > 0:
            trades_per_year = len(trade_pnls) / max(0.5, years)
            sharpe = (mean_pnl / std_pnl) * math.sqrt(max(1.0, trades_per_year))

    # Institutional Calmar-Sharpe Efficiency Score
    # Heavily penalizes high drawdowns (>40%), drops to 0 if drawdown exceeds 80%
    dd_penalty = max(0.0, 1.0 - (max_dd_pct / 80.0))
    efficiency = max(0.0, calmar * min(5.0, max(0.2, sharpe)) * dd_penalty)

    return {
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_r": round(net_r, 1),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "final_balance": round(balance, 2),
        "roi_pct": round(roi_pct, 1),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "calmar_ratio": round(calmar, 2),
        "sharpe_ratio": round(sharpe, 2),
        "efficiency_score": round(efficiency, 2),
    }


def main():
    start_time = time.time()
    print("=" * 110)
    print("🔬 MULTI-ASSET GRID SEARCH OPTIMIZATION ENGINE")
    print("   Exploration Parameters:")
    print(f"   • Seed Capital per Asset: {SEED_CAPITALS}")
    print(f"   • Fixed Risk per Trade:   {RISK_LEVELS}")
    print(f"   • TSL Buffers:            {list(TSL_REGIMES.keys())}")
    print(f"   • ML Conviction Gates:    {[lbl for lbl, _ in ML_THRESHOLDS]}")
    print(f"   • Concurrency Modes:      {CONCURRENCY_MODES}")
    print("=" * 110)

    # 1. Load Market Data
    print("\n[Phase 1/3] Loading 5-Year Candle Archives for Gold, Bitcoin, Ethereum...")
    market_data = {
        "XAUUSD": load_5y_market_data("XAUUSD"),
        "BTCUSD": load_5y_market_data("BTCUSD"),
        "ETHUSD": load_5y_market_data("ETHUSD"),
    }

    # 2. Pre-generate and ML Meta-Label Trades for each TSL regime
    print("\n[Phase 2/3] Simulating Trades & Fitting Purged 5-Fold LightGBM across TSL Regimes...")
    trade_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

    for tsl_name, tsl_dict in TSL_REGIMES.items():
        print(f"\n  ▶ Processing TSL Regime: {tsl_name} ({tsl_dict})")
        for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
            t0 = time.time()
            tsl_val = tsl_dict[sym]
            session_filt = True if sym == "XAUUSD" else False
            raw_trades = scan_and_train_asset_tsl(sym, market_data[sym], tsl_val, session_filt)
            trade_cache[tsl_name][sym] = raw_trades
            dur = time.time() - t0
            print(f"    • {sym} ({tsl_name}): {len(raw_trades)} trades simulated & ML-labeled in {dur:.2f}s")

    # 3. Combinatorial Grid Search Evaluation
    print("\n[Phase 3/3] Evaluating 288 Multi-Asset Portfolio Permutations...")
    portfolio_results = []
    asset_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for tsl_name in TSL_REGIMES.keys():
        for ml_label, ml_thresh in ML_THRESHOLDS:
            # Filter trades by ML threshold for each asset
            filtered_trades = {}
            filtered_conc_m1_trades = {}
            for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
                raw_list = trade_cache[tsl_name][sym]
                if ml_thresh is None:
                    flt = raw_list
                else:
                    flt = [t for t in raw_list if t.get("ml_prob", 0) >= ml_thresh]
                filtered_trades[sym] = flt
                filtered_conc_m1_trades[sym] = filter_concurrency(flt)

            for conc_mode in CONCURRENCY_MODES:
                trades_to_use = filtered_conc_m1_trades if conc_mode == "Max 1 Position" else filtered_trades

                # Combined portfolio trades sorted by entry date
                blended_trades = sorted(
                    trades_to_use["XAUUSD"] + trades_to_use["BTCUSD"] + trades_to_use["ETHUSD"],
                    key=lambda x: x["entry_date"],
                )

                for seed in SEED_CAPITALS:
                    port_starting_balance = seed * 3.0  # 3 assets
                    for risk in RISK_LEVELS:
                        # Portfolio Simulation
                        port_metrics = simulate_equity_curve(
                            blended_trades,
                            starting_balance=port_starting_balance,
                            planned_risk_usd=risk,
                            years=5.0,
                        )

                        record = {
                            "seed_per_asset": seed,
                            "portfolio_seed": port_starting_balance,
                            "risk_usd": risk,
                            "risk_pct_of_seed": round((risk / seed) * 100.0, 2),
                            "tsl_regime": tsl_name,
                            "ml_gate": ml_label,
                            "concurrency": conc_mode,
                            **port_metrics,
                        }
                        portfolio_results.append(record)

                        # Individual Asset Evaluations
                        for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
                            asset_m = simulate_equity_curve(
                                trades_to_use[sym],
                                starting_balance=seed,
                                planned_risk_usd=risk,
                                years=5.0,
                            )
                            asset_results[sym].append({
                                "symbol": sym,
                                "seed": seed,
                                "risk_usd": risk,
                                "risk_pct_of_seed": round((risk / seed) * 100.0, 2),
                                "tsl_regime": tsl_name,
                                "ml_gate": ml_label,
                                "concurrency": conc_mode,
                                **asset_m,
                            })

    # Sort Portfolio Combinations by Efficiency Score
    ranked_portfolio = sorted(portfolio_results, key=lambda x: x["efficiency_score"], reverse=True)

    print("\n" + "=" * 165)
    print("🏆 TOP 15 RANKED MULTI-ASSET PORTFOLIO CONFIGURATIONS (BY EFFICIENCY SCORE)")
    print("=" * 165)
    header = (
        f"{'Rank':<4} | {'Seed ($)':<8} | {'Risk ($)':<8} | {'Risk %':<7} | {'TSL Buffer':<10} | {'ML Gate':<12} | "
        f"{'Concurrency':<14} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net P&L ($)':<12} | {'ROI %':<8} | "
        f"{'Max DD %':<8} | {'Calmar':<7} | {'Sharpe':<6} | {'Efficiency':<10}"
    )
    print(header)
    print("-" * 165)

    for r_idx, item in enumerate(ranked_portfolio[:15], 1):
        print(
            f"{r_idx:<4} | ${item['portfolio_seed']:>7,.0f} | ${item['risk_usd']:>6.2f} | {item['risk_pct_of_seed']:>5.1f}% | "
            f"{item['tsl_regime']:<10} | {item['ml_gate']:<12} | {item['concurrency']:<14} | {item['total_trades']:<6} | "
            f"{item['win_rate_pct']:>5.1f}% | {item['profit_factor']:>5.2f} | ${item['net_pnl_usd']:>+10,.2f} | "
            f"{item['roi_pct']:>+7.1f}% | {item['max_drawdown_pct']:>7.1f}% | {item['calmar_ratio']:>6.2f} | "
            f"{item['sharpe_ratio']:>6.2f} | {item['efficiency_score']:>10.2f}"
        )
    print("=" * 165)

    # Save to JSON
    out_dir = "/home/mcpuser/.gold_trading_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "grid_search_optimization_results.json")

    top_per_asset = {
        sym: sorted(asset_results[sym], key=lambda x: x["efficiency_score"], reverse=True)[:5]
        for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]
    }

    final_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "period": "5_years",
        "total_combinations_tested": len(portfolio_results),
        "execution_duration_seconds": round(time.time() - start_time, 2),
        "top_portfolio_ranked": ranked_portfolio[:30],
        "top_per_asset": top_per_asset,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    # Also mirror to UAT
    uat_dir = "/home/mcpuser/.gold_trading_data_uat"
    if os.path.exists(uat_dir):
        uat_path = os.path.join(uat_dir, "grid_search_optimization_results.json")
        with open(uat_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)

    print(f"\n✅ Grid Search Results Saved to {out_path} (Executed in {time.time() - start_time:.2f}s)")


if __name__ == "__main__":
    main()
