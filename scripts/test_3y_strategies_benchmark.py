#!/usr/bin/env python3
"""
Comparative 3-Year Quantitative Backtest (Oct 26, 2023 – Oct 26, 2026)
Testing High Win-Rate (>55%) Strategies Across Spot Gold, Bitcoin, and Ethereum.

Strategies Tested:
1. PROD_BASELINE: 1H Donchian 10 Touch + SuperTrend + 50 EMA + Two-Stage (50% @ 1.0R, BE offset, Medium TSL) + ML >= 0.44
2. HIGH_CONVICTION_ML: 1H Donchian 10 Touch + Two-Stage + Tight TSL + Strict ML Gate (P >= 0.52)
3. FAST_1R_TARGET: 1H Donchian 10 Touch + 100% Scale-Out @ 1.0R (No runner risk, locks 100% at TP1) + ML >= 0.48
4. VALUE_ZONE_PULLBACK: 1H EMA 9/21 Pullback in 50 EMA Trend + Two-Stage Execution + ML Gate
5. SMC_LIQUIDITY_SWEEP: Asian/Session Liquidity Sweep Reversal + Candle Confirmation + Two-Stage Execution

All simulations use:
- Realistic Spreads, Slippage, and Overnight Swaps
- Flat Risk: $75.44 on $1,000 Seed
- Out-of-Sample Purged Walk-Forward ML Meta-Labeling
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import lightgbm as lgb
import numpy as np

from scripts.run_production_backtest_5years import (
    ASSET_CONFIG,
    extract_features,
    load_5y_market_data,
    purged_time_series_split,
    simulate_trade,
)

DATA_DIR = os.path.expanduser("~/.gold_trading_data")
START_DATE = "2023-10-26 00:00"
END_DATE = "2026-10-26 23:59"

TSL_TIGHT = {
    "XAUUSD": 3.00,
    "BTCUSD": 200.00,
    "ETHUSD": 6.00,
}
TSL_MEDIUM = {
    "XAUUSD": 4.50,
    "BTCUSD": 350.00,
    "ETHUSD": 10.00,
}


def simulate_single_stage_trade(
    data: Dict[str, Any],
    entry_idx: int,
    action: str,
    entry_price: float,
    sl_dist: float,
    target_r: float = 1.0,
    max_bars: int = 48,
    spread: float = 0.0,
    slippage: float = 0.0,
    swap_daily: float = 0.0,
) -> Dict[str, Any]:
    """Simulates 100% exit at single target (e.g. 1.0R) or SL hit."""
    highs = data["highs"]
    lows = data["lows"]
    closes = data["closes"]
    dates = data["dates"]
    n = data["n"]

    entry_friction = (spread / 2.0) + slippage
    exit_friction = (spread / 2.0) + slippage

    if action == "BUY":
        eff_entry = entry_price + entry_friction
        sl_price = entry_price - sl_dist
        tp_price = entry_price + target_r * sl_dist

        # Entry bar check
        if lows[entry_idx] <= sl_price:
            exit_fill = sl_price - exit_friction
            net_pts = exit_fill - eff_entry
            return {"outcome": 0, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": 0, "exit_date": dates[entry_idx]}
        elif highs[entry_idx] >= tp_price:
            exit_fill = tp_price - exit_friction
            net_pts = exit_fill - eff_entry
            return {"outcome": 1, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": 0, "exit_date": dates[entry_idx]}

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            hours_held = k - entry_idx
            swap_cost = (hours_held / 24.0) * swap_daily
            # Adverse path first
            if lows[k] <= sl_price:
                exit_fill = sl_price - exit_friction
                net_pts = (exit_fill - eff_entry) - swap_cost
                return {"outcome": 0, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": hours_held, "exit_date": dates[k]}
            if highs[k] >= tp_price:
                exit_fill = tp_price - exit_friction
                net_pts = (exit_fill - eff_entry) - swap_cost
                return {"outcome": 1, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": hours_held, "exit_date": dates[k]}

        final_idx = min(entry_idx + max_bars, n - 1)
        hours_held = max_bars
        swap_cost = (hours_held / 24.0) * swap_daily
        exit_fill = closes[final_idx] - exit_friction
        net_pts = (exit_fill - eff_entry) - swap_cost
        return {"outcome": 1 if net_pts > 0 else 0, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": hours_held, "exit_date": dates[final_idx]}

    else:  # SELL
        eff_entry = entry_price - entry_friction
        sl_price = entry_price + sl_dist
        tp_price = entry_price - target_r * sl_dist

        if highs[entry_idx] >= sl_price:
            exit_fill = sl_price + exit_friction
            net_pts = eff_entry - exit_fill
            return {"outcome": 0, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": 0, "exit_date": dates[entry_idx]}
        elif lows[entry_idx] <= tp_price:
            exit_fill = tp_price + exit_friction
            net_pts = eff_entry - exit_fill
            return {"outcome": 1, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": 0, "exit_date": dates[entry_idx]}

        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            hours_held = k - entry_idx
            swap_cost = (hours_held / 24.0) * swap_daily
            if highs[k] >= sl_price:
                exit_fill = sl_price + exit_friction
                net_pts = (eff_entry - exit_fill) - swap_cost
                return {"outcome": 0, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": hours_held, "exit_date": dates[k]}
            if lows[k] <= tp_price:
                exit_fill = tp_price + exit_friction
                net_pts = (eff_entry - exit_fill) - swap_cost
                return {"outcome": 1, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": hours_held, "exit_date": dates[k]}

        final_idx = min(entry_idx + max_bars, n - 1)
        hours_held = max_bars
        swap_cost = (hours_held / 24.0) * swap_daily
        exit_fill = closes[final_idx] + exit_friction
        net_pts = (eff_entry - exit_fill) - swap_cost
        return {"outcome": 1 if net_pts > 0 else 0, "pnl_r": net_pts / sl_dist, "pnl_pts": net_pts, "bars": hours_held, "exit_date": dates[final_idx]}


def run_benchmark():
    symbols = ["XAUUSD", "BTCUSD", "ETHUSD"]
    results_by_asset = {}

    for symbol in symbols:
        print(f"\n=======================================================")
        print(f"Loading data & running full tests for {symbol}...")
        print(f"=======================================================")
        cfg = ASSET_CONFIG[symbol]
        data = load_5y_market_data(symbol)
        n = data["n"]
        closes, highs, lows, opens, dates = data["closes"], data["highs"], data["lows"], data["opens"], data["dates"]
        st, ema20, ema50, ema200, dc10, macd, atr14 = (
            data["st"],
            data["ema20"],
            data["ema50"],
            data["ema200"],
            data["dc10"],
            data["macd"],
            data["atr14"],
        )

        # -------------------------------------------------------------
        # System 1 & 2: Donchian 10 Touch (Two-Stage Execution)
        # -------------------------------------------------------------
        touch_trades_medium_tsl = []
        touch_trades_tight_tsl = []
        fast_1r_trades = []

        last_trade_bar = -100

        for i in range(200, n - 48):
            dt_str = dates[i]
            if not (START_DATE <= dt_str <= END_DATE):
                continue
            if ema50[i] is None or atr14[i] is None or dc10["upper"][i - 1] is None or st["direction"][i] is None:
                continue
            if i - last_trade_bar < 1:
                continue

            c, o, h, l = closes[i], opens[i], highs[i], lows[i]
            prev_d_up = dc10["upper"][i - 1]
            prev_d_low = dc10["lower"][i - 1]
            st_dir = st["direction"][i]
            trend_val = ema50[i]

            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])

            if symbol == "XAUUSD" and not (7 <= hour <= 19):
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
            sl_dist = max(cfg["sl_atr_mult"] * atr_val, cfg["min_sl"])
            feat = extract_features(data, i - 1, action)
            if feat is None:
                continue

            # Medium TSL (Prod Baseline)
            t_med = simulate_trade(
                data=data,
                entry_idx=i,
                action=action,
                entry_price=eff_entry,
                sl_dist=sl_dist,
                scale_out_r=1.0,
                runner_trail_buffer=TSL_MEDIUM[symbol],
                runner_t2_multiplier=3.0,
                be_offset=cfg["be_offset"],
                max_bars=48,
                spread=cfg["spread"],
                slippage=cfg["slippage"],
                swap_daily=cfg["swap_daily"],
            )
            touch_trades_medium_tsl.append({
                "bar_idx": i,
                "entry_date": dt_str,
                "outcome": t_med["outcome"],
                "pnl_r": t_med["pnl_r"],
                "features": feat,
            })

            # Tight TSL (High Conviction ML)
            t_tight = simulate_trade(
                data=data,
                entry_idx=i,
                action=action,
                entry_price=eff_entry,
                sl_dist=sl_dist,
                scale_out_r=1.0,
                runner_trail_buffer=TSL_TIGHT[symbol],
                runner_t2_multiplier=3.0,
                be_offset=cfg["be_offset"],
                max_bars=48,
                spread=cfg["spread"],
                slippage=cfg["slippage"],
                swap_daily=cfg["swap_daily"],
            )
            touch_trades_tight_tsl.append({
                "bar_idx": i,
                "entry_date": dt_str,
                "outcome": t_tight["outcome"],
                "pnl_r": t_tight["pnl_r"],
                "features": feat,
            })

            # Fast 1.0R Target (100% exit at 1.0R)
            t_fast = simulate_single_stage_trade(
                data=data,
                entry_idx=i,
                action=action,
                entry_price=eff_entry,
                sl_dist=sl_dist,
                target_r=1.0,
                max_bars=48,
                spread=cfg["spread"],
                slippage=cfg["slippage"],
                swap_daily=cfg["swap_daily"],
            )
            fast_1r_trades.append({
                "bar_idx": i,
                "entry_date": dt_str,
                "outcome": t_fast["outcome"],
                "pnl_r": t_fast["pnl_r"],
                "features": feat,
            })

            last_trade_bar = i

        # -------------------------------------------------------------
        # System 4: Value Zone Pullback (EMA 9/21 Pullback)
        # -------------------------------------------------------------
        value_zone_trades = []
        last_vz_bar = -100

        for i in range(200, n - 48):
            dt_str = dates[i]
            if not (START_DATE <= dt_str <= END_DATE):
                continue
            if ema20[i] is None or ema50[i] is None or atr14[i] is None:
                continue
            if i - last_vz_bar < 1:
                continue

            c, o, h, l = closes[i], opens[i], highs[i], lows[i]
            prev_c, prev_o = closes[i - 1], opens[i - 1]
            e20, e50 = ema20[i], ema50[i]

            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])
            if symbol == "XAUUSD" and not (7 <= hour <= 19):
                continue

            action = None
            eff_entry = c

            # Uptrend + Pullback to EMA 20
            if c > e50 and e20 > e50 and l <= e20 and c > o and c > prev_c:
                action = "BUY"
            elif c < e50 and e20 < e50 and h >= e20 and c < o and c < prev_c:
                action = "SELL"

            if not action:
                continue

            atr_val = atr14[i]
            sl_dist = max(cfg["sl_atr_mult"] * atr_val, cfg["min_sl"])
            feat = extract_features(data, i - 1, action)
            if feat is None:
                continue

            t_vz = simulate_trade(
                data=data,
                entry_idx=i,
                action=action,
                entry_price=eff_entry,
                sl_dist=sl_dist,
                scale_out_r=1.0,
                runner_trail_buffer=TSL_MEDIUM[symbol],
                runner_t2_multiplier=3.0,
                be_offset=cfg["be_offset"],
                max_bars=48,
                spread=cfg["spread"],
                slippage=cfg["slippage"],
                swap_daily=cfg["swap_daily"],
            )
            value_zone_trades.append({
                "bar_idx": i,
                "entry_date": dt_str,
                "outcome": t_vz["outcome"],
                "pnl_r": t_vz["pnl_r"],
                "features": feat,
            })
            last_vz_bar = i

        # -------------------------------------------------------------
        # System 5: SMC Liquidity Sweep & Reversal
        # -------------------------------------------------------------
        smc_trades = []
        last_smc_bar = -100

        for i in range(200, n - 48):
            dt_str = dates[i]
            if not (START_DATE <= dt_str <= END_DATE):
                continue
            if ema50[i] is None or atr14[i] is None:
                continue
            if i - last_smc_bar < 1:
                continue

            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])
            if symbol == "XAUUSD" and not (7 <= hour <= 19):
                continue

            # 20-bar swing high/low
            swing_hi = max(highs[max(0, i - 21) : i - 1])
            swing_lo = min(lows[max(0, i - 21) : i - 1])
            c, o, h, l = closes[i], opens[i], highs[i], lows[i]
            e50 = ema50[i]

            action = None
            eff_entry = c

            # Bearish Judas Sweep: high penetrated 20-bar high, but candle closed bearish below swing high
            if h > swing_hi and c < swing_hi and c < o and c < e50:
                action = "SELL"
            # Bullish Judas Sweep: low penetrated 20-bar low, but candle closed bullish above swing low
            elif l < swing_lo and c > swing_lo and c > o and c > e50:
                action = "BUY"

            if not action:
                continue

            atr_val = atr14[i]
            sl_dist = max(cfg["sl_atr_mult"] * atr_val, cfg["min_sl"])
            feat = extract_features(data, i - 1, action)
            if feat is None:
                continue

            t_smc = simulate_trade(
                data=data,
                entry_idx=i,
                action=action,
                entry_price=eff_entry,
                sl_dist=sl_dist,
                scale_out_r=1.0,
                runner_trail_buffer=TSL_MEDIUM[symbol],
                runner_t2_multiplier=3.0,
                be_offset=cfg["be_offset"],
                max_bars=48,
                spread=cfg["spread"],
                slippage=cfg["slippage"],
                swap_daily=cfg["swap_daily"],
            )
            smc_trades.append({
                "bar_idx": i,
                "entry_date": dt_str,
                "outcome": t_smc["outcome"],
                "pnl_r": t_smc["pnl_r"],
                "features": feat,
            })
            last_smc_bar = i

        # -------------------------------------------------------------
        # ML Training via Purged 5-Fold Walk-Forward Cross-Validation
        # -------------------------------------------------------------
        def apply_purged_ml(trade_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if len(trade_list) < 50:
                for t in trade_list:
                    t["ml_prob"] = 0.50
                return trade_list

            X = np.array([t["features"] for t in trade_list])
            y = np.array([t["outcome"] for t in trade_list])
            probs = np.full(len(trade_list), np.nan)

            splits = purged_time_series_split(trade_list, n_splits=5)
            clf = lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=15,
                subsample=0.80,
                colsample_bytree=0.80,
                random_state=42,
                verbose=-1,
            )
            for fold, (tr_idx, val_idx) in enumerate(splits):
                if len(tr_idx) == 0:
                    continue
                clf.fit(X[tr_idx], y[tr_idx])
                probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

            for idx, p in enumerate(probs):
                trade_list[idx]["ml_prob"] = 0.50 if np.isnan(p) else float(p)
            return trade_list

        touch_trades_medium_tsl = apply_purged_ml(touch_trades_medium_tsl)
        touch_trades_tight_tsl = apply_purged_ml(touch_trades_tight_tsl)
        fast_1r_trades = apply_purged_ml(fast_1r_trades)
        value_zone_trades = apply_purged_ml(value_zone_trades)
        smc_trades = apply_purged_ml(smc_trades)

        # -------------------------------------------------------------
        # Evaluate Performance Metrics ($75.44 risk on $1,000 seed)
        # -------------------------------------------------------------
        def evaluate(trade_list: List[Dict[str, Any]], min_prob: Optional[float] = None) -> Dict[str, Any]:
            sel = trade_list
            if min_prob is not None:
                sel = [t for t in sel if t.get("ml_prob", 0.50) >= min_prob]

            total = len(sel)
            if total == 0:
                return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "payoff": 0.0, "net_pnl": 0.0, "max_dd": 0.0}

            wins = [t for t in sel if t["pnl_r"] > 0]
            losses = [t for t in sel if t["pnl_r"] <= 0]
            wr = round(len(wins) / total * 100.0, 2)

            win_r = [t["pnl_r"] for t in wins]
            loss_r = [abs(t["pnl_r"]) for t in losses]
            sum_win_r = sum(win_r)
            sum_loss_r = sum(loss_r)

            pf = round(sum_win_r / sum_loss_r, 2) if sum_loss_r > 0 else 99.0
            avg_win_r = round(np.mean(win_r), 2) if win_r else 0.0
            avg_loss_r = round(np.mean(loss_r), 2) if loss_r else 1.0
            payoff = round(avg_win_r / avg_loss_r, 2) if avg_loss_r > 0 else 0.0

            # Dollar metrics with $75.44 fixed risk
            fixed_risk = 75.44
            equity = 1000.0
            peak = 1000.0
            max_dd = 0.0

            for t in sel:
                pnl_usd = t["pnl_r"] * fixed_risk
                equity += pnl_usd
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd

            net_pnl = round(equity - 1000.0, 2)

            return {
                "trades": total,
                "win_rate": wr,
                "pf": pf,
                "payoff": payoff,
                "avg_win_r": avg_win_r,
                "avg_loss_r": avg_loss_r,
                "net_pnl": net_pnl,
                "max_dd": round(max_dd, 2),
                "trades_per_mo": round(total / 36.0, 1),
            }

        candidate_evals = {
            "1. PROD_BASELINE (Two-Stage + ML >= 0.44)": evaluate(touch_trades_medium_tsl, min_prob=0.44),
            "2. HIGH_CONVICTION_ML (Tight TSL + ML >= 0.52)": evaluate(touch_trades_tight_tsl, min_prob=0.52),
            "3. HIGH_CONVICTION_ML (Tight TSL + ML >= 0.55)": evaluate(touch_trades_tight_tsl, min_prob=0.55),
            "4. FAST_1R_TARGET (100% Take Profit @ 1.0R)": evaluate(fast_1r_trades, min_prob=0.48),
            "5. VALUE_ZONE_PULLBACK (Two-Stage + ML >= 0.44)": evaluate(value_zone_trades, min_prob=0.44),
            "6. SMC_LIQUIDITY_SWEEP (Two-Stage + ML >= 0.44)": evaluate(smc_trades, min_prob=0.44),
        }
        results_by_asset[symbol] = candidate_evals

    # Print Summary Tables
    for sym, evals in results_by_asset.items():
        print("\n" + "=" * 95)
        print(f"🏆 {sym} STRATEGY BENCHMARK LEADERBOARD (OCT 2023 – OCT 2026)")
        print("=" * 95)
        print(f"{'Strategy Architecture':<46} | {'Trades':<6} | {'Win Rate':<9} | {'PF':<5} | {'Payoff':<6} | {'Net P&L ($)':<12} | {'Max DD ($)':<10}")
        print("-" * 95)
        for name, m in evals.items():
            badge = "🔥" if m["win_rate"] >= 55.0 else ("⭐" if m["win_rate"] >= 52.0 else "  ")
            print(f"{badge}{name:<44} | {m['trades']:<6} | {m['win_rate']:>6.1f}%  | {m['pf']:>4.2f} | {m['payoff']:>4.2f}R | ${m['net_pnl']:>10,.2f} | -${m['max_dd']:>8,.2f}")

    # Save to JSON
    out_file = os.path.join(DATA_DIR, "rigorous_high_winrate_3years_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results_by_asset, f, indent=2)
    print(f"\n✅ Rigorous benchmark results saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
