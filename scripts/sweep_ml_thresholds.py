#!/usr/bin/env python3
"""
Sweeps ML probability score thresholds across the exact 2-year production dataset (Nov 2024 - Nov 2026).
Evaluates both individual assets (XAUUSD, BTCUSD, ETHUSD) and the blended portfolio.
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_production_backtest import (
    load_market_data, simulate_trade, compute_deflated_sharpe_ratio, DEFAULT_ASSET_COSTS
)
from gold_trader.core.indicators import calc_adx
from gold_trader.core.ml_filter import GoldMLPredictor, extract_candle_features

def extract_asset_candidates(
    symbol: str,
    sl_atr_mult: float,
    min_sl_dist: float,
    tsl_buffer: float,
    tp2_mult: float,
    be_offset: float,
    scale_out_r: float,
    session_filter: bool,
    require_slope_alignment: bool = True,
    require_wick_filter: bool = True,
    require_candle_confirmation: bool = True,
    adx_min_threshold: float = None,
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

    predictor = GoldMLPredictor(symbol=symbol)
    prod_model = predictor.model

    candidates = []
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
        atr_val = atr14[i]
        slope_val = ema50_slopes[i]
        adx_val = adx_vals[i] if (i < len(adx_vals) and adx_vals[i] is not None) else 20.0

        dt_str = dates[i]
        try:
            time_part = dt_str.split(" ")[1] if " " in dt_str else dt_str.split("T")[1]
            hour = int(time_part.split(":")[0])
        except Exception:
            hour = 12

        if session_filter and not (7 <= hour <= 19):
            continue

        macro_bull = c > trend_val
        macro_bear = c < trend_val

        is_raw_bull = (h >= prev_d_up or c >= prev_d_up) and st_dir == 1 and macro_bull
        is_raw_bear = (l <= prev_d_low or c <= prev_d_low) and st_dir == -1 and macro_bear

        if not (is_raw_bull or is_raw_bear):
            continue

        if adx_min_threshold is not None and adx_val < adx_min_threshold:
            continue

        if require_slope_alignment:
            if is_raw_bull and slope_val < 0.0:
                continue
            elif is_raw_bear and slope_val > 0.0:
                continue

        min_body = max(1.0, 0.15 * atr_val)
        action = None
        eff_entry = c

        if is_raw_bull:
            body = max(c - o, 0.0)
            upper_wick = h - max(c, o)
            wick_ok = (not require_wick_filter) or (upper_wick <= 1.8 * max(body, min_body))
            candle_ok = (not require_candle_confirmation) or (c >= o)

            if not (wick_ok and candle_ok):
                continue
            action = "BUY"
            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c

        elif is_raw_bear:
            body = max(o - c, 0.0)
            lower_wick = min(c, o) - l
            wick_ok = (not require_wick_filter) or (lower_wick <= 1.8 * max(body, min_body))
            candle_ok = (not require_candle_confirmation) or (c <= o)

            if not (wick_ok and candle_ok):
                continue
            action = "SELL"
            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c

        if not action:
            continue

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
            spread=spread,
            slippage=slippage,
            swap_daily=swap_daily,
        )

        dir_sign = 1.0 if action == "BUY" else -1.0
        feat_vec = np.append(X_all[i - 1], [dir_sign])
        if np.isnan(feat_vec).any():
            continue

        prod_prob = float(prod_model.predict_proba(feat_vec.reshape(1, -1))[0][1]) if prod_model is not None else 0.50

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
            "prod_prob": prod_prob,
        })
        last_trade_bar = i

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
        prob = c.get("prod_prob", 0.0)
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
    configs = [
        {"symbol": "XAUUSD", "sl_atr_mult": 1.5, "min_sl_dist": 8.00, "tsl_buffer": 4.50, "tp2_mult": 3.00, "be_offset": 0.50, "scale_out_r": 1.25, "session_filter": True, "adx_min_threshold": None},
        {"symbol": "BTCUSD", "sl_atr_mult": 1.5, "min_sl_dist": 500.00, "tsl_buffer": 350.00, "tp2_mult": 3.00, "be_offset": 30.00, "scale_out_r": 1.00, "session_filter": False, "adx_min_threshold": None},
        {"symbol": "ETHUSD", "sl_atr_mult": 1.5, "min_sl_dist": 20.00, "tsl_buffer": 10.00, "tp2_mult": 3.00, "be_offset": 2.00, "scale_out_r": 1.00, "session_filter": False, "adx_min_threshold": 22.0},
    ]

    all_candidates = {}
    for cfg in configs:
        sym = cfg["symbol"]
        print(f"Extracting 2-Year candidates for {sym}...")
        cands = extract_asset_candidates(
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
        print(f"  {sym}: {len(cands)} raw technical candidates generated.")

    # Sweep ML Thresholds from 0.00 (No ML) to 0.70 (Extreme Conviction)
    thresholds = [0.00, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

    sweep_results = []
    asset_results = {s: [] for s in all_candidates}

    for th in thresholds:
        # Asset individual
        combined_selected = []
        for sym, cands in all_candidates.items():
            res = simulate_selection(cands, ml_threshold=th)
            asset_results[sym].append(res)
            combined_selected.extend(res["selected_trades"])

        # Portfolio combined
        combined_selected.sort(key=lambda x: x["date"])
        tot_trades = len(combined_selected)
        if tot_trades > 0:
            wins = [t for t in combined_selected if t["pnl_r"] > 0]
            wr = (len(wins) / tot_trades) * 100.0
            tot_r = sum(t["pnl_r"] for t in combined_selected)
            tot_pnl = sum(t["pnl_r"] * 75.44 for t in combined_selected)

            # Portfolio DD
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

            sweep_results.append({
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

    # Output JSON
    out_file = "/home/mcpuser/.gold_trading_data/ml_threshold_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "portfolio_sweep": sweep_results,
            "asset_sweep": {
                s: [{k: v for k, v in r.items() if k != "selected_trades"} for r in res_list]
                for s, res_list in asset_results.items()
            }
        }, f, indent=2)

    print(f"\nSaved ML sweep results to {out_file}")

if __name__ == "__main__":
    main()
