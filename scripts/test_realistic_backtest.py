import sys
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_production_backtest import (
    load_market_data, purged_time_series_split,
    compute_deflated_sharpe_ratio, DEFAULT_ASSET_COSTS
)
from gold_trader.core.indicators import calc_adx
from gold_trader.core.ml_filter import extract_candle_features
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Any

def simulate_trade_realistic(
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

    entry_friction = (spread / 2.0) + slippage
    exit_friction = (spread / 2.0) + slippage

    if action == "BUY":
        eff_entry_fill = entry_price + entry_friction
        current_sl = entry_price - sl_dist
        tp1_price = entry_price + tp1_dist
        tp2_price = entry_price + tp2_dist
        peak_price = entry_price
        stage1_closed = False
        stage1_pts = 0.0

        # Bar 0 (Entry bar): Conservative check.
        # If low on entry bar touched SL, immediate stop out!
        if lows[entry_idx] <= current_sl:
            exit_fill = current_sl - exit_friction
            net_pts = exit_fill - eff_entry_fill
            net_r = net_pts / sl_dist
            return {
                "outcome": 0, "pnl_r": net_r, "pnl_pts": net_pts,
                "bars": 0, "exit_type": "SAME_BAR_SL", "exit_bar_idx": entry_idx
            }

        # Progress through subsequent bars
        for k in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
            c_h = highs[k]
            c_l = lows[k]
            peak_price = max(peak_price, c_h)
            hours_held = k - entry_idx
            swap_cost = (hours_held / 24.0) * swap_daily

            if not stage1_closed:
                # Check adverse stop excursion FIRST (conservative)
                if c_l <= current_sl:
                    exit_fill = current_sl - exit_friction
                    net_pts = (exit_fill - eff_entry_fill) - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
                        "bars": hours_held, "exit_type": "INITIAL_SL", "exit_bar_idx": k
                    }
                # Check Target 1
                if c_h >= tp1_price:
                    stage1_closed = True
                    stage1_exit_fill = tp1_price - exit_friction
                    stage1_pts = 0.5 * (stage1_exit_fill - eff_entry_fill)
                    current_sl = max(current_sl, entry_price + be_offset)
            else:
                # Trailing runner execution
                trailed_sl = max(entry_price + be_offset, peak_price - runner_trail_buffer)
                current_sl = max(current_sl, trailed_sl)

                if c_l <= current_sl:
                    stage2_exit_fill = current_sl - exit_friction
                    stage2_pts = 0.5 * (stage2_exit_fill - eff_entry_fill)
                    net_pts = stage1_pts + stage2_pts - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
                        "bars": hours_held, "exit_type": "RUNNER_TRAIL", "exit_bar_idx": k
                    }
                if c_h >= tp2_price:
                    stage2_exit_fill = tp2_price - exit_friction
                    stage2_pts = 0.5 * (stage2_exit_fill - eff_entry_fill)
                    net_pts = stage1_pts + stage2_pts - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
                        "bars": hours_held, "exit_type": "RUNNER_TP2", "exit_bar_idx": k
                    }

        # Time barrier expiration
        final_idx = min(entry_idx + max_bars, n - 1)
        final_c = closes[final_idx]
        hours_held = max_bars
        swap_cost = (hours_held / 24.0) * swap_daily
        if not stage1_closed:
            exit_fill = final_c - exit_friction
            net_pts = (exit_fill - eff_entry_fill) - swap_cost
        else:
            stage2_exit_fill = final_c - exit_friction
            stage2_pts = 0.5 * (stage2_exit_fill - eff_entry_fill)
            net_pts = stage1_pts + stage2_pts - swap_cost
        net_r = net_pts / sl_dist
        return {
            "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
            "bars": max_bars, "exit_type": "TIME_BARRIER", "exit_bar_idx": final_idx
        }

    else:  # SELL
        eff_entry_fill = entry_price - entry_friction
        current_sl = entry_price + sl_dist
        tp1_price = entry_price - tp1_dist
        tp2_price = entry_price - tp2_dist
        trough_price = entry_price
        stage1_closed = False
        stage1_pts = 0.0

        if highs[entry_idx] >= current_sl:
            exit_fill = current_sl + exit_friction
            net_pts = eff_entry_fill - exit_fill
            net_r = net_pts / sl_dist
            return {
                "outcome": 0, "pnl_r": net_r, "pnl_pts": net_pts,
                "bars": 0, "exit_type": "SAME_BAR_SL", "exit_bar_idx": entry_idx
            }

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
                        "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
                        "bars": hours_held, "exit_type": "INITIAL_SL", "exit_bar_idx": k
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
                        "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
                        "bars": hours_held, "exit_type": "RUNNER_TRAIL", "exit_bar_idx": k
                    }
                if c_l <= tp2_price:
                    stage2_exit_fill = tp2_price + exit_friction
                    stage2_pts = 0.5 * (eff_entry_fill - stage2_exit_fill)
                    net_pts = stage1_pts + stage2_pts - swap_cost
                    net_r = net_pts / sl_dist
                    return {
                        "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
                        "bars": hours_held, "exit_type": "RUNNER_TP2", "exit_bar_idx": k
                    }

        final_idx = min(entry_idx + max_bars, n - 1)
        final_c = closes[final_idx]
        hours_held = max_bars
        swap_cost = (hours_held / 24.0) * swap_daily
        if not stage1_closed:
            exit_fill = final_c + exit_friction
            net_pts = (eff_entry_fill - exit_fill) - swap_cost
        else:
            stage2_exit_fill = final_c + exit_friction
            stage2_pts = 0.5 * (eff_entry_fill - stage2_exit_fill)
            net_pts = stage1_pts + stage2_pts - swap_cost
        net_r = net_pts / sl_dist
        return {
            "outcome": 1 if net_r > 0 else 0, "pnl_r": net_r, "pnl_pts": net_pts,
            "bars": max_bars, "exit_type": "TIME_BARRIER", "exit_bar_idx": final_idx
        }

print("simulate_trade_realistic defined successfully")
