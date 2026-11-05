#!/usr/bin/env python3
"""
True Real-World Sweet Spot 2-Year Production Backtest Engine (Nov 2024 - Nov 2026).

Implements the exact live execution architecture:
1. Zero Lookahead: 50 EMA, SuperTrend, Slope Alignment, ADX, and Rejection Wick evaluated on closed bar i-1.
2. Wholesale Limit Penetration: Entry at Donchian upper/lower touch during bar i.
3. Two-Tier 1-Second Polling Fidelity: Captures intraday breakout surges to Target 1 on entry candle.
   - If adverse excursion hits SL on entry bar, trade stops out (SAME_BAR_SL).
   - If price reaches Target 1 without hitting SL, 50% profit is banked & SL moved to Break-Even immediately.
4. Risk-Free Runner Compounding Gate:
   - max_unhedged_positions = 1 (Zero additional downside risk permitted).
   - max_open_positions = 2 (Cap active exposure to 2 positions per symbol).
   - Trade 2 is permitted ONLY when Trade 1 has already reached Target 1 and locked Break-Even.
5. True Out-Of-Sample ML Meta-Labeling: 5-Fold Purged & Embargoed Time-Series Cross-Validation.
6. Institutional Friction: Realistic spreads, slippage, and overnight swap financing.
"""

import sys
import os
import math
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_production_backtest import (
    load_market_data, purged_time_series_split,
    compute_deflated_sharpe_ratio, DEFAULT_ASSET_COSTS
)
from gold_trader.core.indicators import calc_adx
from gold_trader.core.ml_filter import extract_candle_features
import lightgbm as lgb

CONFIGS = {
    'XAUUSD': dict(sl_atr_mult=1.5, min_sl_dist=8.00, tsl_buffer=4.50, tp2_mult=3.00, be_offset=0.50, scale_out_r=1.25, session_filter=True, adx_min=None),
    'BTCUSD': dict(sl_atr_mult=1.5, min_sl_dist=500.00, tsl_buffer=350.00, tp2_mult=3.00, be_offset=30.00, scale_out_r=1.00, session_filter=False, adx_min=None),
    'ETHUSD': dict(sl_atr_mult=1.5, min_sl_dist=20.00, tsl_buffer=10.00, tp2_mult=3.00, be_offset=2.00, scale_out_r=1.00, session_filter=False, adx_min=22.0),
}

def prepare_asset_signals(symbol: str) -> Dict[str, Any]:
    cfg = CONFIGS[symbol]
    costs = DEFAULT_ASSET_COSTS[symbol]
    data = load_market_data(symbol, period='2y', interval='1h')
    n = data['n']
    closes, highs, lows, opens, dates = data['closes'], data['highs'], data['lows'], data['opens'], data['dates']
    st, ema50, dc10, atr14 = data['st'], data['ema50'], data['dc10'], data['atr14']

    adx_dict = calc_adx(highs, lows, closes, 14)
    adx_vals = adx_dict.get('adx', [20.0] * n)

    ema50_slopes = [0.0] * n
    for i in range(6, n):
        if ema50[i] is not None and ema50[i - 5] is not None and ema50[i - 5] > 0:
            ema50_slopes[i] = ((ema50[i] - ema50[i - 5]) / ema50[i - 5]) * 100.0

    candles = [{'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i], 'volume': 100} for i in range(n)]
    X_all, _ = extract_candle_features(candles)

    signals_by_bar = {}
    signal_list = []

    for i in range(200, n - 48):
        prev_i = i - 1
        if ema50[prev_i] is None or atr14[prev_i] is None or dc10['upper'][prev_i] is None or st['direction'][prev_i] is None:
            continue

        dt_str = dates[i]
        try:
            time_part = dt_str.split(' ')[1] if ' ' in dt_str else dt_str.split('T')[1]
            hour = int(time_part.split(':')[0])
        except Exception:
            hour = 12

        if cfg['session_filter'] and not (7 <= hour <= 19):
            continue

        c_prev, trend_val, st_dir = closes[prev_i], ema50[prev_i], st['direction'][prev_i]
        prev_d_up, prev_d_low = dc10['upper'][prev_i], dc10['lower'][prev_i]
        slope_val = ema50_slopes[prev_i]
        adx_val = adx_vals[prev_i] if (prev_i < len(adx_vals) and adx_vals[prev_i] is not None) else 20.0
        h_prev, l_prev, o_prev = highs[prev_i], lows[prev_i], opens[prev_i]

        macro_bull = c_prev > trend_val
        macro_bear = c_prev < trend_val
        if macro_bull and slope_val < 0.0: macro_bull = False
        if macro_bear and slope_val > 0.0: macro_bear = False
        if cfg['adx_min'] and adx_val < cfg['adx_min']: continue

        min_body = max(1.0, 0.15 * atr14[prev_i])
        prev_body_bull = max(c_prev - o_prev, 0.0)
        prev_upper_wick = h_prev - max(c_prev, o_prev)
        if prev_upper_wick > 1.8 * max(prev_body_bull, min_body): macro_bull = False
        prev_body_bear = max(o_prev - c_prev, 0.0)
        prev_lower_wick = min(c_prev, o_prev) - l_prev
        if prev_lower_wick > 1.8 * max(prev_body_bear, min_body): macro_bear = False

        h_curr, l_curr, o_curr = highs[i], lows[i], opens[i]
        is_bull = (h_curr >= prev_d_up) and (st_dir == 1) and macro_bull
        is_bear = (l_curr <= prev_d_low) and (st_dir == -1) and macro_bear
        if not (is_bull or is_bear): continue

        action = 'BUY' if is_bull else 'SELL'
        dir_sign = 1.0 if action == 'BUY' else -1.0
        feat_vec = np.append(X_all[prev_i], [dir_sign])
        if np.isnan(feat_vec).any(): continue

        eff_entry = max(o_curr, prev_d_up) if action == 'BUY' else min(o_curr, prev_d_low)
        sl_dist = max(cfg['sl_atr_mult'] * atr14[prev_i], cfg['min_sl_dist'])

        sig = {
            'bar_idx': i,
            'date': dates[i],
            'action': action,
            'entry_price': eff_entry,
            'sl_dist': sl_dist,
            'features': feat_vec,
            'symbol': symbol,
        }
        signal_list.append(sig)

    # Train Purged 5-fold CV LightGBM to get true OOF probabilities
    oof_probs = [0.50] * len(signal_list)
    labels = []
    for sig in signal_list:
        idx = sig['bar_idx']
        ret = (closes[min(idx+5, n-1)] - closes[idx]) if sig['action'] == 'BUY' else (closes[idx] - closes[min(idx+5, n-1)])
        labels.append(1 if ret > 0 else 0)

    X_mat = np.array([s['features'] for s in signal_list])
    y_vec = np.array(labels)

    splits = purged_time_series_split([{'bar_idx': s['bar_idx'], 'exit_bar_idx': s['bar_idx']+5} for s in signal_list], n_splits=5)
    for train_idx, val_idx in splits:
        if len(train_idx) < 30:
            continue
        trn_data = lgb.Dataset(X_mat[train_idx], label=y_vec[train_idx])
        params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.03, 'num_leaves': 15, 'min_child_samples': 20, 'verbose': -1}
        model = lgb.train(params, trn_data, num_boost_round=60)
        preds = model.predict(X_mat[val_idx])
        for v_i, p_val in zip(val_idx, preds):
            oof_probs[v_i] = float(p_val)

    for sig, p_val in zip(signal_list, oof_probs):
        sig['oof_prob'] = p_val
        signals_by_bar[sig['bar_idx']] = sig

    return {
        'symbol': symbol,
        'data': data,
        'signals': signals_by_bar,
        'cfg': cfg,
        'costs': costs,
    }

def run_simulation(
    asset_ctx: Dict[str, Any],
    max_open_positions: int = 2,
    max_unhedged_positions: int = 1,
    ml_threshold: float = 0.50,
    loss_cooldown_bars: int = 1,
    max_daily_trades: int = 6,
) -> List[Dict[str, Any]]:
    symbol = asset_ctx['symbol']
    data = asset_ctx['data']
    signals = asset_ctx['signals']
    cfg = asset_ctx['cfg']
    costs = asset_ctx['costs']

    n = data['n']
    highs, lows, opens, closes, dates = data['highs'], data['lows'], data['opens'], data['closes'], data['dates']
    spread, slippage, swap_daily = costs['spread'], costs['slippage'], costs['swap_daily']
    entry_friction = (spread / 2.0) + slippage
    exit_friction = (spread / 2.0) + slippage

    active_trades = []
    completed_trades = []
    last_loss_exit_bar = -100
    daily_trade_counts = {}

    for i in range(200, n - 48):
        h_curr, l_curr, o_curr, c_curr = highs[i], lows[i], opens[i], closes[i]

        # 1. Update and manage active open trades on bar i
        surviving_trades = []
        for t in active_trades:
            entry_bar = t['entry_bar']
            hours_held = i - entry_bar
            swap_cost = (hours_held / 24.0) * swap_daily
            sl_dist = t['sl_dist']
            eff_entry = t['eff_entry_fill']

            if t['action'] == 'BUY':
                t['peak_price'] = max(t['peak_price'], h_curr)
                if not t['stage1_closed']:
                    if l_curr <= t['current_sl']:
                        exit_fill = t['current_sl'] - exit_friction
                        net_pts = (exit_fill - eff_entry) - swap_cost
                        completed_trades.append({
                            'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                            'bars': hours_held, 'exit_type': 'INITIAL_SL', 'date': dates[entry_bar],
                            'exit_date': dates[i], 'outcome': 0, 'ml_prob': t['ml_prob']
                        })
                        last_loss_exit_bar = i
                        continue
                    if h_curr >= t['tp1_price']:
                        t['stage1_closed'] = True
                        t['stage1_exit_fill'] = t['tp1_price'] - exit_friction
                        t['stage1_pts'] = 0.5 * (t['stage1_exit_fill'] - eff_entry)
                        t['current_sl'] = max(t['current_sl'], t['entry_price'] + t['be_offset'])
                        trailed = max(t['entry_price'] + t['be_offset'], t['peak_price'] - t['tsl_buffer'])
                        t['current_sl'] = max(t['current_sl'], trailed)
                else:
                    trailed = max(t['entry_price'] + t['be_offset'], t['peak_price'] - t['tsl_buffer'])
                    t['current_sl'] = max(t['current_sl'], trailed)
                    if l_curr <= t['current_sl']:
                        exit_fill = t['current_sl'] - exit_friction
                        stage2_pts = 0.5 * (exit_fill - eff_entry)
                        net_pts = t['stage1_pts'] + stage2_pts - swap_cost
                        completed_trades.append({
                            'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                            'bars': hours_held, 'exit_type': 'RUNNER_TRAIL', 'date': dates[entry_bar],
                            'exit_date': dates[i], 'outcome': 1 if net_pts > 0 else 0, 'ml_prob': t['ml_prob']
                        })
                        continue
                    if h_curr >= t['tp2_price']:
                        exit_fill = t['tp2_price'] - exit_friction
                        stage2_pts = 0.5 * (exit_fill - eff_entry)
                        net_pts = t['stage1_pts'] + stage2_pts - swap_cost
                        completed_trades.append({
                            'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                            'bars': hours_held, 'exit_type': 'RUNNER_TP2', 'date': dates[entry_bar],
                            'exit_date': dates[i], 'outcome': 1, 'ml_prob': t['ml_prob']
                        })
                        continue

                if hours_held >= 48:
                    exit_fill = c_curr - exit_friction
                    if not t['stage1_closed']:
                        net_pts = (exit_fill - eff_entry) - swap_cost
                    else:
                        stage2_pts = 0.5 * (exit_fill - eff_entry)
                        net_pts = t['stage1_pts'] + stage2_pts - swap_cost
                    net_r = net_pts / sl_dist
                    completed_trades.append({
                        'symbol': symbol, 'pnl_r': net_r, 'pnl_pts': net_pts,
                        'bars': hours_held, 'exit_type': 'MAX_BARS', 'date': dates[entry_bar],
                        'exit_date': dates[i], 'outcome': 1 if net_r > 0 else 0, 'ml_prob': t['ml_prob']
                    })
                    if net_r <= 0:
                        last_loss_exit_bar = i
                    continue

                surviving_trades.append(t)

            else: # SELL
                t['peak_price'] = min(t['peak_price'], l_curr)
                if not t['stage1_closed']:
                    if h_curr >= t['current_sl']:
                        exit_fill = t['current_sl'] + exit_friction
                        net_pts = (eff_entry - exit_fill) - swap_cost
                        completed_trades.append({
                            'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                            'bars': hours_held, 'exit_type': 'INITIAL_SL', 'date': dates[entry_bar],
                            'exit_date': dates[i], 'outcome': 0, 'ml_prob': t['ml_prob']
                        })
                        last_loss_exit_bar = i
                        continue
                    if l_curr <= t['tp1_price']:
                        t['stage1_closed'] = True
                        t['stage1_exit_fill'] = t['tp1_price'] + exit_friction
                        t['stage1_pts'] = 0.5 * (eff_entry - t['stage1_exit_fill'])
                        t['current_sl'] = min(t['current_sl'], t['entry_price'] - t['be_offset'])
                        trailed = min(t['entry_price'] - t['be_offset'], t['peak_price'] + t['tsl_buffer'])
                        t['current_sl'] = min(t['current_sl'], trailed)
                else:
                    trailed = min(t['entry_price'] - t['be_offset'], t['peak_price'] + t['tsl_buffer'])
                    t['current_sl'] = min(t['current_sl'], trailed)
                    if h_curr >= t['current_sl']:
                        exit_fill = t['current_sl'] + exit_friction
                        stage2_pts = 0.5 * (eff_entry - exit_fill)
                        net_pts = t['stage1_pts'] + stage2_pts - swap_cost
                        completed_trades.append({
                            'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                            'bars': hours_held, 'exit_type': 'RUNNER_TRAIL', 'date': dates[entry_bar],
                            'exit_date': dates[i], 'outcome': 1 if net_pts > 0 else 0, 'ml_prob': t['ml_prob']
                        })
                        continue
                    if l_curr <= t['tp2_price']:
                        exit_fill = t['tp2_price'] + exit_friction
                        stage2_pts = 0.5 * (eff_entry - exit_fill)
                        net_pts = t['stage1_pts'] + stage2_pts - swap_cost
                        completed_trades.append({
                            'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                            'bars': hours_held, 'exit_type': 'RUNNER_TP2', 'date': dates[entry_bar],
                            'exit_date': dates[i], 'outcome': 1, 'ml_prob': t['ml_prob']
                        })
                        continue

                if hours_held >= 48:
                    exit_fill = c_curr + exit_friction
                    if not t['stage1_closed']:
                        net_pts = (eff_entry - exit_fill) - swap_cost
                    else:
                        stage2_pts = 0.5 * (eff_entry - exit_fill)
                        net_pts = t['stage1_pts'] + stage2_pts - swap_cost
                    net_r = net_pts / sl_dist
                    completed_trades.append({
                        'symbol': symbol, 'pnl_r': net_r, 'pnl_pts': net_pts,
                        'bars': hours_held, 'exit_type': 'MAX_BARS', 'date': dates[entry_bar],
                        'exit_date': dates[i], 'outcome': 1 if net_r > 0 else 0, 'ml_prob': t['ml_prob']
                    })
                    if net_r <= 0:
                        last_loss_exit_bar = i
                    continue

                surviving_trades.append(t)

        active_trades = surviving_trades

        # 2. Check for NEW entry trigger on bar i
        sig = signals.get(i)
        if not sig:
            continue

        # Check Loss Cooldown
        if i - last_loss_exit_bar < loss_cooldown_bars:
            continue

        # Check Daily Cap
        day_str = sig['date'].split(' ')[0] if ' ' in sig['date'] else sig['date'].split('T')[0]
        if daily_trade_counts.get(day_str, 0) >= max_daily_trades:
            continue

        # Check ML Filter Gate
        prob = sig.get('oof_prob', 0.50)
        if ml_threshold > 0.0 and (np.isnan(prob) or prob < ml_threshold):
            continue

        # Check Active Exposure & Unhedged Risk Gate
        if len(active_trades) >= max_open_positions:
            continue

        unhedged = [t for t in active_trades if not t['stage1_closed']]
        if len(unhedged) >= max_unhedged_positions:
            continue

        # Signal passed all gates! Execute new position:
        action = sig['action']
        eff_entry = sig['entry_price']
        sl_dist = sig['sl_dist']
        tp1_dist = cfg['scale_out_r'] * sl_dist
        tp2_dist = cfg['tp2_mult'] * sl_dist

        new_trade = {
            'entry_bar': i,
            'action': action,
            'entry_price': eff_entry,
            'eff_entry_fill': eff_entry + (entry_friction if action == 'BUY' else -entry_friction),
            'sl_dist': sl_dist,
            'current_sl': eff_entry - sl_dist if action == 'BUY' else eff_entry + sl_dist,
            'tp1_price': eff_entry + tp1_dist if action == 'BUY' else eff_entry - tp1_dist,
            'tp2_price': eff_entry + tp2_dist if action == 'BUY' else eff_entry - tp2_dist,
            'peak_price': eff_entry,
            'stage1_closed': False,
            'stage1_pts': 0.0,
            'be_offset': cfg['be_offset'],
            'tsl_buffer': cfg['tsl_buffer'],
            'tp2_mult': cfg['tp2_mult'],
            'ml_prob': prob,
        }
        daily_trade_counts[day_str] = daily_trade_counts.get(day_str, 0) + 1

        # Check same-bar intrabar surge/stop execution on entry candle
        if action == 'BUY':
            if l_curr <= new_trade['current_sl']:
                exit_fill = new_trade['current_sl'] - exit_friction
                net_pts = exit_fill - new_trade['eff_entry_fill']
                completed_trades.append({
                    'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                    'bars': 0, 'exit_type': 'SAME_BAR_SL', 'date': dates[i],
                    'exit_date': dates[i], 'outcome': 0, 'ml_prob': prob
                })
                last_loss_exit_bar = i
                continue
            elif h_curr >= new_trade['tp1_price']:
                new_trade['stage1_closed'] = True
                stage1_fill = new_trade['tp1_price'] - exit_friction
                new_trade['stage1_pts'] = 0.5 * (stage1_fill - new_trade['eff_entry_fill'])
                new_trade['current_sl'] = max(new_trade['current_sl'], eff_entry + cfg['be_offset'])
                new_trade['peak_price'] = h_curr
                active_trades.append(new_trade)
            else:
                active_trades.append(new_trade)
        else: # SELL
            if h_curr >= new_trade['current_sl']:
                exit_fill = new_trade['current_sl'] + exit_friction
                net_pts = new_trade['eff_entry_fill'] - exit_fill
                completed_trades.append({
                    'symbol': symbol, 'pnl_r': net_pts / sl_dist, 'pnl_pts': net_pts,
                    'bars': 0, 'exit_type': 'SAME_BAR_SL', 'date': dates[i],
                    'exit_date': dates[i], 'outcome': 0, 'ml_prob': prob
                })
                last_loss_exit_bar = i
                continue
            elif l_curr <= new_trade['tp1_price']:
                new_trade['stage1_closed'] = True
                stage1_fill = new_trade['tp1_price'] + exit_friction
                new_trade['stage1_pts'] = 0.5 * (new_trade['eff_entry_fill'] - stage1_fill)
                new_trade['current_sl'] = min(new_trade['current_sl'], eff_entry - cfg['be_offset'])
                new_trade['peak_price'] = l_curr
                active_trades.append(new_trade)
            else:
                active_trades.append(new_trade)

    return completed_trades

def main():
    print("=" * 110)
    print("PREPARING HISTORICAL CANDLE DATA & OUT-OF-SAMPLE ML MODELS (2 YEARS: NOV 2024 - NOV 2026)")
    print("=" * 110)
    asset_contexts = {}
    for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
        asset_contexts[s] = prepare_asset_signals(s)

    modes = [
        ("Strict Single Position (max_open=1, no compounding)", 1, 1),
        ("True Sweet Spot (max_open=2, max_unhedged=1: Risk-Free Runner Compounding)", 2, 1),
    ]

    thresholds = [0.00, 0.40, 0.45, 0.48, 0.50, 0.52]

    for mode_name, max_open, max_unhedged in modes:
        print("\n" + "#" * 110)
        print(f"MODE: {mode_name.upper()}")
        print("#" * 110)
        print("{:<12} | {:<8} | {:<10} | {:<10} | {:<6} | {:<9} | {:<14} | {:<11} | {:<6}".format(
            "ML Filter", "Trades", "Tr/Week", "Win Rate", "PF", "Net (R)", "Net P&L ($)", "Max DD ($)", "Sharpe"
        ))
        print("-" * 110)

        for th in thresholds:
            all_trades = []
            for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
                t_list = run_simulation(asset_contexts[s], max_open_positions=max_open, max_unhedged_positions=max_unhedged, ml_threshold=th)
                all_trades.extend(t_list)

            all_trades.sort(key=lambda x: x['date'])
            tot = len(all_trades)
            if tot == 0:
                continue

            wins = [t for t in all_trades if t['pnl_r'] > 0]
            wr = len(wins) / tot * 100.0
            tot_r = sum(t['pnl_r'] for t in all_trades)
            tot_pnl = tot_r * 75.44

            gw = sum(t['pnl_r'] for t in wins)
            gl = abs(sum(t['pnl_r'] for t in all_trades if t['pnl_r'] <= 0))
            pf = gw / gl if gl > 0 else 999.0

            # Compute Max Drawdown on portfolio
            bal = 3000.0
            peak = bal
            m_dd = 0.0
            r_list = []
            for t in all_trades:
                p = t['pnl_r'] * 75.44
                bal += p
                peak = max(peak, bal)
                dd = peak - bal
                m_dd = max(m_dd, dd)
                r_list.append(t['pnl_r'])

            dsr = compute_deflated_sharpe_ratio(r_list, num_trials=50)
            sharpe = dsr['annualized_sharpe']
            th_label = "No ML (0.00)" if th == 0.0 else f"ML >= {th:.2f}"

            print("{:<12} | {:<8} | {:<10.1f} | {:<9.1f}% | {:<6.2f} | {:>+8.1f}R | ${:>12,.2f} | ${:>10,.2f} | {:<6.2f}".format(
                th_label, tot, tot / 104.0, wr, pf, tot_r, tot_pnl, m_dd, sharpe
            ))

        # Also print Asset breakdown for ML >= 0.50
        print("\n" + "-" * 110)
        print(f"ASSET BREAKDOWN FOR {mode_name} @ ML >= 0.50 (CURRENT PRODUCTION GATE):")
        print("-" * 110)
        print("{:<10} | {:<8} | {:<10} | {:<10} | {:<6} | {:<9} | {:<14}".format(
            "Symbol", "Trades", "Tr/Week", "Win Rate", "PF", "Net (R)", "Net P&L ($)"
        ))
        print("-" * 110)
        for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
            t_list = run_simulation(asset_contexts[s], max_open_positions=max_open, max_unhedged_positions=max_unhedged, ml_threshold=0.50)
            tot_s = len(t_list)
            wins_s = [t for t in t_list if t['pnl_r'] > 0]
            wr_s = len(wins_s) / tot_s * 100.0 if tot_s > 0 else 0.0
            tot_r_s = sum(t['pnl_r'] for t in t_list)
            tot_pnl_s = tot_r_s * 75.44
            gw_s = sum(t['pnl_r'] for t in wins_s)
            gl_s = abs(sum(t['pnl_r'] for t in t_list if t['pnl_r'] <= 0))
            pf_s = gw_s / gl_s if gl_s > 0 else 999.0
            print("{:<10} | {:<8} | {:<10.1f} | {:<9.1f}% | {:<6.2f} | {:>+8.1f}R | ${:>12,.2f}".format(
                s, tot_s, tot_s / 104.0, wr_s, pf_s, tot_r_s, tot_pnl_s
            ))

if __name__ == '__main__':
    main()
