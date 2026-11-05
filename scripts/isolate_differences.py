import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')

from run_production_backtest import load_market_data, simulate_trade, DEFAULT_ASSET_COSTS
from test_realistic_backtest import simulate_trade_realistic

data = load_market_data('BTCUSD', period='2y', interval='1h')
n = data['n']
closes, highs, lows, opens = data['closes'], data['highs'], data['lows'], data['opens']
st, ema50, dc10, atr14 = data['st'], data['ema50'], data['dc10'], data['atr14']
costs = DEFAULT_ASSET_COSTS['BTCUSD']

def test_config(enable_same_bar_tp1, sequential_only, require_slope):
    trades = []
    last_exit = -1
    for i in range(200, n - 48):
        prev_i = i - 1
        if ema50[prev_i] is None or atr14[prev_i] is None or dc10['upper'][prev_i] is None or st['direction'][prev_i] is None:
            continue

        c_prev, trend_val, st_dir = closes[prev_i], ema50[prev_i], st["direction"][prev_i]
        prev_d_up, prev_d_low = dc10["upper"][prev_i], dc10["lower"][prev_i]
        h_curr, l_curr, o_curr = highs[i], lows[i], opens[i]

        macro_bull = c_prev > trend_val
        macro_bear = c_prev < trend_val

        # Slope
        if require_slope:
            slope = ((ema50[prev_i] - ema50[prev_i - 5]) / ema50[prev_i - 5]) * 100.0 if prev_i >= 5 else 0.0
            if macro_bull and slope < 0: macro_bull = False
            if macro_bear and slope > 0: macro_bear = False

        is_bull = (h_curr >= prev_d_up) and (st_dir == 1) and macro_bull
        is_bear = (l_curr <= prev_d_low) and (st_dir == -1) and macro_bear

        if not (is_bull or is_bear):
            continue

        if sequential_only and i <= last_exit:
            continue

        action = "BUY" if is_bull else "SELL"
        eff_entry = max(o_curr, prev_d_up) if action == "BUY" else min(o_curr, prev_d_low)
        sl_dist = max(1.5 * atr14[prev_i], 500.0)

        sim_fn = simulate_trade if enable_same_bar_tp1 else simulate_trade_realistic
        t_res = sim_fn(
            data=data, entry_idx=i, action=action, entry_price=eff_entry,
            sl_dist=sl_dist, scale_out_r=1.0, runner_trail_buffer=350.0,
            runner_t2_multiplier=3.0, be_offset=30.0, max_bars=48,
            spread=costs['spread'], slippage=costs['slippage'], swap_daily=costs['swap_daily']
        )
        trades.append(t_res)
        last_exit = t_res['exit_bar_idx']

    wins = [t for t in trades if t['pnl_r'] > 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    tot_r = sum(t['pnl_r'] for t in trades)
    gross_w = sum(t['pnl_r'] for t in wins)
    gross_l = abs(sum(t['pnl_r'] for t in trades if t['pnl_r'] <= 0))
    pf = gross_w / gross_l if gross_l > 0 else 0
    return len(trades), wr, pf, tot_r

print("=" * 95)
print("FACTOR ISOLATION ANALYSIS ON BITCOIN (BTCUSD)")
print("=" * 95)
experiments = [
    ("1. Old Setup (Same-Bar TP1 allowed, Overlapping allowed, No Slope)", True, False, False),
    ("2. Sequential Only (Same-Bar TP1 allowed, No Slope)", True, True, False),
    ("3. Sequential + Slope Guard (Same-Bar TP1 allowed)", True, True, True),
    ("4. Strict No-Same-Bar-TP1 + Sequential + Slope Guard (Our most recent test)", False, True, True),
]

for name, tp1, seq, slp in experiments:
    tr, wr, pf, r = test_config(tp1, seq, slp)
    print(f"{name:<75} | Trades={tr:<5} | WinRate={wr:>5.1f}% | PF={pf:>5.2f} | NetR={r:>+7.1f}R")
print("=" * 95)
