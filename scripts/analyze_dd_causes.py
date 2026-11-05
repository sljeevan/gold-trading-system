import sys
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_sweet_spot_backtest import prepare_asset_signals, run_simulation

asset_contexts = {}
for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    asset_contexts[s] = prepare_asset_signals(s)

def analyze_drawdown(th):
    all_trades = []
    for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
        t_list = run_simulation(asset_contexts[s], max_open_positions=2, max_unhedged_positions=1, ml_threshold=th)
        all_trades.extend(t_list)

    all_trades.sort(key=lambda x: x['date'])
    bal = 3000.0
    m_dd = 0.0
    cur_peak = bal
    cur_peak_date = all_trades[0]['date']
    max_dd_info = {}

    for t in all_trades:
        p = t['pnl_r'] * 75.44
        bal += p
        if bal > cur_peak:
            cur_peak = bal
            cur_peak_date = t['date']
        dd = cur_peak - bal
        if dd > m_dd:
            m_dd = dd
            max_dd_info = {
                'max_dd': dd,
                'peak_val': cur_peak,
                'peak_date': cur_peak_date,
                'trough_val': bal,
                'trough_date': t['date'],
            }

    print(f"\n=======================================================")
    print(f"DRAWDOWN FORENSIC ANALYSIS: THRESHOLD {th}")
    print(f"=======================================================")
    print(f"Max Drawdown: ${m_dd:,.2f}")
    print(f"Peak:   ${max_dd_info['peak_val']:,.2f} on {max_dd_info['peak_date']}")
    print(f"Trough: ${max_dd_info['trough_val']:,.2f} on {max_dd_info['trough_date']}")

    # Find trades in this window
    p_date = max_dd_info['peak_date']
    t_date = max_dd_info['trough_date']
    window_trades = [t for t in all_trades if p_date < t['date'] <= t_date]
    wins = [t for t in window_trades if t['pnl_r'] > 0]
    losses = [t for t in window_trades if t['pnl_r'] <= 0]
    print(f"Trades during DD: {len(window_trades)} (Wins: {len(wins)}, Losses: {len(losses)})")
    if window_trades:
        print(f"Win Rate during DD: {len(wins)/len(window_trades)*100:.1f}%")
        print(f"Net R during DD: {sum(t['pnl_r'] for t in window_trades):.1f}R")
        
    return all_trades, max_dd_info

t_no_ml, dd_no_ml = analyze_drawdown(0.00)
t_040, dd_040 = analyze_drawdown(0.40)
t_050, dd_050 = analyze_drawdown(0.50)

# Compare what trades occurred around dd_040 period in No ML vs ML 0.40
print("\n" + "="*70)
print(f"Comparing the ML 0.40 Drawdown Period ({dd_040['peak_date']} to {dd_040['trough_date']})")
print("="*70)

p_d = dd_040['peak_date']
t_d = dd_040['trough_date']

trades_in_period_no_ml = [t for t in t_no_ml if p_d < t['date'] <= t_d]
trades_in_period_040 = [t for t in t_040 if p_d < t['date'] <= t_d]

print(f"In that EXACT same date window:")
print(f"  No ML took:   {len(trades_in_period_no_ml)} trades | Net R = {sum(t['pnl_r'] for t in trades_in_period_no_ml):+.1f}R | Net P&L = ${sum(t['pnl_r'] for t in trades_in_period_no_ml)*75.44:+,.2f}")
print(f"  ML 0.40 took: {len(trades_in_period_040)} trades | Net R = {sum(t['pnl_r'] for t in trades_in_period_040):+.1f}R | Net P&L = ${sum(t['pnl_r'] for t in trades_in_period_040)*75.44:+,.2f}")

# Look at the trades filtered out by ML 0.40 in that window
no_ml_wins = [t for t in trades_in_period_no_ml if t['pnl_r'] > 0]
ml_wins = [t for t in trades_in_period_040 if t['pnl_r'] > 0]
print(f"  No ML had {len(no_ml_wins)} winning trades in that window, but ML 0.40 had only {len(ml_wins)} wins!")


print("\n" + "="*70)
print("Comparing the No ML Drawdown Period (2026-02-06 to 2026-02-22)")
print("="*70)

p_d2 = '2026-02-06 14:00'
t_d2 = '2026-02-22 11:00'

for label, t_list in [('No ML', t_no_ml), ('ML 0.40', t_040), ('ML 0.50', t_050)]:
    w_t = [t for t in t_list if p_d2 <= t['date'] <= t_d2]
    wins = [t for t in w_t if t['pnl_r'] > 0]
    r = sum(t['pnl_r'] for t in w_t)
    print(f"{label:<10}: trades={len(w_t):<3} | wins={len(wins):<2} | Net R={r:>+6.1f}R | PnL=${r*75.44:>+8,.2f}")

