import sys
from collections import defaultdict

sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_sweet_spot_backtest import prepare_asset_signals, run_simulation

asset_contexts = {}
for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    asset_contexts[s] = prepare_asset_signals(s)

print("\n" + "="*85)
print("ASSET-BY-ASSET MONTHLY WITHDRAWAL CASH FLOW ($1,000.00 Starting Seed per Asset)")
print("="*85)

for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    for th, th_name in [(0.0, "No ML"), (0.5, "ML >= 0.50")]:
        trades = run_simulation(asset_contexts[s], max_open_positions=2, max_unhedged_positions=1, ml_threshold=th)
        trades.sort(key=lambda x: x['date'])

        monthly = defaultdict(list)
        for t in trades:
            monthly[t['date'][:7]].append(t)

        bal = 1000.0
        tot_w = 0.0
        min_b = 1000.0
        pos_m = 0

        for m in sorted(monthly.keys()):
            for t in monthly[m]:
                bal += t['pnl_r'] * 75.44
                min_b = min(min_b, bal)
            if bal > 1000.0:
                w = bal - 1000.0
                tot_w += w
                bal = 1000.0
                pos_m += 1

        avg_w = tot_w / len(monthly) if monthly else 0
        print(f"{s:<8} | {th_name:<11} | Withdrawn: ${tot_w:>9,.2f} | Avg/Mo: ${avg_w:>7,.2f} | Pos Months: {pos_m:>2}/{len(monthly)} | Lowest Dip: ${min_b:>7,.2f}")

