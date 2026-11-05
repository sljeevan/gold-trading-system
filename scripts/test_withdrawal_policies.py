import sys
from collections import defaultdict

sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_sweet_spot_backtest import prepare_asset_signals, run_simulation

asset_contexts = {}
for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    asset_contexts[s] = prepare_asset_signals(s)

all_trades = []
for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    all_trades.extend(run_simulation(asset_contexts[s], max_open_positions=2, max_unhedged_positions=1, ml_threshold=0.0))

all_trades.sort(key=lambda x: x['date'])

monthly = defaultdict(list)
for t in all_trades:
    monthly[t['date'][:7]].append(t)

print("Policy Comparison (No ML, 2 Years):")
for pct in [1.0, 0.8, 0.7]:
    bal = 3000.0
    tot_w = 0.0
    min_b = 3000.0
    for m in sorted(monthly.keys()):
        for t in monthly[m]:
            bal += t['pnl_r'] * 75.44
            min_b = min(min_b, bal)
        if bal > 3000.0:
            surplus = bal - 3000.0
            w = surplus * pct
            tot_w += w
            bal -= w
    print(f"Withdraw {int(pct*100)}% Profit: Total Withdrawn = ${tot_w:>10,.2f} | Final Balance = ${bal:>10,.2f} | Lowest Dip = ${min_b:>10,.2f}")


all_trades_50 = []
for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    all_trades_50.extend(run_simulation(asset_contexts[s], max_open_positions=2, max_unhedged_positions=1, ml_threshold=0.5))

all_trades_50.sort(key=lambda x: x['date'])

monthly_50 = defaultdict(list)
for t in all_trades_50:
    monthly_50[t['date'][:7]].append(t)

print("\nPolicy Comparison (ML >= 0.50, 2 Years):")
for pct in [1.0, 0.8, 0.7]:
    bal = 3000.0
    tot_w = 0.0
    min_b = 3000.0
    for m in sorted(monthly_50.keys()):
        for t in monthly_50[m]:
            bal += t['pnl_r'] * 75.44
            min_b = min(min_b, bal)
        if bal > 3000.0:
            surplus = bal - 3000.0
            w = surplus * pct
            tot_w += w
            bal -= w
    print(f"Withdraw {int(pct*100)}% Profit: Total Withdrawn = ${tot_w:>10,.2f} | Final Balance = ${bal:>10,.2f} | Lowest Dip = ${min_b:>10,.2f}")

