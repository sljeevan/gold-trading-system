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

# 1. Look at February 2026 trades specifically
feb_trades = [t for t in all_trades if t['date'].startswith('2026-02')]

print(f"Total trades in Feb 2026: {len(feb_trades)}")

# Divide Feb 2026 into 3 segments:
# Part 1: Feb 1 - Feb 6 (Before the DD peak)
# Part 2: Feb 6 14:00 - Feb 22 11:00 (The drawdown dip)
# Part 3: Feb 22 11:00 - Feb 28 (The recovery)

part1 = [t for t in feb_trades if t['date'] < '2026-02-06 14:00']
part2 = [t for t in feb_trades if '2026-02-06 14:00' <= t['date'] <= '2026-02-22 11:00']
part3 = [t for t in feb_trades if t['date'] > '2026-02-22 11:00']

r1 = sum(t['pnl_r'] for t in part1)
r2 = sum(t['pnl_r'] for t in part2)
r3 = sum(t['pnl_r'] for t in part3)

print("\n--- February 2026 Intramonth Breakdown ---")
print(f"1. Feb 01 to Feb 06:  {len(part1)} trades | Net R = {r1:+.1f}R | PnL = ${r1*75.44:+,.2f}  <-- Huge surge at start of month!")
print(f"2. Feb 06 to Feb 22:  {len(part2)} trades | Net R = {r2:+.1f}R | PnL = ${r2*75.44:+,.2f}  <-- The $1,618 Peak-to-Trough Drawdown!")
print(f"3. Feb 22 to Feb 28:  {len(part3)} trades | Net R = {r3:+.1f}R | PnL = ${r3*75.44:+,.2f}  <-- Massive recovery at end of month!")
print("-" * 50)
print(f"Calendar Month Total: {len(feb_trades)} trades | Net R = {r1+r2+r3:+.1f}R | PnL = ${(r1+r2+r3)*75.44:+,.2f} (PROFITABLE!)")

# 2. Check the balance tracking under Monthly Withdrawal:
bal = 3000.0 # Started Feb 1 at 3000
min_feb_bal = 3000.0
feb_track = []
for t in feb_trades:
    bal += t['pnl_r'] * 75.44
    min_feb_bal = min(min_feb_bal, bal)
    feb_track.append((t['date'], t['pnl_r']*75.44, bal))

print(f"\nIn the Monthly Withdrawal account (starting Feb 1 at $3,000):")
print(f"Peak balance in early Feb: ${max(x[2] for x in feb_track):,.2f}")
print(f"Lowest balance dip in Feb: ${min_feb_bal:,.2f}")
print(f"Ending balance on Feb 28:  ${bal:,.2f} (Withdrew ${bal - 3000:,.2f})")

# 3. Where did the $2,228.33 Lowest Dip actually occur?
print("\n--- Where did the Lowest Dip ($2,228.33) happen across the 2 years? ---")
bal_all = 3000.0
min_overall = 3000.0
min_date = None
monthly = defaultdict(list)
for t in all_trades:
    monthly[t['date'][:7]].append(t)

for m in sorted(monthly.keys()):
    for t in monthly[m]:
        bal_all += t['pnl_r'] * 75.44
        if bal_all < min_overall:
            min_overall = bal_all
            min_date = t['date']
    if bal_all > 3000.0:
        bal_all = 3000.0

print(f"Overall Lowest Account Dip: ${min_overall:,.2f} occurred on: {min_date}")

