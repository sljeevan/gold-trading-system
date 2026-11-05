import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_sweet_spot_backtest import prepare_asset_signals, run_simulation

asset_contexts = {}
for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
    asset_contexts[s] = prepare_asset_signals(s)

def simulate_withdrawals(ml_threshold, starting_baseline=3000.0, fixed_risk_usd=75.44):
    all_trades = []
    for s in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
        t_list = run_simulation(asset_contexts[s], max_open_positions=2, max_unhedged_positions=1, ml_threshold=ml_threshold)
        all_trades.extend(t_list)

    all_trades.sort(key=lambda x: x['date'])

    # Group trades by month (YYYY-MM)
    monthly_trades = defaultdict(list)
    for t in all_trades:
        m_str = t['date'][:7]
        monthly_trades[m_str].append(t)

    sorted_months = sorted(monthly_trades.keys())

    balance = starting_baseline
    total_withdrawn = 0.0
    monthly_records = []
    min_balance_seen = starting_baseline

    for m in sorted_months:
        trades_m = monthly_trades[m]
        start_bal = balance
        month_pnl = 0.0
        m_wins = 0
        m_losses = 0

        for t in trades_m:
            p = t['pnl_r'] * fixed_risk_usd
            balance += p
            month_pnl += p
            if p > 0:
                m_wins += 1
            else:
                m_losses += 1
            min_balance_seen = min(min_balance_seen, balance)

        # Monthly Withdrawal Policy:
        # If balance > starting_baseline at month end, withdraw the entire surplus above baseline!
        withdrawal = 0.0
        if balance > starting_baseline:
            withdrawal = balance - starting_baseline
            balance = starting_baseline
            total_withdrawn += withdrawal

        monthly_records.append({
            'month': m,
            'trades': len(trades_m),
            'win_rate': (m_wins / len(trades_m) * 100) if trades_m else 0,
            'month_pnl': month_pnl,
            'withdrawal': withdrawal,
            'end_balance': balance,
        })

    return sorted_months, monthly_records, total_withdrawn, min_balance_seen

print("Running Monthly Withdrawal Simulation...")
for th, label in [(0.00, "No ML (Pure Price Action)"), (0.40, "ML >= 0.40"), (0.50, "ML >= 0.50 (Live Gate)")]:
    months, records, tot_w, min_b = simulate_withdrawals(th)
    pos_months = len([r for r in records if r['withdrawal'] > 0])
    avg_w = tot_w / len(months)
    max_w = max(r['withdrawal'] for r in records)
    min_w = min(r['withdrawal'] for r in records)
    print(f"\n==========================================================================================")
    print(f"MONTHLY WITHDRAWAL CASH FLOW: {label} (Baseline Capital: $3,000.00 | Risk: $75.44)")
    print(f"==========================================================================================")
    print(f"Total Cash Withdrawn:  ${tot_w:,.2f}")
    print(f"Profitable Months:     {pos_months} / {len(months)} ({pos_months/len(months)*100:.1f}%)")
    print(f"Average Monthly Cash:  ${avg_w:,.2f} / month")
    print(f"Lowest Account Dip:    ${min_b:,.2f} (Max Drawdown from Baseline: ${3000.0 - min_b:,.2f})")
    print("-" * 90)
    print("{:<8} | {:<7} | {:<9} | {:<14} | {:<14} | {:<12}".format("Month", "Trades", "Win Rate", "Month P&L ($)", "Withdrawn ($)", "Account Bal"))
    print("-" * 90)
    for r in records:
        print("{:<8} | {:<7} | {:<8.1f}% | ${:>+12,.2f} | ${:>12,.2f} | ${:>10,.2f}".format(
            r['month'], r['trades'], r['win_rate'], r['month_pnl'], r['withdrawal'], r['end_balance']
        ))

