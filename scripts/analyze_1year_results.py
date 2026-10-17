#!/usr/bin/env python3
import json

with open("/home/mcpuser/.gold_trading_data/compounding_touch_1year_backtest_results.json") as f:
    data = json.load(f)

prod = data["PROD_TWO_STAGE_1000"]
trades = prod["trades"]

buy_trades = [t for t in trades if t["direction"] == "BUY"]
sell_trades = [t for t in trades if t["direction"] == "SELL"]

buy_wins = [t for t in buy_trades if t["outcome"] == "WIN"]
sell_wins = [t for t in sell_trades if t["outcome"] == "WIN"]

win_pnls = [t["pnl"] for t in trades if t["outcome"] == "WIN"]
loss_pnls = [t["pnl"] for t in trades if t["outcome"] == "LOSS"]

bars = [t["bars_held"] for t in trades]

exit_reasons = {}
for t in trades:
    r = t.get("exit_reason", "UNKNOWN")
    exit_reasons[r] = exit_reasons.get(r, 0) + 1

print("=" * 80)
print("📊 DETAILED METRICS FOR PRODUCTION DAEMON ($1,000 STARTING CAPITAL)")
print("=" * 80)
print(f"Total Trades: {len(trades)}")
print(f"Long (BUY):  {len(buy_trades)} trades | Wins: {len(buy_wins)} ({len(buy_wins)/len(buy_trades)*100:.1f}%) | Net P&L: ${sum(t['pnl'] for t in buy_trades):+,.2f}")
print(f"Short (SELL): {len(sell_trades)} trades | Wins: {len(sell_wins)} ({len(sell_wins)/len(sell_trades)*100:.1f}%) | Net P&L: ${sum(t['pnl'] for t in sell_trades):+,.2f}")
print(f"Average Win:  ${sum(win_pnls)/len(win_pnls):,.2f}")
print(f"Average Loss: ${sum(loss_pnls)/len(loss_pnls):,.2f}")
print(f"Payoff Ratio: {abs((sum(win_pnls)/len(win_pnls))/(sum(loss_pnls)/len(loss_pnls))):.2f}")
print(f"Average Duration: {sum(bars)/len(bars):.1f} hours ({sum(bars)/len(bars)/24:.1f} days)")
print(f"Max Consecutive Wins:   {prod['max_consec_wins']}")
print(f"Max Consecutive Losses: {prod['max_consec_losses']}")

print("\n🎯 Exit Reasons Breakdown:")
for r, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {r:<25}: {count:>3} trades ({count/len(trades)*100:>5.1f}%)")

print("\n🏆 Milestones Achieved:")
for m, info in sorted(prod["milestones"].items(), key=lambda x: int(x[0])):
    print(f"  - ${int(m):>5,}: Trade #{info['trade_no']:>3} on {info['date']} (Balance: ${info['balance']:>8,.2f}, Level {info['level']})")
