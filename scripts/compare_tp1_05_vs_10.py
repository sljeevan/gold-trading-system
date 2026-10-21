"""
Comparative Study: Initial Profit Capture at 0.5R vs 1.0R with Trailing Stop Loss.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_production_backtest import run_backtest_for_asset
from scripts.analyze_monthly_breakout import analyze_monthly_trades

configs = [
    ("Spot Gold (XAUUSD)", "XAUUSD", 1.5, 8.0, 4.5, 3.0, 0.5, True),
    ("Bitcoin (BTCUSD)", "BTCUSD", 1.5, 500.0, 350.0, 3.0, 30.0, False),
    ("Ethereum (ETHUSD)", "ETHUSD", 1.5, 20.0, 10.0, 3.0, 2.0, True),
]

print("=" * 90)
print("🔍 COMPARATIVE AUDIT: INITIAL PROFIT CAPTURE @ 0.50R VS 1.00R (TRAILING RUNNER)")
print("=" * 90)

all_results = {}

for label, sym, mult, min_sl, tsl, tp2, be, sess in configs:
    print(f"\n[+] Analyzing {label}...")
    # Baseline: TP1 @ 1.0R
    res_10 = run_backtest_for_asset(sym, mult, min_sl, tsl, tp2, be, sess, scale_out_r=1.0)
    mom_10 = analyze_monthly_trades(sym, res_10["all_trades"], ml_threshold=0.44, start_month="2025-10", end_month="2026-09")
    c12_10 = mom_10["completed_12m"]
    
    # Variant: TP1 @ 0.5R
    res_05 = run_backtest_for_asset(sym, mult, min_sl, tsl, tp2, be, sess, scale_out_r=0.5)
    mom_05 = analyze_monthly_trades(sym, res_05["all_trades"], ml_threshold=0.44, start_month="2025-10", end_month="2026-09")
    c12_05 = mom_05["completed_12m"]
    
    all_results[sym] = {
        "label": label,
        "base_10": c12_10,
        "variant_05": c12_05,
        "records_10": mom_10["monthly_records"],
        "records_05": mom_05["monthly_records"],
    }
    
    print(f"  --- Baseline (TP1 @ 1.0R) ---")
    print(f"    Trades: {c12_10['trades']} | WR: {c12_10['win_rate_pct']}% | Net R: {c12_10['net_r']}R | P&L: ${c12_10['net_pnl_usd']:.2f} | PF: {c12_10['profit_factor']:.2f} | Winning Months: {c12_10['profitable_months']}/12")
    print(f"  --- Variant (TP1 @ 0.5R) ---")
    print(f"    Trades: {c12_05['trades']} | WR: {c12_05['win_rate_pct']}% | Net R: {c12_05['net_r']}R | P&L: ${c12_05['net_pnl_usd']:.2f} | PF: {c12_05['profit_factor']:.2f} | Winning Months: {c12_05['profitable_months']}/12")
    
    delta_r = c12_05['net_r'] - c12_10['net_r']
    delta_pnl = c12_05['net_pnl_usd'] - c12_10['net_pnl_usd']
    delta_wr = c12_05['win_rate_pct'] - c12_10['win_rate_pct']
    print(f"    ==> DELTA: Win Rate: {delta_wr:+.1f}% | Net R: {delta_r:+.1f}R | Net P&L: ${delta_pnl:+.2f}")

# Compute combined portfolio comparison for completed 12 months
print("\n" + "=" * 90)
print("💎 BLENDED PORTFOLIO 12-MONTH PERFORMANCE COMPARISON ($3,000 SEED)")
print("=" * 90)

for mode, key in [("Baseline (TP1 @ 1.00R)", "base_10"), ("Variant (TP1 @ 0.50R)", "variant_05")]:
    tot_trades = sum(all_results[s][key]["trades"] for s in all_results)
    tot_wins = sum(all_results[s][key]["wins"] for s in all_results)
    tot_losses = sum(all_results[s][key]["losses"] for s in all_results)
    wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    tot_r = sum(all_results[s][key]["net_r"] for s in all_results)
    tot_pnl = sum(all_results[s][key]["net_pnl_usd"] for s in all_results)
    
    # Calculate portfolio monthly records
    by_m = {}
    for s in all_results:
        rec_key = "records_10" if key == "base_10" else "records_05"
        for r in all_results[s][rec_key]:
            m = r["month"]
            if m < "2026-10":
                by_m[m] = by_m.get(m, 0.0) + r["net_pnl_usd"]
                
    win_m = sum(1 for m, p in by_m.items() if p > 0)
    print(f"{mode}:")
    print(f"  Total Trades: {tot_trades} | Win Rate: {wr:.1f}% | Net R: {tot_r:+.1f}R | Net P&L: ${tot_pnl:,.2f} (+{tot_pnl/3000.0*100:.1f}%) | Winning Months: {win_m}/12")

# Print month-by-month table comparison for Gold
print("\n" + "=" * 90)
print("📅 SPOT GOLD: MONTH-BY-MONTH COMPARISON (TP1 @ 0.5R VS 1.0R)")
print("=" * 90)
g_10 = {r["month"]: r for r in all_results["XAUUSD"]["records_10"]}
g_05 = {r["month"]: r for r in all_results["XAUUSD"]["records_05"]}
all_m = sorted([m for m in g_10 if m < "2026-10"])

print(f"{'Month':<10} | {'1.0R Trades':<11} | {'1.0R WR':<8} | {'1.0R Net R':<10} | {'1.0R P&L':<12} || {'0.5R Trades':<11} | {'0.5R WR':<8} | {'0.5R Net R':<10} | {'0.5R P&L':<12}")
print("-" * 105)
for m in all_m:
    r1 = g_10[m]
    r0 = g_05[m]
    p1 = f"${r1['net_pnl_usd']:.2f}"
    p0 = f"${r0['net_pnl_usd']:.2f}"
    print(f"{m:<10} | {r1['trades']:<11} | {r1['win_rate_pct']:<7.1f}% | {r1['net_r']:<+10.1f} | {p1:<12} || {r0['trades']:<11} | {r0['win_rate_pct']:<7.1f}% | {r0['net_r']:<+10.1f} | {p0:<12}")
