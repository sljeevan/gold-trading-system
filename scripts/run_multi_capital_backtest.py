from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_production_backtest import run_backtest_for_asset
from scripts.analyze_monthly_breakout import analyze_monthly_trades

configs = [
    ("Spot Gold (XAUUSD)", "XAUUSD", 1.5, 8.0, 4.5, 3.0, 0.5, True),
    ("Bitcoin (BTCUSD)", "BTCUSD", 1.5, 500.0, 350.0, 3.0, 30.0, False),
    ("Ethereum (ETHUSD)", "ETHUSD", 1.5, 20.0, 10.0, 3.0, 2.0, True),
]

capitals = [2000.0, 3000.0, 5000.0]
base_risk_ratio = 75.44 / 1000.0  # 7.544%

print("=" * 100)
print("📊 PRODUCTION MULTI-CAPITAL BACKTEST ($2,000 | $3,000 | $5,000)")
print("Period: 12 Full Months (October 2025 – September 2026) | Full Trading Friction Deducted")
print("=" * 100)

# Step 1: Precompute trades for both 0.5R and 1.0R to avoid redundant recalculation
cached_trades = {}
for label, sym, mult, min_sl, tsl, tp2, be, sess in configs:
    print(f"Generating backtest data for {sym}...")
    res_05 = run_backtest_for_asset(sym, mult, min_sl, tsl, tp2, be, sess, scale_out_r=0.5)
    res_10 = run_backtest_for_asset(sym, mult, min_sl, tsl, tp2, be, sess, scale_out_r=1.0)
    cached_trades[sym] = {
        "0.5R": res_05["all_trades"],
        "1.0R": res_10["all_trades"],
        "label": label,
    }

print("\n" + "=" * 100)
print("1️⃣ FLAT RISK PRODUCTION BASELINE (7.544% Planned Risk per Trade)")
print("=" * 100)

for cap in capitals:
    risk_usd = cap * base_risk_ratio
    print(f"\n==========================================================================================")
    print(f"💰 INITIAL CAPITAL: ${cap:,.2f}  |  PLANNED RISK PER TRADE: ${risk_usd:.2f} (7.544%)")
    print(f"==========================================================================================")
    
    print(f"{'Asset':<22} | {'Variant':<9} | {'Trades':<6} | {'WR %':<6} | {'Net R':<8} | {'PF':<5} | {'Net P&L ($)':<12} | {'Final Bal ($)':<14} | {'ROI %':<8} | {'Max DD ($)':<12} | {'Max DD %':<8}")
    print("-" * 125)
    
    portfolio_pnl_05 = 0.0
    portfolio_pnl_10 = 0.0
    
    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        lbl = cached_trades[sym]["label"]
        
        # 0.5R Variant
        m_05 = analyze_monthly_trades(sym, cached_trades[sym]["0.5R"], ml_threshold=0.44, planned_risk_usd=risk_usd, starting_balance=cap, start_month="2025-10", end_month="2026-09")
        c_05 = m_05["completed_12m"]
        fin_05 = cap + c_05["net_pnl_usd"]
        portfolio_pnl_05 += c_05["net_pnl_usd"]
        
        # 1.0R Baseline
        m_10 = analyze_monthly_trades(sym, cached_trades[sym]["1.0R"], ml_threshold=0.44, planned_risk_usd=risk_usd, starting_balance=cap, start_month="2025-10", end_month="2026-09")
        c_10 = m_10["completed_12m"]
        fin_10 = cap + c_10["net_pnl_usd"]
        portfolio_pnl_10 += c_10["net_pnl_usd"]
        
        print(f"{lbl:<22} | {'0.50R TP':<9} | {c_05['trades']:<6} | {c_05['win_rate_pct']:<5.1f}% | {c_05['net_r']:<+8.1f} | {c_05['profit_factor']:<5.2f} | ${c_05['net_pnl_usd']:<11,.2f} | ${fin_05:<13,.2f} | +{c_05['return_pct']:<6.1f}% | ${m_05['max_drawdown_usd']:<11,.2f} | {m_05['max_drawdown_pct']:<6.1f}%")
        print(f"{lbl:<22} | {'1.00R TP':<9} | {c_10['trades']:<6} | {c_10['win_rate_pct']:<5.1f}% | {c_10['net_r']:<+8.1f} | {c_10['profit_factor']:<5.2f} | ${c_10['net_pnl_usd']:<11,.2f} | ${fin_10:<13,.2f} | +{c_10['return_pct']:<6.1f}% | ${m_10['max_drawdown_usd']:<11,.2f} | {m_10['max_drawdown_pct']:<6.1f}%")
        print("-" * 125)

print("\n" + "=" * 100)
print("2️⃣ UNIFIED PORTFOLIO SIMULATION (Total Account Seed = $2k / $3k / $5k allocated across all 3 assets)")
print("=" * 100)
print("Assuming a single shared trading account running all 3 assets concurrently with risk proportionally scaled to total capital:")
print(f"{'Total Capital':<14} | {'Risk/Trade (7.54%)':<19} | {'TP Strategy':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Net R':<8} | {'Net P&L ($)':<12} | {'Final Equity ($)':<16} | {'ROI %':<8} | {'Profitable Mo'}")
print("-" * 125)

for cap in capitals:
    # If the single total account is $2k, $3k, $5k, each asset gets 1/3 of the risk budget per trade, or each trade risks 7.544% of that asset's $cap/3 allocation.
    # Alternatively, if total account risks 7.544% of total account per trade:
    # Let's show both or show the combined 3-asset performance:
    # 1. Total Account allocated 1/3 to each asset (Total initial = $cap):
    asset_cap = cap / 3.0
    risk_usd = asset_cap * base_risk_ratio
    
    for variant, v_key in [("0.50R TP", "0.5R"), ("1.00R TP", "1.0R")]:
        tot_trades = 0
        tot_wins = 0
        tot_losses = 0
        tot_r = 0.0
        tot_pnl = 0.0
        monthly_pnls = {}
        
        for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
            m = analyze_monthly_trades(sym, cached_trades[sym][v_key], ml_threshold=0.44, planned_risk_usd=risk_usd, starting_balance=asset_cap, start_month="2025-10", end_month="2026-09")
            c = m["completed_12m"]
            tot_trades += c["trades"]
            tot_wins += c["wins"]
            tot_losses += c["losses"]
            tot_r += c["net_r"]
            tot_pnl += c["net_pnl_usd"]
            for r in m["monthly_records"]:
                if r["month"] < "2026-10":
                    monthly_pnls[r["month"]] = monthly_pnls.get(r["month"], 0.0) + r["net_pnl_usd"]
        
        wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
        fin_eq = cap + tot_pnl
        roi = (tot_pnl / cap * 100.0)
        win_months = sum(1 for m, p in monthly_pnls.items() if p > 0)
        
        print(f"${cap:<13,.2f} | ${risk_usd:<18.2f} | {variant:<12} | {tot_trades:<6} | {wr:<7.1f}% | {tot_r:<+8.1f} | ${tot_pnl:<11,.2f} | ${fin_eq:<15,.2f} | +{roi:<7.1f}% | {win_months}/12")
    print("-" * 125)

print("\n" + "=" * 100)
print("3️⃣ DYNAMIC EQUITY COMPOUNDING SIMULATION (Reinvesting Profits Each Trade)")
print("=" * 100)
print("Simulating sequential compounding where risk scales with account balance:\n")

def run_compounding_sim(trades: List[Dict[str, Any]], start_cap: float, risk_pct: float) -> Dict[str, Any]:
    # Sort trades chronologically
    sorted_trades = sorted(trades, key=lambda t: t["date"])
    bal = start_cap
    hwm = start_cap
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    wins = 0
    losses = 0
    
    for t in sorted_trades:
        r_mult = t["pnl_r"]
        risk_dollar = bal * risk_pct
        trade_pnl = risk_dollar * r_mult
        bal += trade_pnl
        if trade_pnl > 0:
            wins += 1
        else:
            losses += 1
        if bal > hwm:
            hwm = bal
        dd_usd = hwm - bal
        dd_pct = (dd_usd / hwm * 100.0) if hwm > 0 else 0.0
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            
    return {
        "start_cap": start_cap,
        "final_bal": bal,
        "net_pnl": bal - start_cap,
        "roi_pct": (bal - start_cap) / start_cap * 100.0,
        "max_dd_pct": max_dd_pct,
        "max_dd_usd": max_dd_usd,
        "total_trades": len(sorted_trades),
        "win_rate": (wins / len(sorted_trades) * 100.0) if sorted_trades else 0.0,
    }

# Gather 12-month filtered trades
all_c12_trades_05 = []
all_c12_trades_10 = []
for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
    t_05 = [t for t in cached_trades[sym]["0.5R"] if t.get("ml_prob") is not None and t["ml_prob"] >= 0.44 and "2025-10-01" <= t["date"] <= "2026-09-30 23:59"]
    t_10 = [t for t in cached_trades[sym]["1.0R"] if t.get("ml_prob") is not None and t["ml_prob"] >= 0.44 and "2025-10-01" <= t["date"] <= "2026-09-30 23:59"]
    all_c12_trades_05.extend(t_05)
    all_c12_trades_10.extend(t_10)

print("A. Conservative Risk Compounding (3.0% Risk / Trade):")
print(f"{'Starting Capital':<18} | {'Strategy':<12} | {'Trades':<6} | {'Final Equity ($)':<18} | {'Net P&L ($)':<16} | {'ROI %':<10} | {'Max DD %':<8}")
print("-" * 100)
for cap in capitals:
    for lbl, tr in [("0.50R TP", all_c12_trades_05), ("1.00R TP", all_c12_trades_10)]:
        res = run_compounding_sim(tr, cap, risk_pct=0.03)
        print(f"${cap:<17,.2f} | {lbl:<12} | {res['total_trades']:<6} | ${res['final_bal']:<17,.2f} | ${res['net_pnl']:<15,.2f} | +{res['roi_pct']:<9.1f}% | {res['max_dd_pct']:<7.1f}%")
    print("-" * 100)

print("\nB. Standard Risk Compounding (5.0% Risk / Trade):")
print(f"{'Starting Capital':<18} | {'Strategy':<12} | {'Trades':<6} | {'Final Equity ($)':<18} | {'Net P&L ($)':<16} | {'ROI %':<10} | {'Max DD %':<8}")
print("-" * 100)
for cap in capitals:
    for lbl, tr in [("0.50R TP", all_c12_trades_05), ("1.00R TP", all_c12_trades_10)]:
        res = run_compounding_sim(tr, cap, risk_pct=0.05)
        print(f"${cap:<17,.2f} | {lbl:<12} | {res['total_trades']:<6} | ${res['final_bal']:<17,.2f} | ${res['net_pnl']:<15,.2f} | +{res['roi_pct']:<9.1f}% | {res['max_dd_pct']:<7.1f}%")
    print("-" * 100)

print("\nC. Aggressive Compounding (7.54% Risk / Trade):")
print(f"{'Starting Capital':<18} | {'Strategy':<12} | {'Trades':<6} | {'Final Equity ($)':<18} | {'Net P&L ($)':<16} | {'ROI %':<10} | {'Max DD %':<8}")
print("-" * 100)
for cap in capitals:
    for lbl, tr in [("0.50R TP", all_c12_trades_05), ("1.00R TP", all_c12_trades_10)]:
        res = run_compounding_sim(tr, cap, risk_pct=0.07544)
        print(f"${cap:<17,.2f} | {lbl:<12} | {res['total_trades']:<6} | ${res['final_bal']:<17,.2f} | ${res['net_pnl']:<15,.2f} | +{res['roi_pct']:<9.1f}% | {res['max_dd_pct']:<7.1f}%")
    print("-" * 100)
