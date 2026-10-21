#!/usr/bin/env python3
"""
Comprehensive 3.0% Risk-Per-Trade Quantitative Backtest Engine.
Evaluates Spot Gold (XAUUSD), Bitcoin (BTCUSD), and Ethereum (ETHUSD) under 3.0% risk allocation:
1. Flat 3.0% Risk (Fixed Dollar Risk: $30 on $1,000 seed, and scaled across capital tiers $1k-$10k)
2. Dynamic 3.0% Equity Compounding (Fixed Fractional: 3% of current equity recalculated per trade)
3. 3-Way Comparative Leaderboard: 3.0% vs 5.0% vs 7.544% Baseline ($75.44 on $1,000)
4. Multi-Asset Unified Portfolio Simulation (all 3 assets running concurrently)
5. Trailing 12-Month Month-by-Month Breakdown (Oct 2025 - Sep 2026)
6. Institutional Metrics: Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), Calmar Ratio, Max DD
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_production_backtest import compute_deflated_sharpe_ratio, run_backtest_for_asset
from scripts.backtest_5pct_risk import (
    compute_comprehensive_metrics,
    analyze_monthly_breakdown,
    simulate_flat_risk,
    simulate_dynamic_compounding_risk,
)


def main():
    print("=" * 115)
    print("🔬 COMPREHENSIVE 3.0% RISK-PER-TRADE QUANTITATIVE BACKTEST AUDIT")
    print("   Evaluating Spot Gold, Bitcoin & Ethereum with Real Broker Spreads, Slippage & Overnight Swaps")
    print("=" * 115)

    configs = [
        ("Spot Gold (XAU/USD)", "XAUUSD", 1.5, 8.0, 4.5, 3.0, 0.5, True),
        ("Bitcoin (BTC/USD)", "BTCUSD", 1.5, 500.0, 350.0, 3.0, 30.0, False),
        ("Ethereum (ETH/USD)", "ETHUSD", 1.5, 20.0, 10.0, 3.0, 2.0, True),
    ]

    cached_data = {}
    for label, sym, mult, min_sl, tsl, tp2, be, sess in configs:
        print(f"\n[+] Running Walk-Forward Simulation for {label}...")
        res = run_backtest_for_asset(
            symbol=sym,
            sl_atr_mult=mult,
            min_sl_dist=min_sl,
            tsl_buffer=tsl,
            tp2_mult=tp2,
            be_offset=be,
            session_filter=sess,
            scale_out_r=1.0,
        )
        cached_data[sym] = {
            "label": label,
            "trades": res["all_trades"],
        }

    # =========================================================================
    # SECTION 1: 3.0% vs 5.0% vs 7.544% Production Baseline on $1,000 Seed
    # =========================================================================
    print("\n" + "=" * 154)
    print("📊 SECTION 1: 2-YEAR PERFORMANCE COMPARISON — 3.0% RISK ($30) VS 5.0% ($50) VS 7.544% ($75.44) ON $1,000")
    print("=" * 154)
    hdr = f"{'Asset / Sizing Strategy':<46} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Payoff':<6} | {'Sharpe':<6} | {'Calmar':<6} | {'Max DD %':<8} | {'Max DD ($)':<11} | {'Net P&L ($)':<14} | {'Final Balance'}"
    print(hdr)
    print("-" * 154)

    def print_row(name, m):
        print(
            f"{name:<46} | {m['total_trades']:<6} | {m['win_rate_pct']:>5.1f}% | {m['profit_factor']:>5.2f} | "
            f"{m['net_r']:>+7.1f}R | {m['payoff_ratio']:>5.2f}:1 | {m['sharpe_ratio']:>6.2f} | {m['calmar_ratio']:>6.2f} | "
            f"{m['max_dd_pct']:>7.1f}% | ${m['max_dd_usd']:>9,.2f} | ${m['net_pnl']:>+12,.2f} | ${m['final_balance']:>13,.2f}"
        )

    comparison_results = {}
    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        lbl = cached_data[sym]["label"]
        trades = cached_data[sym]["trades"]

        m_3pct = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.03, is_compounding=False)
        m_5pct = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.05, is_compounding=False)
        m_base = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.07544, is_compounding=False)
        m_3pct_comp = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.03, is_compounding=True)

        comparison_results[sym] = {
            "3pct_flat": m_3pct,
            "5pct_flat": m_5pct,
            "baseline_flat": m_base,
            "3pct_compounding": m_3pct_comp,
        }

        print_row(f"{lbl} — 3.0% Flat Risk ($30/tr)", m_3pct)
        print_row(f"{lbl} — 5.0% Flat Risk ($50/tr)", m_5pct)
        print_row(f"{lbl} — 7.54% Baseline ($75.44/tr)", m_base)
        print_row(f"{lbl} — 3.0% Compounding (Fixed Frac)", m_3pct_comp)
        print("-" * 154)

    # =========================================================================
    # SECTION 2: Multi-Capital Scaling Tiers under 3.0% Flat Risk ($1k-$10k)
    # =========================================================================
    print("\n" + "=" * 132)
    print("💰 SECTION 2: 3.0% FLAT RISK SCALED ACROSS ACCOUNT CAPITAL TIERS ($1,000 TO $10,000)")
    print("=" * 132)
    cap_hdr = f"{'Capital Seed':<14} | {'Planned Risk (3%)':<18} | {'Asset':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Net R':<8} | {'Max DD ($)':<12} | {'Max DD %':<8} | {'Net P&L ($)':<14} | {'Final Equity'}"
    print(cap_hdr)
    print("-" * 132)

    capital_tiers = [1000.0, 2000.0, 3000.0, 5000.0, 10000.0]
    tier_results = defaultdict(dict)

    for cap in capital_tiers:
        risk_usd = cap * 0.03
        for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
            lbl = cached_data[sym]["label"]
            trades = cached_data[sym]["trades"]
            m = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=cap, risk_pct=0.03, is_compounding=False)
            tier_results[cap][sym] = m
            print(
                f"${cap:>10,.2f}   | ${risk_usd:>8,.2f} / trade    | {lbl:<20} | {m['total_trades']:<6} | {m['win_rate_pct']:>6.1f}% | "
                f"{m['net_r']:>+7.1f}R | ${m['max_dd_usd']:>10,.2f} | {m['max_dd_pct']:>6.1f}% | ${m['net_pnl']:>+12,.2f} | ${m['final_balance']:>12,.2f}"
            )
        print("-" * 132)

    # =========================================================================
    # SECTION 3: Unified Multi-Asset Portfolio Simulation at 3.0% Risk
    # =========================================================================
    print("\n" + "=" * 132)
    print("🌐 SECTION 3: UNIFIED MULTI-ASSET PORTFOLIO (GOLD + BTC + ETH CONCURRENT)")
    print("=" * 132)

    all_pooled_trades = []
    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        t_list = cached_data[sym]["trades"]
        t_valid = [t for t in t_list if t.get("ml_prob") is not None and t["ml_prob"] >= 0.44]
        for t in t_valid:
            item = dict(t)
            item["symbol"] = sym
            all_pooled_trades.append(item)

    all_pooled_trades.sort(key=lambda x: x["date"])

    def simulate_pooled(pooled_trades, initial_capital, risk_fraction, is_compounding=False):
        balance = initial_capital
        peak = initial_capital
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        trade_pnls = []
        wins = 0

        for t in pooled_trades:
            if balance <= 10.0:
                break
            planned_risk = (balance * risk_fraction) if is_compounding else (initial_capital * risk_fraction)
            pnl = planned_risk * t["pnl_r"]
            trade_pnls.append(pnl)
            balance += pnl
            balance = max(0.0, balance)
            peak = max(peak, balance)
            dd_usd = peak - balance
            dd_pct = (dd_usd / peak * 100.0) if peak > 0 else 0.0
            max_dd_usd = max(max_dd_usd, dd_usd)
            max_dd_pct = max(max_dd_pct, dd_pct)
            if t["pnl_r"] > 0:
                wins += 1

        tot = len(pooled_trades)
        net_r = sum(t["pnl_r"] for t in pooled_trades)
        gw = sum(planned_risk * t["pnl_r"] for t in pooled_trades if t["pnl_r"] > 0)
        gl = abs(sum(planned_risk * t["pnl_r"] for t in pooled_trades if t["pnl_r"] <= 0))
        pf = gw / gl if gl > 0 else 999.0

        return {
            "trades": tot,
            "win_rate": (wins / tot * 100.0) if tot > 0 else 0.0,
            "profit_factor": round(pf, 2),
            "net_r": round(net_r, 1),
            "final_balance": round(balance, 2),
            "net_pnl": round(balance - initial_capital, 2),
            "roi_pct": round((balance - initial_capital) / initial_capital * 100.0, 1),
            "max_dd_usd": round(max_dd_usd, 2),
            "max_dd_pct": round(max_dd_pct, 1),
        }

    # 1. $3,000 seed with $30 / asset trade (3% of $1,000 sub-allocation)
    p_alloc_30 = simulate_pooled(all_pooled_trades, 3000.0, 30.0 / 3000.0, is_compounding=False)
    # 2. $3,000 seed with 3.0% of total account ($90 / trade)
    p_flat_90 = simulate_pooled(all_pooled_trades, 3000.0, 0.03, is_compounding=False)
    # 3. Dynamic Compounding 3.0%
    p_comp_3pct = simulate_pooled(all_pooled_trades, 3000.0, 0.03, is_compounding=True)
    # 4. 5% and Baseline Comparisons for reference
    p_alloc_50 = simulate_pooled(all_pooled_trades, 3000.0, 50.0 / 3000.0, is_compounding=False)
    p_base_alloc = simulate_pooled(all_pooled_trades, 3000.0, 75.44 / 3000.0, is_compounding=False)

    port_hdr = f"{'Portfolio Architecture':<44} | {'Seed ($)':<10} | {'Risk Model':<20} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Max DD %':<8} | {'Max DD ($)':<11} | {'Net P&L ($)':<13} | {'Final Equity'}"
    print(port_hdr)
    print("-" * 154)

    def print_port_row(name, seed, model, m):
        print(
            f"{name:<44} | ${seed:>8,.2f} | {model:<20} | {m['trades']:<6} | {m['win_rate']:>5.1f}% | {m['profit_factor']:>5.2f} | "
            f"{m['net_r']:>+7.1f}R | {m['max_dd_pct']:>7.1f}% | ${m['max_dd_usd']:>9,.2f} | ${m['net_pnl']:>+11,.2f} | ${m['final_balance']:>12,.2f}"
        )

    print_port_row("Portfolio: $30 Risk / Asset Trade (3%)", 3000.0, "Flat $30 (3% per asset)", p_alloc_30)
    print_port_row("Portfolio: 3.0% Risk of Total Account", 3000.0, "Flat $90 (3% total)", p_flat_90)
    print_port_row("Portfolio: 3.0% Dynamic Compounding", 3000.0, "Compounding 3.0%", p_comp_3pct)
    print_port_row("Portfolio: $50 Risk / Asset Trade (5%)", 3000.0, "Flat $50 (5% per asset)", p_alloc_50)
    print_port_row("Portfolio: Baseline $75.44 per Asset", 3000.0, "Flat $75.44 (Baseline)", p_base_alloc)
    print("-" * 154)

    # =========================================================================
    # SECTION 4: Month-by-Month Trailing 12 Months Breakdown (3.0% = $30.00 / Trade)
    # =========================================================================
    print("\n" + "=" * 115)
    print("📅 SECTION 4: TRAILING 12-MONTH MONTH-ON-MONTH BREAKOUT (3.0% FLAT RISK = $30.00 / TRADE)")
    print("=" * 115)

    gold_monthly = analyze_monthly_breakdown(comparison_results["XAUUSD"]["3pct_flat"]["selected_trades"], risk_usd=30.0)
    btc_monthly = analyze_monthly_breakdown(comparison_results["BTCUSD"]["3pct_flat"]["selected_trades"], risk_usd=30.0)
    eth_monthly = analyze_monthly_breakdown(comparison_results["ETHUSD"]["3pct_flat"]["selected_trades"], risk_usd=30.0)

    m_hdr = f"{'Month':<10} | {'Gold P&L':<14} | {'Gold R':<8} | {'BTC P&L':<14} | {'BTC R':<8} | {'ETH P&L':<14} | {'ETH R':<8} | {'Portfolio P&L':<16} | {'Portfolio Net R'}"
    print(m_hdr)
    print("-" * 115)

    months = [r["month"] for r in gold_monthly]
    gold_map = {r["month"]: r for r in gold_monthly}
    btc_map = {r["month"]: r for r in btc_monthly}
    eth_map = {r["month"]: r for r in eth_monthly}

    tot_gold_pnl = 0.0
    tot_btc_pnl = 0.0
    tot_eth_pnl = 0.0
    tot_port_pnl = 0.0
    tot_port_r = 0.0
    winning_months = 0

    for m in months:
        g = gold_map.get(m, {"pnl_usd": 0.0, "net_r": 0.0})
        b = btc_map.get(m, {"pnl_usd": 0.0, "net_r": 0.0})
        e = eth_map.get(m, {"pnl_usd": 0.0, "net_r": 0.0})

        port_m_pnl = g["pnl_usd"] + b["pnl_usd"] + e["pnl_usd"]
        port_m_r = g["net_r"] + b["net_r"] + e["net_r"]

        tot_gold_pnl += g["pnl_usd"]
        tot_btc_pnl += b["pnl_usd"]
        tot_eth_pnl += e["pnl_usd"]
        tot_port_pnl += port_m_pnl
        tot_port_r += port_m_r

        if port_m_pnl > 0:
            winning_months += 1

        print(
            f"{m:<10} | ${g['pnl_usd']:>+11,.2f} | {g['net_r']:>+6.1f}R | ${b['pnl_usd']:>+11,.2f} | {b['net_r']:>+6.1f}R | "
            f"${e['pnl_usd']:>+11,.2f} | {e['net_r']:>+6.1f}R | ${port_m_pnl:>+13,.2f} | {port_m_r:>+13.1f}R"
        )

    print("-" * 115)
    print(
        f"{'TOTAL 12M':<10} | ${tot_gold_pnl:>+11,.2f} |        | ${tot_btc_pnl:>+11,.2f} |        | "
        f"${tot_eth_pnl:>+11,.2f} |        | ${tot_port_pnl:>+13,.2f} | {tot_port_r:>+13.1f}R"
    )
    print(f"\nWinning Months: {winning_months} / 12 ({winning_months / 12 * 100.0:.1f}%)")

    # Save output to JSON
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": "3.0% Risk-Per-Trade Backtest Audit",
        "gold": {
            "3pct_flat": comparison_results["XAUUSD"]["3pct_flat"],
            "5pct_flat": comparison_results["XAUUSD"]["5pct_flat"],
            "7_54pct_baseline": comparison_results["XAUUSD"]["baseline_flat"],
            "3pct_compounding": comparison_results["XAUUSD"]["3pct_compounding"],
        },
        "bitcoin": {
            "3pct_flat": comparison_results["BTCUSD"]["3pct_flat"],
            "5pct_flat": comparison_results["BTCUSD"]["5pct_flat"],
            "7_54pct_baseline": comparison_results["BTCUSD"]["baseline_flat"],
            "3pct_compounding": comparison_results["BTCUSD"]["3pct_compounding"],
        },
        "ethereum": {
            "3pct_flat": comparison_results["ETHUSD"]["3pct_flat"],
            "5pct_flat": comparison_results["ETHUSD"]["5pct_flat"],
            "7_54pct_baseline": comparison_results["ETHUSD"]["baseline_flat"],
            "3pct_compounding": comparison_results["ETHUSD"]["3pct_compounding"],
        },
        "portfolio": {
            "alloc_30": p_alloc_30,
            "flat_90": p_flat_90,
            "compounding_3pct": p_comp_3pct,
            "alloc_50": p_alloc_50,
            "baseline_alloc_75": p_base_alloc,
        },
        "trailing_12m": {
            "months": months,
            "tot_port_pnl": tot_port_pnl,
            "tot_port_r": tot_port_r,
            "winning_months": winning_months,
        },
    }

    out_file = os.path.expanduser("~/.gold_trading_data/backtest_3pct_risk_results.json")
    def clean(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "selected_trades"}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    with open(out_file, "w") as f:
        json.dump(clean(report), f, indent=2)

    print(f"\n✅ Detailed 3% Risk Backtest Results Stored At: {out_file}")


if __name__ == "__main__":
    main()
