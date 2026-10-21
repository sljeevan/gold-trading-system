#!/usr/bin/env python3
"""
Comprehensive 5% Risk-Per-Trade Quantitative Backtest Engine.
Evaluates Spot Gold (XAUUSD), Bitcoin (BTCUSD), and Ethereum (ETHUSD) under 5% risk allocation:
1. Flat 5.0% Risk (Fixed Dollar Risk: $50 on $1,000 seed, and scaled across capital tiers $1k-$10k)
2. Dynamic 5.0% Equity Compounding (Fixed Fractional: 5% of current equity recalculated per trade)
3. Direct Comparison: 5.0% Risk vs 7.544% Production Baseline ($75.44 on $1,000)
4. Multi-Asset Unified Portfolio Simulation (all 3 assets running concurrently)
5. Trailing 12-Month Month-by-Month Breakdown (Oct 2025 - Oct 2026)
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


def simulate_flat_risk(
    trades: List[Dict[str, Any]],
    initial_capital: float = 1000.0,
    risk_pct: float = 0.05,
) -> Dict[str, Any]:
    """Simulate flat dollar risk (fixed risk = initial_capital * risk_pct)."""
    planned_risk = initial_capital * risk_pct
    balance = initial_capital
    peak = initial_capital
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    trade_pnls = []

    consecutive_losses = 0
    max_consecutive_losses = 0
    consecutive_wins = 0
    max_consecutive_wins = 0

    for t in trades:
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
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    net_pnl = balance - initial_capital
    roi_pct = (net_pnl / initial_capital) * 100.0

    return {
        "initial_capital": initial_capital,
        "planned_risk_usd": planned_risk,
        "risk_pct": risk_pct * 100.0,
        "final_balance": round(balance, 2),
        "net_pnl": round(net_pnl, 2),
        "roi_pct": round(roi_pct, 1),
        "peak_balance": round(peak, 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "trade_pnls": trade_pnls,
    }


def simulate_dynamic_compounding_risk(
    trades: List[Dict[str, Any]],
    symbol: str,
    initial_capital: float = 1000.0,
    risk_pct: float = 0.05,
    min_lot: float = 0.01,
) -> Dict[str, Any]:
    """
    Simulate dynamic equity % compounding (recalculating risk = equity * risk_pct on each trade).
    Enforces broker contract size and minimum lot size constraints.
    """
    contract_size = 100.0 if "XAU" in symbol.upper() or "GOLD" in symbol.upper() else 1.0
    balance = initial_capital
    peak = initial_capital
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    trade_pnls = []

    consecutive_losses = 0
    max_consecutive_losses = 0
    consecutive_wins = 0
    max_consecutive_wins = 0

    for t in trades:
        if balance <= 10.0:  # account busted / margin call
            break

        target_risk = balance * risk_pct
        sl_pts = max(0.01, t["sl_dist"])
        lots = round(target_risk / (sl_pts * contract_size), 2)
        lots = max(min_lot, lots)
        actual_risk = lots * sl_pts * contract_size

        pnl = actual_risk * t["pnl_r"]
        trade_pnls.append(pnl)
        balance += pnl
        balance = max(0.0, balance)
        peak = max(peak, balance)
        dd_usd = peak - balance
        dd_pct = (dd_usd / peak * 100.0) if peak > 0 else 0.0
        max_dd_usd = max(max_dd_usd, dd_usd)
        max_dd_pct = max(max_dd_pct, dd_pct)

        if t["pnl_r"] > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    net_pnl = balance - initial_capital
    roi_pct = (net_pnl / initial_capital) * 100.0

    return {
        "initial_capital": initial_capital,
        "risk_pct": risk_pct * 100.0,
        "final_balance": round(balance, 2),
        "net_pnl": round(net_pnl, 2),
        "roi_pct": round(roi_pct, 1),
        "peak_balance": round(peak, 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "trade_pnls": trade_pnls,
    }


def compute_comprehensive_metrics(
    symbol: str,
    trades: List[Dict[str, Any]],
    ml_threshold: Optional[float] = 0.44,
    initial_capital: float = 1000.0,
    risk_pct: float = 0.05,
    is_compounding: bool = False,
    years: float = 2.0,
) -> Dict[str, Any]:
    """Compute institutional quantitative performance metrics for a trade set."""
    selected = trades
    if ml_threshold is not None:
        selected = [t for t in trades if t.get("ml_prob") is not None and t["ml_prob"] >= ml_threshold]

    total_trades = len(selected)
    if total_trades == 0:
        return {"symbol": symbol, "total_trades": 0, "net_pnl": 0.0}

    wins = [t for t in selected if t["pnl_r"] > 0]
    losses = [t for t in selected if t["pnl_r"] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0

    total_r = sum(t["pnl_r"] for t in selected)
    gross_profit_pts = sum(t["pnl_pts"] for t in wins)
    gross_loss_pts = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_profit_pts / gross_loss_pts if gross_loss_pts > 0 else 999.0

    avg_win_r = float(np.mean([t["pnl_r"] for t in wins])) if wins else 0.0
    avg_loss_r = abs(float(np.mean([t["pnl_r"] for t in losses]))) if losses else 1.0
    payoff_ratio = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0.0

    exit_counts = {}
    for t in selected:
        et = t.get("exit_type", "UNKNOWN")
        exit_counts[et] = exit_counts.get(et, 0) + 1

    # Run simulation
    if is_compounding:
        sim = simulate_dynamic_compounding_risk(
            selected, symbol=symbol, initial_capital=initial_capital, risk_pct=risk_pct
        )
    else:
        sim = simulate_flat_risk(
            selected, initial_capital=initial_capital, risk_pct=risk_pct
        )

    trade_pnls = sim["trade_pnls"]
    sharpe = 0.0
    dsr_metrics = {"annualized_sharpe": 0.0, "dsr_pct": 0.0, "psr_pct": 0.0}

    if len(trade_pnls) > 1:
        mean_pnl = float(np.mean(trade_pnls))
        std_pnl = float(np.std(trade_pnls))
        if std_pnl > 0:
            trades_per_year = len(trade_pnls) / years
            sharpe = (mean_pnl / std_pnl) * math.sqrt(max(1.0, trades_per_year))
            trade_returns = [pnl / initial_capital for pnl in trade_pnls]
            dsr_metrics = compute_deflated_sharpe_ratio(trade_returns, num_trials=50)

    # Calmar ratio (annualized return / max drawdown %)
    ann_return_pct = sim["roi_pct"] / years
    calmar = (ann_return_pct / sim["max_dd_pct"]) if sim["max_dd_pct"] > 0 else 0.0

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_r": round(total_r, 1),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "exit_counts": exit_counts,
        "initial_capital": initial_capital,
        "risk_pct": risk_pct * 100.0,
        "is_compounding": is_compounding,
        "final_balance": sim["final_balance"],
        "net_pnl": sim["net_pnl"],
        "roi_pct": sim["roi_pct"],
        "peak_balance": sim["peak_balance"],
        "max_dd_usd": sim["max_dd_usd"],
        "max_dd_pct": sim["max_dd_pct"],
        "max_consecutive_wins": sim["max_consecutive_wins"],
        "max_consecutive_losses": sim["max_consecutive_losses"],
        "annualized_return_pct": round(ann_return_pct, 1),
        "calmar_ratio": round(calmar, 2),
        "sharpe_ratio": round(sharpe, 2),
        "deflated_sharpe_pct": dsr_metrics["dsr_pct"],
        "probabilistic_sharpe_pct": dsr_metrics["psr_pct"],
        "selected_trades": selected,
    }


def analyze_monthly_breakdown(
    trades: List[Dict[str, Any]],
    risk_usd: float = 50.0,
    start_month: str = "2025-10",
    end_month: str = "2026-09",
) -> List[Dict[str, Any]]:
    """Generate monthly breakdown of P&L and metrics for the completed 12-month window."""
    groups = defaultdict(list)
    for t in trades:
        m = t["date"][:7]
        if start_month <= m <= end_month:
            groups[m].append(t)

    all_months = sorted(groups.keys())
    results = []
    cum_pnl = 0.0
    cum_r = 0.0

    for m in all_months:
        m_trades = groups[m]
        tot = len(m_trades)
        wins = [t for t in m_trades if t["pnl_r"] > 0]
        losses = [t for t in m_trades if t["pnl_r"] <= 0]
        wr = (len(wins) / tot * 100.0) if tot > 0 else 0.0
        net_r = sum(t["pnl_r"] for t in m_trades)
        pnl = sum(risk_usd * t["pnl_r"] for t in m_trades)
        cum_pnl += pnl
        cum_r += net_r

        results.append({
            "month": m,
            "trades": tot,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(wr, 1),
            "net_r": round(net_r, 1),
            "pnl_usd": round(pnl, 2),
            "cumulative_pnl_usd": round(cum_pnl, 2),
            "cumulative_r": round(cum_r, 1),
        })

    return results


def main():
    print("=" * 115)
    print("🔬 COMPREHENSIVE 5.0% RISK-PER-TRADE QUANTITATIVE BACKTEST AUDIT")
    print("   Evaluating Spot Gold, Bitcoin & Ethereum with Real Broker Spreads, Slippage & Overnight Swaps")
    print("=" * 115)

    # Asset configurations
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
            scale_out_r=1.0,  # Production standard 1.0R scale out
        )
        cached_data[sym] = {
            "label": label,
            "trades": res["all_trades"],
        }

    # =========================================================================
    # SECTION 1: 5.0% Flat Risk ($50 on $1,000) vs 7.544% Production Baseline ($75.44)
    # =========================================================================
    print("\n" + "=" * 148)
    print("📊 SECTION 1: 2-YEAR PERFORMANCE COMPARISON — 5.0% FLAT RISK ($50) VS 7.544% BASELINE ($75.44) ON $1,000")
    print("=" * 148)
    hdr = f"{'Asset / Sizing Strategy':<46} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Payoff':<6} | {'Sharpe':<6} | {'Calmar':<6} | {'Max DD %':<8} | {'Max DD ($)':<11} | {'Net P&L ($)':<13} | {'Final Balance'}"
    print(hdr)
    print("-" * 148)

    def print_row(name, m):
        print(
            f"{name:<46} | {m['total_trades']:<6} | {m['win_rate_pct']:>5.1f}% | {m['profit_factor']:>5.2f} | "
            f"{m['net_r']:>+7.1f}R | {m['payoff_ratio']:>5.2f}:1 | {m['sharpe_ratio']:>6.2f} | {m['calmar_ratio']:>6.2f} | "
            f"{m['max_dd_pct']:>7.1f}% | ${m['max_dd_usd']:>9,.2f} | ${m['net_pnl']:>+11,.2f} | ${m['final_balance']:>12,.2f}"
        )

    comparison_results = {}
    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        lbl = cached_data[sym]["label"]
        trades = cached_data[sym]["trades"]

        # 5.0% Flat Risk ($50 on $1,000)
        m_5pct = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.05, is_compounding=False)
        # 7.544% Baseline ($75.44 on $1,000)
        m_base = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.07544, is_compounding=False)
        # 5.0% Dynamic Compounding (recalculated per trade)
        m_comp = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=1000.0, risk_pct=0.05, is_compounding=True)

        comparison_results[sym] = {
            "5pct_flat": m_5pct,
            "baseline_flat": m_base,
            "5pct_compounding": m_comp,
        }

        print_row(f"{lbl} — 5.0% Flat Risk ($50/tr)", m_5pct)
        print_row(f"{lbl} — 7.54% Baseline ($75.44/tr)", m_base)
        print_row(f"{lbl} — 5.0% Compounding (Fixed Frac)", m_comp)
        print("-" * 148)

    # =========================================================================
    # SECTION 2: Multi-Capital Tiers under 5.0% Flat Risk ($1k, $2k, $3k, $5k, $10k)
    # =========================================================================
    print("\n" + "=" * 128)
    print("💰 SECTION 2: 5.0% FLAT RISK SCALED ACROSS ACCOUNT CAPITAL TIERS ($1,000 TO $10,000)")
    print("=" * 128)
    cap_hdr = f"{'Capital Seed':<14} | {'Planned Risk (5%)':<18} | {'Asset':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Net R':<8} | {'Max DD ($)':<12} | {'Max DD %':<8} | {'Net P&L ($)':<14} | {'Final Equity'}"
    print(cap_hdr)
    print("-" * 128)

    capital_tiers = [1000.0, 2000.0, 3000.0, 5000.0, 10000.0]
    tier_results = defaultdict(dict)

    for cap in capital_tiers:
        risk_usd = cap * 0.05
        for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
            lbl = cached_data[sym]["label"]
            trades = cached_data[sym]["trades"]
            m = compute_comprehensive_metrics(sym, trades, ml_threshold=0.44, initial_capital=cap, risk_pct=0.05, is_compounding=False)
            tier_results[cap][sym] = m
            print(
                f"${cap:>10,.2f}   | ${risk_usd:>8,.2f} / trade    | {lbl:<20} | {m['total_trades']:<6} | {m['win_rate_pct']:>6.1f}% | "
                f"{m['net_r']:>+7.1f}R | ${m['max_dd_usd']:>10,.2f} | {m['max_dd_pct']:>6.1f}% | ${m['net_pnl']:>+12,.2f} | ${m['final_balance']:>12,.2f}"
            )
        print("-" * 128)

    # =========================================================================
    # SECTION 3: Unified Multi-Asset Portfolio Simulation at 5.0% Risk
    # =========================================================================
    print("\n" + "=" * 128)
    print("🌐 SECTION 3: UNIFIED MULTI-ASSET PORTFOLIO (GOLD + BTC + ETH CONCURRENT)")
    print("=" * 128)
    print("Chronological cross-asset order stream simulating a single account trading Gold, BTC, and ETH concurrently.")

    all_pooled_trades = []
    for sym in ["XAUUSD", "BTCUSD", "ETHUSD"]:
        t_list = cached_data[sym]["trades"]
        # Filter by production ML threshold 0.44
        t_valid = [t for t in t_list if t.get("ml_prob") is not None and t["ml_prob"] >= 0.44]
        for t in t_valid:
            item = dict(t)
            item["symbol"] = sym
            all_pooled_trades.append(item)

    # Sort strictly chronologically by date
    all_pooled_trades.sort(key=lambda x: x["date"])
    print(f"Total Portfolio Trades (Gold + BTC + ETH combined): {len(all_pooled_trades)}")

    port_hdr = f"{'Portfolio Architecture':<42} | {'Seed ($)':<10} | {'Risk Model':<20} | {'Trades':<6} | {'Win %':<6} | {'PF':<5} | {'Net R':<8} | {'Max DD %':<8} | {'Max DD ($)':<11} | {'Net P&L ($)':<13} | {'Final Equity'}"
    print(port_hdr)
    print("-" * 148)

    # 1. Flat 5% risk on $3,000 ($1,000 per asset seed -> $150 total risk / trade if 5% of total account, or $50 per asset trade)
    # Scenario A: Dedicated sub-account per asset ($50 risk per asset trade on $3,000 total capital)
    tot_pnl_flat_50 = sum(comparison_results[s]["5pct_flat"]["net_pnl"] for s in ["XAUUSD", "BTCUSD", "ETHUSD"])
    tot_dd_usd_flat_50 = sum(comparison_results[s]["5pct_flat"]["max_dd_usd"] for s in ["XAUUSD", "BTCUSD", "ETHUSD"])
    # True pooled chronological simulation:
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

    # Unified Account $3,000 Seed:
    # 1. Flat 5% of asset allocation (each trade risks $50, which is 1.67% of total account, or 5% of that asset's $1k allocation)
    p_alloc_50 = simulate_pooled(all_pooled_trades, 3000.0, 50.0 / 3000.0, is_compounding=False)
    # 2. Flat 5% of total account ($150 risk per trade)
    p_flat_150 = simulate_pooled(all_pooled_trades, 3000.0, 0.05, is_compounding=False)
    # 3. Dynamic Compounding 5% of total account
    p_comp_5pct = simulate_pooled(all_pooled_trades, 3000.0, 0.05, is_compounding=True)
    # 4. Production Baseline Comparison (Flat $75.44 per trade, 7.544% of $1k asset allocation)
    p_base_alloc = simulate_pooled(all_pooled_trades, 3000.0, 75.44 / 3000.0, is_compounding=False)

    def print_port_row(name, seed, model, m):
        print(
            f"{name:<42} | ${seed:>8,.2f} | {model:<20} | {m['trades']:<6} | {m['win_rate']:>5.1f}% | {m['profit_factor']:>5.2f} | "
            f"{m['net_r']:>+7.1f}R | {m['max_dd_pct']:>7.1f}% | ${m['max_dd_usd']:>9,.2f} | ${m['net_pnl']:>+11,.2f} | ${m['final_balance']:>12,.2f}"
        )

    print_port_row("Portfolio: $50 Risk / Asset Trade", 3000.0, "Flat $50 (5% per asset)", p_alloc_50)
    print_port_row("Portfolio: 5.0% Risk of Total Account", 3000.0, "Flat $150 (5% total)", p_flat_150)
    print_port_row("Portfolio: 5.0% Dynamic Compounding", 3000.0, "Compounding 5.0%", p_comp_5pct)
    print_port_row("Portfolio: Baseline $75.44 per Asset", 3000.0, "Flat $75.44 (Baseline)", p_base_alloc)
    print("-" * 148)

    # =========================================================================
    # SECTION 4: Month-by-Month Trailing 12 Months Breakdown (Oct 2025 – Sep 2026)
    # =========================================================================
    print("\n" + "=" * 115)
    print("📅 SECTION 4: TRAILING 12-MONTH MONTH-ON-MONTH BREAKOUT (5.0% FLAT RISK = $50.00 / TRADE)")
    print("=" * 115)

    gold_monthly = analyze_monthly_breakdown(comparison_results["XAUUSD"]["5pct_flat"]["selected_trades"], risk_usd=50.0)
    btc_monthly = analyze_monthly_breakdown(comparison_results["BTCUSD"]["5pct_flat"]["selected_trades"], risk_usd=50.0)
    eth_monthly = analyze_monthly_breakdown(comparison_results["ETHUSD"]["5pct_flat"]["selected_trades"], risk_usd=50.0)

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
        "summary": "5.0% Risk-Per-Trade Backtest Audit",
        "gold": {
            "5pct_flat": comparison_results["XAUUSD"]["5pct_flat"],
            "7_54pct_baseline": comparison_results["XAUUSD"]["baseline_flat"],
            "5pct_compounding": comparison_results["XAUUSD"]["5pct_compounding"],
        },
        "bitcoin": {
            "5pct_flat": comparison_results["BTCUSD"]["5pct_flat"],
            "7_54pct_baseline": comparison_results["BTCUSD"]["baseline_flat"],
            "5pct_compounding": comparison_results["BTCUSD"]["5pct_compounding"],
        },
        "ethereum": {
            "5pct_flat": comparison_results["ETHUSD"]["5pct_flat"],
            "7_54pct_baseline": comparison_results["ETHUSD"]["baseline_flat"],
            "5pct_compounding": comparison_results["ETHUSD"]["5pct_compounding"],
        },
        "portfolio": {
            "alloc_50": p_alloc_50,
            "flat_150": p_flat_150,
            "compounding_5pct": p_comp_5pct,
            "baseline_alloc_75": p_base_alloc,
        },
        "trailing_12m": {
            "months": months,
            "tot_port_pnl": tot_port_pnl,
            "tot_port_r": tot_port_r,
            "winning_months": winning_months,
        },
    }

    out_file = os.path.expanduser("~/.gold_trading_data/backtest_5pct_risk_results.json")
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

    print(f"\n✅ Detailed 5% Risk Backtest Results Stored At: {out_file}")


if __name__ == "__main__":
    main()
