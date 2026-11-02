#!/usr/bin/env python3
"""
Rigorous 3-Year Backtest Comparison: Model 1 (Production Baseline) vs. Model 2 (Profile A Yield Engine)
Evaluation Horizon: October 26, 2023 00:00 to October 26, 2026 12:00 (Exact 36 Months / 3 Years)
Assets: Bitcoin (BTCUSD) & Ethereum (ETHUSD), plus Gold (XAUUSD) for Model 1 portfolio context.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import scipy.stats as stats

from scripts.run_production_backtest_5years import (
    ASSET_CONFIG,
    filter_concurrency,
    load_5y_market_data,
)
from scripts.run_grid_search_optimization import scan_and_train_asset_tsl, TSL_REGIMES

START_3Y = "2023-10-26 00:00"
END_3Y = "2026-10-26 12:00"

def evaluate_performance(
    trades: List[Dict[str, Any]],
    starting_balance: float,
    risk_usd: float,
    label: str,
) -> Dict[str, Any]:
    if not trades:
        return {"label": label, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl_usd": 0.0}

    total_trades = len(trades)
    pnls_usd = [t["pnl_r"] * risk_usd for t in trades]
    pnl_r_list = [t["pnl_r"] for t in trades]
    wins = [p for p in pnls_usd if p > 0]
    losses = [p for p in pnls_usd if p < 0]
    scratches = [p for p in pnls_usd if p == 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    net_pnl_usd = gross_profit - gross_loss
    net_r = sum(pnl_r_list)

    avg_win_usd = np.mean(wins) if wins else 0.0
    avg_loss_usd = abs(np.mean(losses)) if losses else 0.0
    payoff_ratio = avg_win_usd / avg_loss_usd if avg_loss_usd > 0 else 0.0

    # Path simulation for drawdown
    equity = starting_balance
    peak_equity = starting_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    equity_curve = [starting_balance]

    for pnl in pnls_usd:
        equity += pnl
        equity = max(0.0, equity)
        peak_equity = max(peak_equity, equity)
        dd_usd = peak_equity - equity
        dd_pct = (dd_usd / peak_equity * 100.0) if peak_equity > 0 else 0.0
        max_dd_usd = max(max_dd_usd, dd_usd)
        max_dd_pct = max(max_dd_pct, dd_pct)
        equity_curve.append(equity)

    # Sharpe ratio (annualized)
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-6
    trades_per_year = total_trades / 3.0
    sharpe = (mean_ret / std_ret) * math.sqrt(trades_per_year) if std_ret > 1e-6 else 0.0

    # Calmar ratio
    annualized_return_pct = ((equity - starting_balance) / starting_balance * 100.0) / 3.0 if starting_balance > 0 else 0.0
    calmar = (annualized_return_pct / max_dd_pct) if max_dd_pct > 0 else 99.0

    # Exit counts
    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t.get("exit_type", "OTHER")] += 1

    return {
        "label": label,
        "total_trades": total_trades,
        "win_trades": win_count,
        "loss_trades": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 2),
        "net_r": round(net_r, 2),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "starting_balance": starting_balance,
        "final_balance": round(equity, 2),
        "total_return_pct": round(((equity - starting_balance) / starting_balance) * 100.0, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "avg_win_usd": round(avg_win_usd, 2),
        "avg_loss_usd": round(avg_loss_usd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "calmar_ratio": round(calmar, 2),
        "exit_counts": dict(exit_counts),
    }

def simulate_monthly_harvest(trades: List[Dict[str, Any]], risk_usd: float) -> Dict[str, Any]:
    monthly_trades = defaultdict(list)
    for t in trades:
        m_key = t["entry_date"][:7]
        monthly_trades[m_key].append(t)

    all_months = sorted(monthly_trades.keys())
    total_withdrawn = 0.0
    total_topped_up = 0.0
    profitable_months = 0
    deficit_months = 0
    monthly_details = []

    for m_key in all_months:
        t_list = monthly_trades[m_key]
        m_pnl = sum(risk_usd * t["pnl_r"] for t in t_list)
        wins = len([t for t in t_list if t["pnl_r"] > 0])
        wr = (wins / len(t_list) * 100.0) if t_list else 0.0

        if m_pnl > 0:
            withdrawn = m_pnl
            topup = 0.0
            total_withdrawn += withdrawn
            profitable_months += 1
        else:
            withdrawn = 0.0
            topup = abs(m_pnl)
            total_topped_up += topup
            deficit_months += 1

        monthly_details.append({
            "month": m_key,
            "trades": len(t_list),
            "win_rate": round(wr, 1),
            "net_pnl": round(m_pnl, 2),
            "withdrawn": round(withdrawn, 2),
            "topup": round(topup, 2),
        })

    net_cashflow = total_withdrawn - total_topped_up
    avg_monthly = net_cashflow / len(all_months) if all_months else 0.0
    win_rate_months = (profitable_months / len(all_months) * 100.0) if all_months else 0.0

    return {
        "total_months": len(all_months),
        "profitable_months": profitable_months,
        "deficit_months": deficit_months,
        "monthly_win_rate_pct": round(win_rate_months, 1),
        "total_harvested_usd": round(total_withdrawn, 2),
        "total_topped_up_usd": round(total_topped_up, 2),
        "net_cashflow_usd": round(net_cashflow, 2),
        "avg_monthly_harvest_usd": round(avg_monthly, 2),
        "monthly_details": monthly_details,
    }

def main():
    print("=" * 80)
    print("AEGIS ALPHA: 3-YEAR BACKTEST AUDIT — MODEL 1 vs. MODEL 2")
    print(f"Period: {START_3Y} to {END_3Y} (36 Months / 3 Years)")
    print("=" * 80)

    # 1. Load Market Data
    print("\n[Step 1] Loading 5-year cached data...")
    market_data = {
        "BTCUSD": load_5y_market_data("BTCUSD"),
        "ETHUSD": load_5y_market_data("ETHUSD"),
        "XAUUSD": load_5y_market_data("XAUUSD"),
    }

    # 2. Generate Candidate Trades for Model 1 (Baseline Medium TSL)
    print("\n[Step 2] Scanning & ML Meta-Labeling Model 1 Baseline Trades...")
    m1_raw = {}
    for sym in ["BTCUSD", "ETHUSD", "XAUUSD"]:
        tsl = TSL_REGIMES["Medium"][sym]
        s_filt = ASSET_CONFIG[sym]["session_filter"]
        print(f"  • Model 1 {sym} (TSL: {tsl}, SessionFilter: {s_filt})...")
        trades = scan_and_train_asset_tsl(sym, market_data[sym], tsl, session_filter=s_filt)
        # Filter for 3Y window
        trades_3y = [t for t in trades if START_3Y <= t["entry_date"] <= END_3Y]
        m1_raw[sym] = trades_3y
        print(f"    -> {len(trades_3y)} 3-year trades generated.")

    # 3. Generate Candidate Trades for Model 2 (Profile A Tight TSL)
    print("\n[Step 3] Scanning & ML Meta-Labeling Model 2 (Profile A Tight TSL) Trades...")
    m2_raw = {}
    for sym in ["BTCUSD", "ETHUSD"]:
        tsl = TSL_REGIMES["Tight"][sym]
        print(f"  • Model 2 {sym} (TSL: {tsl}, SessionFilter: False)...")
        trades = scan_and_train_asset_tsl(sym, market_data[sym], tsl, session_filter=False)
        # Filter for 3Y window
        trades_3y = [t for t in trades if START_3Y <= t["entry_date"] <= END_3Y]
        m2_raw[sym] = trades_3y
        print(f"    -> {len(trades_3y)} 3-year trades generated.")

    # 4. Apply Model Rules
    # Model 1: ML >= 0.44 (or 0.50), Uncapped Concurrency, Risk $75.44
    m1_btc_trades = [t for t in m1_raw["BTCUSD"] if t.get("ml_prob", 0) >= 0.44]
    m1_eth_trades = [t for t in m1_raw["ETHUSD"] if t.get("ml_prob", 0) >= 0.44]
    m1_xau_trades = [t for t in m1_raw["XAUUSD"] if t.get("ml_prob", 0) >= 0.44]
    m1_crypto_blended = sorted(m1_btc_trades + m1_eth_trades, key=lambda x: x["entry_date"])
    m1_all_blended = sorted(m1_btc_trades + m1_eth_trades + m1_xau_trades, key=lambda x: x["entry_date"])

    # Model 2: ML >= 0.48, Max 1 Position per Asset, Risk $75.00
    m2_btc_trades = filter_concurrency([t for t in m2_raw["BTCUSD"] if t.get("ml_prob", 0) >= 0.48])
    m2_eth_trades = filter_concurrency([t for t in m2_raw["ETHUSD"] if t.get("ml_prob", 0) >= 0.48])
    m2_crypto_blended = sorted(m2_btc_trades + m2_eth_trades, key=lambda x: x["entry_date"])

    # 5. Evaluate Metrics
    print("\n" + "=" * 80)
    print("3-YEAR PERFORMANCE RESULTS")
    print("=" * 80)

    # Asset-Level Evaluations
    m1_btc_perf = evaluate_performance(m1_btc_trades, 1000.0, 75.44, "Model 1 BTCUSD")
    m2_btc_perf = evaluate_performance(m2_btc_trades, 750.0, 75.00, "Model 2 BTCUSD (Profile A)")

    m1_eth_perf = evaluate_performance(m1_eth_trades, 1000.0, 75.44, "Model 1 ETHUSD")
    m2_eth_perf = evaluate_performance(m2_eth_trades, 750.0, 75.00, "Model 2 ETHUSD (Profile A)")

    m1_xau_perf = evaluate_performance(m1_xau_trades, 1000.0, 75.44, "Model 1 XAUUSD")

    # Crypto Portfolio Comparison (BTC + ETH)
    m1_crypto_perf = evaluate_performance(m1_crypto_blended, 2000.0, 75.44, "Model 1 Crypto Portfolio (BTC+ETH)")
    m2_crypto_perf = evaluate_performance(m2_crypto_blended, 1500.0, 75.00, "Model 2 Crypto Portfolio (BTC+ETH)")

    # Full Portfolio for Model 1 (Gold + BTC + ETH)
    m1_full_perf = evaluate_performance(m1_all_blended, 3000.0, 75.44, "Model 1 Full Portfolio (Gold+BTC+ETH)")

    # Model 2 Monthly Harvest Analysis
    m2_harvest = simulate_monthly_harvest(m2_crypto_blended, 75.00)

    output = {
        "period": {"start": START_3Y, "end": END_3Y, "months": 36},
        "m1_btc": m1_btc_perf,
        "m2_btc": m2_btc_perf,
        "m1_eth": m1_eth_perf,
        "m2_eth": m2_eth_perf,
        "m1_xau": m1_xau_perf,
        "m1_crypto_portfolio": m1_crypto_perf,
        "m2_crypto_portfolio": m2_crypto_perf,
        "m1_full_portfolio": m1_full_perf,
        "m2_harvest": m2_harvest,
    }

    save_path = "/home/mcpuser/.gold_trading_data/model1_vs_model2_3years_comparison.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Saved] Complete comparison saved to {save_path}")

    # Print summary tables
    print("\n--- 1. BTCUSD HEAD-TO-HEAD (3 YEARS) ---")
    print(f"{'Metric':<25} | {'Model 1 Baseline':<20} | {'Model 2 Profile A':<20}")
    print("-" * 71)
    for k in ["total_trades", "win_rate_pct", "profit_factor", "net_r", "net_pnl_usd", "max_drawdown_usd", "max_drawdown_pct", "payoff_ratio", "calmar_ratio", "sharpe_ratio"]:
        v1 = m1_btc_perf.get(k)
        v2 = m2_btc_perf.get(k)
        print(f"{k:<25} | {str(v1):<20} | {str(v2):<20}")

    print("\n--- 2. ETHUSD HEAD-TO-HEAD (3 YEARS) ---")
    print(f"{'Metric':<25} | {'Model 1 Baseline':<20} | {'Model 2 Profile A':<20}")
    print("-" * 71)
    for k in ["total_trades", "win_rate_pct", "profit_factor", "net_r", "net_pnl_usd", "max_drawdown_usd", "max_drawdown_pct", "payoff_ratio", "calmar_ratio", "sharpe_ratio"]:
        v1 = m1_eth_perf.get(k)
        v2 = m2_eth_perf.get(k)
        print(f"{k:<25} | {str(v1):<20} | {str(v2):<20}")

    print("\n--- 3. BLENDED CRYPTO PORTFOLIO (BTC + ETH) ---")
    print(f"{'Metric':<25} | {'Model 1 Baseline':<20} | {'Model 2 Profile A':<20}")
    print("-" * 71)
    for k in ["total_trades", "win_rate_pct", "profit_factor", "net_r", "net_pnl_usd", "max_drawdown_usd", "max_drawdown_pct", "payoff_ratio", "calmar_ratio", "sharpe_ratio"]:
        v1 = m1_crypto_perf.get(k)
        v2 = m2_crypto_perf.get(k)
        print(f"{k:<25} | {str(v1):<20} | {str(v2):<20}")

    print("\n--- 4. MODEL 2 MONTHLY TREASURY HARVEST STATS ---")
    print(f"Total Months: {m2_harvest['total_months']}")
    print(f"Profitable Months: {m2_harvest['profitable_months']} / {m2_harvest['total_months']} ({m2_harvest['monthly_win_rate_pct']}%)")
    print(f"Total Cash Harvested: ${m2_harvest['total_harvested_usd']:,.2f}")
    print(f"Total Top-Ups Required: ${m2_harvest['total_topped_up_usd']:,.2f}")
    print(f"Net Cashflow: ${m2_harvest['net_cashflow_usd']:,.2f}")
    print(f"Average Monthly Payout: ${m2_harvest['avg_monthly_harvest_usd']:,.2f} / month")

if __name__ == "__main__":
    main()
