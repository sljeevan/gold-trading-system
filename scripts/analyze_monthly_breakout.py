#!/usr/bin/env python3
"""
Month-on-Month Performance Breakout Analysis for Gold (XAUUSD), Bitcoin (BTCUSD), and Ethereum (ETHUSD).
Evaluates the trailing 12-13 months (October 2025 - October 2026) using the active production engine:
- Strategy: COMPOUNDING_TOUCH (1H Donchian 10 Touch + 50 EMA + SuperTrend)
- Two-Stage Execution: 50% scale-out @ +1.0R, BE shift, Runner Trailing Stop to 3.0R
- Walk-Forward LightGBM Meta-Labeling Filter (P >= 0.44 production threshold, plus strict P >= 0.52 and raw comparison)
- Sizing Mode: FLAT RISK ($75.44 fixed risk per trade on $1,000.00 capital baseline)
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.run_production_backtest import compute_deflated_sharpe_ratio, run_backtest_for_asset


def analyze_monthly_trades(
    symbol: str,
    trades: List[Dict[str, Any]],
    ml_threshold: Optional[float] = 0.44,
    planned_risk_usd: float = 75.44,
    starting_balance: float = 1000.0,
    start_month: str = "2025-10",
    end_month: str = "2026-10",
) -> Dict[str, Any]:
    # 1. Filter by ML threshold if requested
    filtered_trades = trades
    if ml_threshold is not None:
        filtered_trades = [
            t for t in trades if t.get("ml_prob") is not None and t["ml_prob"] >= ml_threshold
        ]

    # 2. Filter by date window (e.g. 2025-10 to 2026-10)
    window_trades = []
    for t in filtered_trades:
        m = t["date"][:7]
        if start_month <= m <= end_month:
            window_trades.append(t)

    # Sort strictly by date/bar_idx
    window_trades.sort(key=lambda x: x["bar_idx"])

    # 3. Group by month
    monthly_groups = defaultdict(list)
    for t in window_trades:
        m = t["date"][:7]
        monthly_groups[m].append(t)

    all_months = sorted(monthly_groups.keys())

    running_balance = starting_balance
    cumulative_pnl = 0.0
    cumulative_r = 0.0
    peak_balance = starting_balance

    monthly_records = []

    for m in all_months:
        m_trades = monthly_groups[m]
        tot = len(m_trades)
        wins = [t for t in m_trades if t["pnl_r"] > 0]
        losses = [t for t in m_trades if t["pnl_r"] <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        wr = (n_wins / tot * 100.0) if tot > 0 else 0.0

        net_r = sum(t["pnl_r"] for t in m_trades)
        net_pts = sum(t["pnl_pts"] for t in m_trades)

        gross_profit_usd = sum(t["pnl_r"] * planned_risk_usd for t in wins)
        gross_loss_usd = abs(sum(t["pnl_r"] * planned_risk_usd for t in losses))
        net_pnl_usd = gross_profit_usd - gross_loss_usd
        pf = (gross_profit_usd / gross_loss_usd) if gross_loss_usd > 0 else (999.0 if gross_profit_usd > 0 else 0.0)

        avg_win_r = (sum(t["pnl_r"] for t in wins) / n_wins) if n_wins > 0 else 0.0
        avg_loss_r = (abs(sum(t["pnl_r"] for t in losses)) / n_losses) if n_losses > 0 else 1.0
        payoff = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0.0

        month_start_balance = running_balance
        month_peak = running_balance
        month_max_dd_usd = 0.0
        month_max_dd_pct = 0.0

        # Calculate intra-month drawdown and update running balance
        for t in m_trades:
            trade_pnl = t["pnl_r"] * planned_risk_usd
            running_balance += trade_pnl
            month_peak = max(month_peak, running_balance)
            dd_usd = month_peak - running_balance
            dd_pct = (dd_usd / month_peak * 100.0) if month_peak > 0 else 0.0
            month_max_dd_usd = max(month_max_dd_usd, dd_usd)
            month_max_dd_pct = max(month_max_dd_pct, dd_pct)

        cumulative_pnl += net_pnl_usd
        cumulative_r += net_r
        peak_balance = max(peak_balance, running_balance)
        month_return_pct = (net_pnl_usd / month_start_balance * 100.0) if month_start_balance > 0 else 0.0

        # Win / loss streaks in month
        cur_consec_w = 0
        max_consec_w = 0
        cur_consec_l = 0
        max_consec_l = 0
        for t in m_trades:
            if t["pnl_r"] > 0:
                cur_consec_w += 1
                cur_consec_l = 0
                max_consec_w = max(max_consec_w, cur_consec_w)
            else:
                cur_consec_l += 1
                cur_consec_w = 0
                max_consec_l = max(max_consec_l, cur_consec_l)

        exit_breakdown = defaultdict(int)
        for t in m_trades:
            exit_breakdown[t.get("exit_type", "UNKNOWN")] += 1

        monthly_records.append({
            "month": m,
            "trades": tot,
            "wins": n_wins,
            "losses": n_losses,
            "win_rate_pct": round(wr, 1),
            "net_r": round(net_r, 2),
            "net_pts": round(net_pts, 2),
            "profit_factor": round(pf, 2),
            "payoff_ratio": round(payoff, 2),
            "gross_profit_usd": round(gross_profit_usd, 2),
            "gross_loss_usd": round(gross_loss_usd, 2),
            "net_pnl_usd": round(net_pnl_usd, 2),
            "month_start_balance": round(month_start_balance, 2),
            "month_end_balance": round(running_balance, 2),
            "month_return_pct": round(month_return_pct, 1),
            "cumulative_pnl_usd": round(cumulative_pnl, 2),
            "cumulative_r": round(cumulative_r, 2),
            "month_max_dd_usd": round(month_max_dd_usd, 2),
            "month_max_dd_pct": round(month_max_dd_pct, 1),
            "max_consec_wins": max_consec_w,
            "max_consec_losses": max_consec_l,
            "exit_breakdown": dict(exit_breakdown),
        })

    # Overall 1-year window aggregate metrics
    tot_trades = len(window_trades)
    tot_wins = sum(r["wins"] for r in monthly_records)
    tot_losses = sum(r["losses"] for r in monthly_records)
    overall_wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    overall_net_r = sum(r["net_r"] for r in monthly_records)
    overall_gross_p = sum(r["gross_profit_usd"] for r in monthly_records)
    overall_gross_l = sum(r["gross_loss_usd"] for r in monthly_records)
    overall_pf = (overall_gross_p / overall_gross_l) if overall_gross_l > 0 else 999.0
    overall_pnl = running_balance - starting_balance
    overall_return_pct = (overall_pnl / starting_balance * 100.0) if starting_balance > 0 else 0.0

    # Max peak-to-trough overall drawdown in the 1-year window
    hwm = starting_balance
    bal = starting_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for t in window_trades:
        bal += t["pnl_r"] * planned_risk_usd
        hwm = max(hwm, bal)
        dd = hwm - bal
        dd_p = (dd / hwm * 100.0) if hwm > 0 else 0.0
        max_dd_usd = max(max_dd_usd, dd)
        max_dd_pct = max(max_dd_pct, dd_p)

    profitable_months = sum(1 for r in monthly_records if r["net_pnl_usd"] > 0)
    loss_months = sum(1 for r in monthly_records if r["net_pnl_usd"] < 0)

    # Clean 12-Month Completed Period Analysis (excluding incomplete MTD)
    c12_records = [r for r in monthly_records if r["month"] < "2026-10"]
    c12_trades = sum(r["trades"] for r in c12_records)
    c12_wins = sum(r["wins"] for r in c12_records)
    c12_losses = sum(r["losses"] for r in c12_records)
    c12_pnl = sum(r["net_pnl_usd"] for r in c12_records)
    c12_r = sum(r["net_r"] for r in c12_records)
    c12_gp = sum(r["gross_profit_usd"] for r in c12_records)
    c12_gl = sum(r["gross_loss_usd"] for r in c12_records)
    c12_pf = (c12_gp / c12_gl) if c12_gl > 0 else 999.0
    c12_wr = (c12_wins / c12_trades * 100.0) if c12_trades > 0 else 0.0

    # Incomplete October 2026 MTD
    mtd_records = [r for r in monthly_records if r["month"] == "2026-10"]
    mtd_summary = mtd_records[0] if mtd_records else {}

    # Deflated Sharpe Ratio (50 trial correction)
    trade_pnls = [t["pnl_r"] * planned_risk_usd for t in window_trades]
    trade_returns = [pnl / starting_balance for pnl in trade_pnls]
    dsr_metrics = compute_deflated_sharpe_ratio(trade_returns, num_trials=50)

    summary = {
        "symbol": symbol,
        "ml_threshold": ml_threshold,
        "planned_risk_usd": planned_risk_usd,
        "starting_balance": starting_balance,
        "final_balance": round(running_balance, 2),
        "total_net_pnl_usd": round(overall_pnl, 2),
        "total_return_pct": round(overall_return_pct, 1),
        "total_trades": tot_trades,
        "total_wins": tot_wins,
        "total_losses": tot_losses,
        "win_rate_pct": round(overall_wr, 1),
        "profit_factor": round(overall_pf, 2),
        "total_net_r": round(overall_net_r, 1),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "profitable_months": profitable_months,
        "loss_months": loss_months,
        "month_win_rate_pct": round(profitable_months / len(monthly_records) * 100.0, 1) if monthly_records else 0.0,
        "completed_12m": {
            "period": "2025-10 to 2026-09",
            "trades": c12_trades,
            "wins": c12_wins,
            "losses": c12_losses,
            "win_rate_pct": round(c12_wr, 1),
            "net_r": round(c12_r, 1),
            "profit_factor": round(c12_pf, 2),
            "net_pnl_usd": round(c12_pnl, 2),
            "return_pct": round((c12_pnl / starting_balance) * 100.0, 1),
            "profitable_months": sum(1 for r in c12_records if r["net_pnl_usd"] > 0),
            "total_months": len(c12_records),
        },
        "mtd_partial": {
            "period": "2026-10 MTD (20 days)",
            "trades": mtd_summary.get("trades", 0),
            "wins": mtd_summary.get("wins", 0),
            "losses": mtd_summary.get("losses", 0),
            "win_rate_pct": mtd_summary.get("win_rate_pct", 0.0),
            "net_r": mtd_summary.get("net_r", 0.0),
            "profit_factor": mtd_summary.get("profit_factor", 0.0),
            "net_pnl_usd": mtd_summary.get("net_pnl_usd", 0.0),
        },
        "dsr_metrics": dsr_metrics,
        "monthly_records": monthly_records,
    }
    return summary


def main():
    print("=" * 110)
    print("🚀 GENERATING 1-YEAR MONTH-ON-MONTH BACKTEST BREAKOUT ACROSS GOLD, BITCOIN & ETHEREUM")
    print("=" * 110)

    # 1. Run backtest data extraction for all three assets
    print("\n[1/3] Extracting Walk-Forward Data for Spot Gold (XAUUSD)...")
    gold_data = run_backtest_for_asset(
        symbol="XAUUSD",
        sl_atr_mult=1.5,
        min_sl_dist=8.00,
        tsl_buffer=4.50,
        tp2_mult=3.00,
        be_offset=0.50,
        session_filter=True,
    )

    print("\n[2/3] Extracting Walk-Forward Data for Bitcoin (BTCUSD)...")
    btc_data = run_backtest_for_asset(
        symbol="BTCUSD",
        sl_atr_mult=1.5,
        min_sl_dist=500.00,
        tsl_buffer=350.00,
        tp2_mult=3.00,
        be_offset=30.00,
        session_filter=False,
    )

    print("\n[3/3] Extracting Walk-Forward Data for Ethereum (ETHUSD)...")
    eth_data = run_backtest_for_asset(
        symbol="ETHUSD",
        sl_atr_mult=1.5,
        min_sl_dist=20.00,
        tsl_buffer=10.00,
        tp2_mult=3.00,
        be_offset=2.00,
        session_filter=True,  # Live production configuration
    )

    # Also extract ETH 24/7 for comparison
    eth_247_data = run_backtest_for_asset(
        symbol="ETHUSD",
        sl_atr_mult=1.5,
        min_sl_dist=20.00,
        tsl_buffer=10.00,
        tp2_mult=3.00,
        be_offset=2.00,
        session_filter=False,
    )

    # 2. Analyze monthly breakout for live production configuration (ML P >= 0.44)
    start_m = "2025-10"
    end_m = "2026-10"

    gold_mom = analyze_monthly_trades("XAUUSD", gold_data["all_trades"], ml_threshold=0.44, start_month=start_m, end_month=end_m)
    btc_mom = analyze_monthly_trades("BTCUSD", btc_data["all_trades"], ml_threshold=0.44, start_month=start_m, end_month=end_m)
    eth_mom = analyze_monthly_trades("ETHUSD", eth_data["all_trades"], ml_threshold=0.44, start_month=start_m, end_month=end_m)
    eth_247_mom = analyze_monthly_trades("ETHUSD_247", eth_247_data["all_trades"], ml_threshold=0.44, start_month=start_m, end_month=end_m)

    # 3. Combined Multi-Asset Portfolio Monthly Aggregation
    all_months = sorted(list(set(
        [r["month"] for r in gold_mom["monthly_records"]] +
        [r["month"] for r in btc_mom["monthly_records"]] +
        [r["month"] for r in eth_mom["monthly_records"]]
    )))

    portfolio_records = []
    port_balance = 3000.0  # $1,000 per asset = $3,000 blended baseline
    port_cumulative_pnl = 0.0

    gold_by_m = {r["month"]: r for r in gold_mom["monthly_records"]}
    btc_by_m = {r["month"]: r for r in btc_mom["monthly_records"]}
    eth_by_m = {r["month"]: r for r in eth_mom["monthly_records"]}

    for m in all_months:
        g = gold_by_m.get(m, {})
        b = btc_by_m.get(m, {})
        e = eth_by_m.get(m, {})

        m_trades = g.get("trades", 0) + b.get("trades", 0) + e.get("trades", 0)
        m_wins = g.get("wins", 0) + b.get("wins", 0) + e.get("wins", 0)
        m_losses = g.get("losses", 0) + b.get("losses", 0) + e.get("losses", 0)
        m_wr = (m_wins / m_trades * 100.0) if m_trades > 0 else 0.0

        m_net_r = round(g.get("net_r", 0.0) + b.get("net_r", 0.0) + e.get("net_r", 0.0), 2)
        m_pnl = round(g.get("net_pnl_usd", 0.0) + b.get("net_pnl_usd", 0.0) + e.get("net_pnl_usd", 0.0), 2)

        start_bal = port_balance
        port_balance += m_pnl
        port_cumulative_pnl += m_pnl
        m_ret_pct = (m_pnl / start_bal * 100.0) if start_bal > 0 else 0.0

        m_gp = g.get("gross_profit_usd", 0.0) + b.get("gross_profit_usd", 0.0) + e.get("gross_profit_usd", 0.0)
        m_gl = g.get("gross_loss_usd", 0.0) + b.get("gross_loss_usd", 0.0) + e.get("gross_loss_usd", 0.0)
        m_pf = (m_gp / m_gl) if m_gl > 0 else 999.0

        portfolio_records.append({
            "month": m,
            "trades": m_trades,
            "wins": m_wins,
            "losses": m_losses,
            "win_rate_pct": round(m_wr, 1),
            "net_r": m_net_r,
            "profit_factor": round(m_pf, 2),
            "net_pnl_usd": m_pnl,
            "gold_pnl_usd": g.get("net_pnl_usd", 0.0),
            "btc_pnl_usd": b.get("net_pnl_usd", 0.0),
            "eth_pnl_usd": e.get("net_pnl_usd", 0.0),
            "start_balance": round(start_bal, 2),
            "end_balance": round(port_balance, 2),
            "return_pct": round(m_ret_pct, 1),
            "cumulative_pnl_usd": round(port_cumulative_pnl, 2),
        })

    # Save output to JSON
    output_path = os.path.expanduser("~/.gold_trading_data/monthly_breakout_1year_results.json")
    full_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_month": start_m,
        "end_month": end_m,
        "gold": gold_mom,
        "bitcoin": btc_mom,
        "ethereum_session": eth_mom,
        "ethereum_247": eth_247_mom,
        "portfolio_combined": {
            "starting_balance": 3000.0,
            "final_balance": round(port_balance, 2),
            "net_pnl_usd": round(port_cumulative_pnl, 2),
            "total_return_pct": round((port_cumulative_pnl / 3000.0) * 100.0, 1),
            "total_trades": sum(r["trades"] for r in portfolio_records),
            "total_wins": sum(r["wins"] for r in portfolio_records),
            "total_losses": sum(r["losses"] for r in portfolio_records),
            "win_rate_pct": round(sum(r["wins"] for r in portfolio_records) / sum(r["trades"] for r in portfolio_records) * 100.0, 1),
            "monthly_records": portfolio_records,
        }
    }

    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n✅ Monthly Breakout Results Stored At: {output_path}")


if __name__ == "__main__":
    main()
