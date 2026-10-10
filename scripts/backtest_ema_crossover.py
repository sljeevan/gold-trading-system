#!/usr/bin/env python3
"""
CLI script to backtest EMA Crossover Strategy (Long & Short) with configurable targets.
Usage:
    python scripts/backtest_ema_crossover.py --symbol XAUUSD --timeframe 1h --period 1y
    python scripts/backtest_ema_crossover.py --symbol XAUUSD --timeframe 15m --period 60d --target 5 --sl 5
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from gold_trader.core.backtester import backtest_ema_crossover_strategy


def print_banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA Crossover Backtester for Gold (Long & Short)")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol (e.g. XAUUSD, GC, GLD)")
    parser.add_argument("--fast", type=int, default=5, help="Fast EMA period (default: 5)")
    parser.add_argument("--slow", type=int, default=20, help="Slow EMA period (default: 20)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Interval: 5m, 15m, 1h, 1d (default: 1h)")
    parser.add_argument("--period", type=str, default="1y", help="Period: 1mo, 3mo, 6mo, 1y, 2y (default: 1y)")
    parser.add_argument("--target", type=float, default=None, help="Minimum target points to exit (Take Profit)")
    parser.add_argument("--sl", type=float, default=None, help="Stop loss in points")
    parser.add_argument("--be", type=float, default=None, help="Breakeven trigger in points (moves SL to entry+spread)")
    parser.add_argument("--trail", type=float, default=None, help="Trailing stop distance in points behind peak")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USD (default: 10000)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    results = backtest_ema_crossover_strategy(
        symbol=args.symbol,
        fast_period=args.fast,
        slow_period=args.slow,
        min_target_points=args.target,
        stop_loss_points=args.sl,
        be_trigger_points=args.be,
        trailing_stop_points=args.trail,
        period=args.period,
        interval=args.timeframe,
        initial_capital=args.capital,
    )

    if "error" in results:
        print(f"Error: {results['error']}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    m = results["metrics"]
    tgt_desc = f"{args.target} pts" if args.target is not None else "Opposite Crossover (Pure Reversal)"
    sl_desc = f"{args.sl} pts" if args.sl is not None else "None (Opposite Cross Exit)"
    be_desc = f"+{args.be} pts" if args.be is not None else "Disabled"
    trail_desc = f"{args.trail} pts" if args.trail is not None else "Disabled"

    print_banner(f"EMA {args.fast} / {args.slow} CROSSOVER BACKTEST — {results['symbol']} ({args.timeframe.upper()})")
    print(f" • Period Analyzed     : {args.period} ({results['candles_analyzed']} candles)")
    print(f" • Target Exit Rule    : {tgt_desc}")
    print(f" • Stop-Loss Rule      : {sl_desc}")
    print(f" • Breakeven Trigger   : {be_desc}")
    print(f" • Trailing Stop       : {trail_desc}")
    print(f" • Buy & Hold Return   : {results['buy_and_hold_return_pct']:+.2f}%")
    print(f" • Transaction Costs   : Standard XM Gold spread deducted (3.5 pips = $0.35/oz)")

    print_banner("PERFORMANCE SUMMARY (LONG + SHORT)")
    print(f" │ Initial Capital      : ${args.capital:,.2f}")
    print(f" │ Final Capital        : ${m['final_capital']:,.2f}")
    print(f" │ Net Return           : {m['total_return_pct']:+.2f}%")
    print(f" │ Net Points Captured  : {m['total_points']:+.2f} pts (${m['total_points']*100:+,.2f} on 1 lot)")
    print(f" │ Total Trades         : {m['total_trades']}")
    print(f" │ Winning Trades       : {m['winning_trades']} ({m['win_rate_pct']}%)")
    print(f" │ Losing Trades        : {m['losing_trades']}")
    print(f" │ Profit Factor        : {m['profit_factor']}")
    print(f" │ Sharpe Ratio         : {m['sharpe_ratio']}")
    print(f" │ Calmar Ratio         : {m['calmar_ratio']}")
    print(f" │ Max Drawdown         : {m['max_drawdown_pct']:.2f}%")
    print(f" │ Avg Points / Trade   : {m['avg_points_per_trade']:+.2f} pts")
    print(f" │ Best Trade           : {m['max_win_points']:+.2f} pts")
    print(f" │ Worst Trade          : {m['max_loss_points']:+.2f} pts")

    print_banner("DIRECTIONAL BREAKDOWN (LONG vs SHORT)")
    print(f" │ {'Direction':<8} │ {'Trades':<8} │ {'Win Rate':<10} │ {'Net Points':<12} │ {'Avg Points/Trade':<16} │")
    print(f" ├──────────┼──────────┼────────────┼──────────────┼──────────────────┤")
    long_avg = m['long_points'] / m['long_trades'] if m['long_trades'] else 0.0
    short_avg = m['short_points'] / m['short_trades'] if m['short_trades'] else 0.0
    print(f" │ {'LONG':<8} │ {m['long_trades']:<8} │ {m['long_win_rate_pct']:>8.1f}% │ {m['long_points']:>10.2f} pts │ {long_avg:>14.2f} pts │")
    print(f" │ {'SHORT':<8} │ {m['short_trades']:<8} │ {m['short_win_rate_pct']:>8.1f}% │ {m['short_points']:>10.2f} pts │ {short_avg:>14.2f} pts │")

    if results.get("mfe_distribution"):
        print_banner("MFE DISTRIBUTION — CHANCE OF CAPTURING MINIMUM POINTS BEFORE REVERSAL")
        print(" (Calculates % of crossover entries that hit at least +X points favorable move)")
        print(f" │ {'Min Points Threshold':<22} │ {'Trades Reached':<16} │ {'Success Probability':<20} │")
        print(f" ├────────────────────────┼──────────────────┼──────────────────────┤")
        for key, stats in results["mfe_distribution"].items():
            pts_label = f"+{key.replace('_pts', ' Points')}"
            print(f" │ {pts_label:<22} │ {stats['trades_reached']:>7} / {m['total_trades']:<6} │ {stats['pct_reached']:>18.1f}% │")

    if results.get("target_sensitivity_analysis"):
        print_banner("TARGET SENSITIVITY TABLE (WHAT HAPPENS AT EACH MINIMUM TARGET LEVEL)")
        print(f" │ {'Min Target':<12} │ {'Win Rate':<10} │ {'Target Hits':<12} │ {'Net Points':<12} │ {'Net Return':<12} │ {'Profit Factor':<14} │")
        print(f" ├──────────────┼────────────┼──────────────┼──────────────┼──────────────┼────────────────┤")
        for row in results["target_sensitivity_analysis"]:
            tgt_lbl = f"+{row['target_points']} pts"
            print(f" │ {tgt_lbl:<12} │ {row['win_rate_pct']:>8.1f}% │ {row['targets_hit']:>5} / {row['total_trades']:<4} │ {row['total_points']:>10.2f} │ {row['total_return_pct']:>10.2f}% │ {row['profit_factor']:>14.2f} │")

    if results.get("recent_trades"):
        print_banner("RECENT EXECUTED TRADES (LAST 10)")
        print(f" │ {'Entry Date':<16} │ {'Type':<5} │ {'Entry':<8} │ {'Exit Date':<16} │ {'Exit':<8} │ {'Reason':<14} │ {'Net Pts':<9} │")
        print(f" ├──────────────────┼───────┼──────────┼──────────────────┼──────────┼────────────────┼───────────┤")
        for t in results["recent_trades"]:
            dt_in = t["entry_date"][:16]
            dt_out = t.get("exit_date", "")[:16]
            side = t.get("direction", "LONG")
            ep = t["entry_price"]
            xp = t.get("exit_price", 0.0)
            reason = t.get("exit_reason", "cross")[:14]
            pts = t.get("pnl_points", 0.0) - 0.35
            print(f" │ {dt_in:<16} │ {side:<5} │ {ep:>8.2f} │ {dt_out:<16} │ {xp:>8.2f} │ {reason:<14} │ {pts:>+8.2f}  │")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
