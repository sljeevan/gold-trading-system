#!/usr/bin/env python3
"""
Backtester & Benchmark Script for Trained Gold PPO Agent.
Evaluates the model against buy-and-hold and outputs trade execution logs.

Usage:
    python3 scripts/backtest_gold_rl.py --symbol XAUUSD --period 6mo --interval 1h
"""
import argparse
import logging
import sys

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.macro import analyze_macro_drivers
from gold_trader.rl.data_processor import GoldRLDataProcessor
from gold_trader.rl.gold_env import GoldTradingEnv
from gold_trader.rl.predictor import GoldRLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestGoldRL")


def main():
    parser = argparse.ArgumentParser(description="Backtest Gold PPO Policy")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol (XAUUSD or GC=F)")
    parser.add_argument("--period", default="6mo", help="Period (e.g. 3mo, 6mo, 1y)")
    parser.add_argument("--interval", default="1h", help="Interval (15m, 1h)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance ($)")
    parser.add_argument("--spread", type=float, default=2.5, help="Spread pips")
    args = parser.parse_args()

    predictor = GoldRLPredictor()
    if not predictor.is_ready:
        print("❌ No trained Gold PPO model found. Run 'scripts/train_gold_rl.py' first.")
        sys.exit(1)

    print(f"\nFetching {args.period} {args.interval} data for {args.symbol}...")
    candles = fetch_gold_candles(args.symbol, period=args.period, interval=args.interval)
    if len(candles) < 250:
        print(f"❌ Need at least 250 candles, got {len(candles)}")
        sys.exit(1)

    macro_data = analyze_macro_drivers()
    macro_score = float(macro_data.get("macro_gold_score", 0.0))

    processor = GoldRLDataProcessor()
    processor.scaler = predictor.processor.scaler

    X, prices, valid_candles = processor.build_features_from_candles(
        candles, macro_score=macro_score, fit_scaler=False
    )

    env = GoldTradingEnv(
        features=X,
        prices=prices,
        initial_balance=args.balance,
        spread_pips=args.spread,
    )

    obs, _ = env.reset()
    done = False
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    while not done:
        action, _ = predictor.model.predict(obs, deterministic=True)
        action_counts[int(action)] = action_counts.get(int(action), 0) + 1
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    metrics = env._get_episode_metrics()

    # Calculate Buy & Hold benchmark
    start_price = float(prices[0, 3])
    end_price = float(prices[-1, 3])
    bnh_return = ((end_price - start_price) / start_price) * 100.0

    print("\n" + "=" * 65)
    print(f"📊 BACKTEST RESULTS: Gold PPO Agent ({args.symbol} {args.period} {args.interval})")
    print("=" * 65)
    print(f"Initial Balance:        ${metrics['initial_balance']:,.2f}")
    print(f"Final Equity:           ${metrics['final_equity']:,.2f}")
    print(f"Total Return:           {metrics['total_return_pct']:+.2f}%")
    print(f"Buy & Hold Return:      {bnh_return:+.2f}%")
    print(f"Alpha vs Benchmark:     {metrics['total_return_pct'] - bnh_return:+.2f}%")
    print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate:               {metrics['win_rate_pct']:.1f}%")
    print(f"Profit Factor:          {metrics['profit_factor']:.2f}")
    print(f"Total Completed Trades: {metrics['total_trades']}")
    print("-" * 65)
    print(
        f"Action Distribution:    HOLD: {action_counts[0]} | BUY: {action_counts[1]} | SELL: {action_counts[2]} | CLOSE: {action_counts[3]}"
    )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
