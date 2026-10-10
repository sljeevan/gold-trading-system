#!/usr/bin/env python3
"""
CLI Training Script for Gold PPO Reinforcement Learning Agent.
Runs independently without affecting existing auto-trader or rule engines.

Usage:
    python3 scripts/train_gold_rl.py --symbol XAUUSD --period 2y --interval 1h --timesteps 30000
"""
import argparse
import logging
import sys

from gold_trader.rl.trainer import GoldRLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrainGoldRL")


def main():
    parser = argparse.ArgumentParser(description="Train PPO RL Agent for Gold Trading")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to train on (XAUUSD or GC=F)")
    parser.add_argument("--period", default="2y", help="Historical data period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--interval", default="1h", help="Bar interval (e.g. 15m, 1h)")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total PPO training timesteps")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🏆 Institutional Gold PPO Reinforcement Learning Training")
    print("=" * 60)
    print(f"Symbol:     {args.symbol}")
    print(f"Period:     {args.period}")
    print(f"Interval:   {args.interval}")
    print(f"Timesteps:  {args.timesteps}")
    print("=" * 60 + "\n")

    trainer = GoldRLTrainer()
    results = trainer.run_training_pipeline(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if "error" in results:
        print(f"❌ Training Failed: {results['error']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Training & Out-of-Sample Evaluation Complete!")
    print("=" * 60)
    metrics = results.get("test_metrics", {})
    print(f"Out-of-Sample Return:  {metrics.get('total_return_pct', 0.0):+.2f}%")
    print(f"Out-of-Sample Sharpe:  {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"Out-of-Sample Max DD:  {metrics.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"Win Rate:              {metrics.get('win_rate_pct', 0.0):.2f}%")
    print(f"Total Trades:          {metrics.get('total_trades', 0)}")
    print(f"Profit Factor:         {metrics.get('profit_factor', 0.0):.2f}")
    print(f"Model Checkpoint:      {results.get('model_path')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
