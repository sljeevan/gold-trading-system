#!/usr/bin/env python3
"""
Evaluation and Benchmarking Script for Gold PPO Trading Agent.
Loads a trained PPO model, evaluates it against historical Gold market data,
and produces comprehensive risk-adjusted metrics, trade logs, and drawdown profiles.

Usage:
    python3 scripts/evaluate_gold_rl.py --model models/gold_ppo_agent.pt --timeframe 1h --period 6mo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure src/ is in python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.macro import analyze_macro_drivers
from gold_trader.rl.data_processor import GoldRLDataProcessor
from gold_trader.rl.gold_env import GoldTradingEnv
from gold_trader.rl.predictor import GoldRLPredictor
from gold_trader.rl.trainer import DEFAULT_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_gold_rl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gold PPO Agent")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, "latest_gold_ppo.zip"),
        help="Path to saved PPO policy (.zip)",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="",
        help="Optional path to saved scaler (.pkl). Defaults to scaler.pkl in model's directory.",
    )
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        choices=["5m", "15m", "30m", "1h", "1d"],
        help="Bar timeframe",
    )
    parser.add_argument("--period", type=str, default="6mo", help="Data range (e.g., 1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial balance in USD")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Risk per trade percentage")
    parser.add_argument("--spread-pips", type=float, default=2.5, help="Broker spread in pips")
    parser.add_argument("--output-trades", type=str, default="", help="Optional path to save trades JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("================================================================")
    logger.info("📈 Running Gold RL PPO Strategy Evaluation & Backtest")
    logger.info("================================================================")
    logger.info(f"Model: {args.model} | Symbol: {args.symbol} | Timeframe: {args.timeframe} | Period: {args.period}")

    scaler_path = args.scaler
    if not scaler_path:
        model_dir = os.path.dirname(os.path.expanduser(args.model))
        candidate_scaler = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(candidate_scaler):
            scaler_path = candidate_scaler
        else:
            scaler_path = os.path.join(DEFAULT_MODEL_DIR, "scaler.pkl")

    predictor = GoldRLPredictor(model_path=args.model, scaler_path=scaler_path)
    if not predictor.is_ready:
        logger.error(
            f"Failed to load Gold PPO model or scaler. (Model: {args.model}, Scaler: {scaler_path}). "
            "Please run 'python3 scripts/train_gold_rl.py' first."
        )
        sys.exit(1)

    # 1. Fetch historical candles
    logger.info(f"Fetching candles for {args.symbol}...")
    try:
        candles = fetch_gold_candles(symbol=args.symbol, period=args.period, interval=args.timeframe)
    except Exception as exc:
        logger.error(f"Failed to fetch market data: {exc}")
        sys.exit(1)

    if len(candles) < 100:
        logger.error(f"Insufficient candles fetched: {len(candles)}. Minimum 100 required.")
        sys.exit(1)

    logger.info(f"Fetched {len(candles)} candles. Date range: {candles[0]['date']} -> {candles[-1]['date']}")

    # 2. Extract features using saved scaler
    macro_data = analyze_macro_drivers()
    macro_score = float(macro_data.get("macro_gold_score", 0.0))

    processor = GoldRLDataProcessor()
    processor.scaler = predictor.processor.scaler

    X, prices, valid_candles = processor.build_features_from_candles(
        candles, macro_score=macro_score, fit_scaler=False
    )

    # 3. Setup environment
    env = GoldTradingEnv(
        features=X,
        prices=prices,
        initial_balance=args.initial_balance,
        risk_per_trade_pct=args.risk_pct,
        spread_pips=args.spread_pips,
    )

    # 4. Run Backtest
    logger.info("Executing evaluation loop...")
    obs, _ = env.reset()
    done = False
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    while not done:
        action, _ = predictor.model.predict(obs, deterministic=True)
        action_int = int(action)
        action_counts[action_int] = action_counts.get(action_int, 0) + 1
        obs, reward, terminated, truncated, _ = env.step(action_int)
        done = terminated or truncated

    summary = env._get_episode_metrics()

    # Calculate Benchmark Buy-and-Hold
    first_close = float(prices[0, 3])
    last_close = float(prices[-1, 3])
    bnh_return_pct = ((last_close - first_close) / first_close) * 100.0
    alpha_pct = summary["total_return_pct"] - bnh_return_pct

    logger.info("================================================================")
    logger.info("🏆 PERFORMANCE SUMMARY")
    logger.info("================================================================")
    logger.info(f"Initial Balance:      ${summary['initial_balance']:,.2f}")
    logger.info(f"Final Equity:         ${summary['final_equity']:,.2f}")
    logger.info(f"Total Return:         {summary['total_return_pct']:+.2f}%")
    logger.info(f"Benchmark Buy&Hold:   {bnh_return_pct:+.2f}%")
    logger.info(f"Alpha vs Benchmark:   {alpha_pct:+.2f}%")
    logger.info(f"Max Drawdown:         {summary['max_drawdown_pct']:.2f}%")
    logger.info(f"Sharpe Ratio:         {summary['sharpe_ratio']:.2f}")
    logger.info(f"Total Closed Trades:  {summary['total_trades']}")
    logger.info(f"Win Rate:             {summary['win_rate_pct']:.1f}%")
    logger.info(f"Profit Factor:        {summary['profit_factor']:.2f}")
    logger.info(
        f"Actions Taken:        HOLD: {action_counts[0]} | BUY: {action_counts[1]} | SELL: {action_counts[2]} | CLOSE: {action_counts[3]}"
    )
    logger.info("================================================================")

    if args.output_trades:
        output_p = Path(args.output_trades)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_p, "w") as f:
            json.dump({"summary": summary, "trades": env.trades_history}, f, indent=2)
        logger.info(f"Saved {len(env.trades_history)} trade records to: {output_p}")


if __name__ == "__main__":
    main()
