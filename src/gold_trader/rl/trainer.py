"""
PPO Training Pipeline for Gold Trading Agent.
Coordinates data ingestion, environment generation, walk-forward train/eval splitting,
PPO agent training via Stable-Baselines3, and artifact persistence.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.macro import analyze_macro_drivers
from gold_trader.rl.data_processor import GoldRLDataProcessor
from gold_trader.rl.gold_env import GoldTradingEnv

logger = logging.getLogger("GoldRLTrainer")

DEFAULT_MODEL_DIR = os.path.expanduser("~/.gold_trading_data/models/gold_ppo")


class GoldRLTrainer:
    """
    Automated PPO Trainer for the Gold Trading Reinforcement Learning System.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = os.path.expanduser(model_dir or DEFAULT_MODEL_DIR)
        os.makedirs(self.model_dir, exist_ok=True)
        self.processor = GoldRLDataProcessor()

    def run_training_pipeline(
        self,
        symbol: str = "XAUUSD",
        period: str = "2y",
        interval: str = "1h",
        total_timesteps: int = 50000,
        train_split_ratio: float = 0.80,
        initial_balance: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
        spread_pips: float = 2.5,
        learning_rate: float = 0.0003,
        batch_size: int = 64,
        n_steps: int = 1024,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end PPO training for the specified Gold symbol.
        """
        logger.info(f"Starting Gold PPO Training for {symbol} ({period} / {interval})...")

        # 1. Fetch candles & Macro score
        candles = fetch_gold_candles(symbol, period=period, interval=interval)
        if len(candles) < 300:
            return {"error": f"Insufficient candles ({len(candles)}) for training. Minimum 300 required."}

        macro_data = analyze_macro_drivers()
        macro_score = float(macro_data.get("macro_gold_score", 0.0))

        # 2. Extract stationary features and fit scaler
        X, prices, valid_candles = self.processor.build_features_from_candles(
            candles, macro_score=macro_score, fit_scaler=True
        )

        n_samples = len(X)
        split_idx = int(n_samples * train_split_ratio)

        X_train, prices_train = X[:split_idx], prices[:split_idx]
        X_test, prices_test = X[split_idx:], prices[split_idx:]

        logger.info(
            f"Dataset prepared: {n_samples} total bars | Train: {len(X_train)} bars | Test: {len(X_test)} bars"
        )

        # 3. Create Gym Environments
        train_env = Monitor(
            GoldTradingEnv(
                features=X_train,
                prices=prices_train,
                initial_balance=initial_balance,
                risk_per_trade_pct=risk_per_trade_pct,
                spread_pips=spread_pips,
            )
        )

        eval_env = Monitor(
            GoldTradingEnv(
                features=X_test,
                prices=prices_test,
                initial_balance=initial_balance,
                risk_per_trade_pct=risk_per_trade_pct,
                spread_pips=spread_pips,
            )
        )

        # 4. Initialize PPO Agent
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

        # 5. Train Model
        logger.info(f"Training PPO policy for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)

        # 6. Evaluate on Out-Of-Sample Test Environment
        logger.info("Evaluating trained agent on out-of-sample test set...")
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        eval_metrics = eval_env.unwrapped._get_episode_metrics()
        logger.info(f"Test Set Evaluation Results: {eval_metrics}")

        # 7. Persist Checkpoints and Artifacts
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_filename = f"gold_ppo_{symbol}_{interval}_{timestamp}.zip"
        model_path = os.path.join(self.model_dir, model_filename)
        latest_model_path = os.path.join(self.model_dir, "latest_gold_ppo.zip")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        metadata_path = os.path.join(self.model_dir, "latest_metadata.json")

        model.save(model_path)
        model.save(latest_model_path)
        self.processor.save_scaler(scaler_path)

        metadata = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "trained_at": timestamp,
            "total_timesteps": total_timesteps,
            "train_bars": len(X_train),
            "test_bars": len(X_test),
            "test_metrics": eval_metrics,
            "model_path": model_path,
            "scaler_path": scaler_path,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
