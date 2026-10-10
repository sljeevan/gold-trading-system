"""
Inference & Prediction Engine for Gold PPO Reinforcement Learning Model.
Loads trained policy checkpoints, computes live stationarized observations,
and outputs actionable trade signals with probability distributions.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.macro import analyze_macro_drivers
from gold_trader.rl.data_processor import GoldRLDataProcessor
from gold_trader.rl.trainer import DEFAULT_MODEL_DIR

logger = logging.getLogger("GoldRLPredictor")


class GoldRLPredictor:
    """
    Production-ready inference predictor for the Gold RL Agent.
    """

    ACTION_MAP = {
        0: "HOLD",
        1: "BUY",
        2: "SELL",
        3: "CLOSE",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        self.model_path = os.path.expanduser(
            model_path or os.path.join(DEFAULT_MODEL_DIR, "latest_gold_ppo.zip")
        )
        self.scaler_path = os.path.expanduser(
            scaler_path or os.path.join(DEFAULT_MODEL_DIR, "scaler.pkl")
        )
        self.model: Optional[PPO] = None
        self.processor = GoldRLDataProcessor()
        self._load_model()

    def _load_model(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = PPO.load(self.model_path)
                self.processor.load_scaler(self.scaler_path)
                logger.info(f"Loaded Gold PPO model from {self.model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load Gold PPO model: {e}")
        return False

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.processor.scaler is not None

    def predict_action(
        self,
        candles: List[Dict[str, Any]],
        macro_score: float = 0.0,
        current_position: int = 0,  # +1 Long, -1 Short, 0 Flat
        unrealized_pnl: float = 0.0,
        holding_bars: int = 0,
        current_drawdown: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluates current market state and returns RL policy decision with probability distribution.
        """
        if not self.is_ready:
            return {
                "error": "Model or scaler not loaded. Train a model first via GoldRLTrainer.",
                "action": "HOLD",
                "confidence": 0.0,
            }

        X, prices, valid_candles = self.processor.build_features_from_candles(
            candles, macro_score=macro_score, fit_scaler=False
        )

        latest_market_features = X[-1]
        close_price = float(prices[-1, 3])
        atr_val = max(0.10, float(prices[-1, 4]))

        # Construct portfolio state vector (6 elements)
        portfolio_vec = np.array(
            [
                float(current_position),
                float(np.clip(unrealized_pnl / 1000.0, -2.0, 2.0)),
                float(np.clip(holding_bars / 48.0, 0.0, 1.0)),
                float(np.clip(current_drawdown, 0.0, 1.0)),
                1.0,  # Balance ratio baseline
                0.0,  # Distance to SL baseline
            ],
            dtype=np.float32,
        )

        full_obs = np.concatenate([latest_market_features, portfolio_vec]).astype(np.float32)

        # 1. Deterministic prediction
        action_idx, _states = self.model.predict(full_obs, deterministic=True)
        action_int = int(action_idx)
        action_name = self.ACTION_MAP.get(action_int, "HOLD")

        # 2. Extract action distribution probabilities via policy network
        obs_tensor = torch.as_tensor(full_obs).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]

        prob_dict = {
            self.ACTION_MAP[i]: round(float(probs[i]), 4) for i in range(len(self.ACTION_MAP))
        }

        confidence = round(float(probs[action_int]), 4)

        return {
            "action": action_name,
            "action_int": action_int,
            "confidence": confidence,
            "action_probabilities": prob_dict,
            "current_price": round(close_price, 2),
            "atr": round(atr_val, 2),
            "suggested_sl_distance": round(1.5 * atr_val, 2),
            "suggested_tp_distance": round(3.0 * atr_val, 2),
        }
