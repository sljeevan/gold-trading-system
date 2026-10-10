"""
Unit Tests for Custom Gold RL Environment, Data Processor, and PPO Pipeline.
"""
import unittest
import numpy as np
import os
import shutil
import tempfile

from gold_trader.rl.data_processor import GoldRLDataProcessor
from gold_trader.rl.gold_env import GoldTradingEnv
from gold_trader.rl.trainer import GoldRLTrainer
from gold_trader.rl.predictor import GoldRLPredictor


class TestGoldRLComponents(unittest.TestCase):

    def setUp(self):
        # Generate synthetic Gold OHLCV candles
        np.random.seed(42)
        n = 300
        base_price = 2650.0
        returns = np.random.normal(0.0001, 0.002, n)
        closes = base_price * np.exp(np.cumsum(returns))

        self.mock_candles = []
        for i in range(n):
            c = closes[i]
            o = closes[i - 1] if i > 0 else c
            h = max(o, c) + abs(np.random.normal(1.0, 0.5))
            l = min(o, c) - abs(np.random.normal(1.0, 0.5))
            self.mock_candles.append({
                "date": f"2026-09-01T{i % 24:02d}:00:00Z",
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1500 + i * 10,
            })

    def test_data_processor(self):
        processor = GoldRLDataProcessor()
        X, prices, valid_candles = processor.build_features_from_candles(
            self.mock_candles, macro_score=25.0, fit_scaler=True
        )

        self.assertEqual(len(X), len(prices))
        self.assertEqual(len(X), len(valid_candles))
        self.assertEqual(X.shape[1], len(GoldRLDataProcessor.FEATURE_NAMES))
        self.assertEqual(prices.shape[1], 5)  # O, H, L, C, ATR
        self.assertFalse(np.isnan(X).any(), "Feature matrix contains NaNs")
        self.assertFalse(np.isinf(X).any(), "Feature matrix contains Infs")

    def test_gold_trading_env(self):
        processor = GoldRLDataProcessor()
        X, prices, _ = processor.build_features_from_candles(
            self.mock_candles, fit_scaler=True
        )

        env = GoldTradingEnv(
            features=X,
            prices=prices,
            initial_balance=10000.0,
            risk_per_trade_pct=1.0,
        )

        obs, info = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(env.balance, 10000.0)

        # Step with BUY
        obs, reward, done, truncated, info = env.step(env.ACTION_BUY)
        self.assertEqual(env.position, 1)  # Long position opened
        self.assertGreater(env.lot_size, 0.0)
        self.assertGreater(env.take_profit, env.entry_price)
        self.assertLess(env.stop_loss, env.entry_price)

        # Step with HOLD
        obs, reward, done, truncated, info = env.step(env.ACTION_HOLD)
        self.assertFalse(done)

        # Step with CLOSE
        obs, reward, done, truncated, info = env.step(env.ACTION_CLOSE)
        self.assertEqual(env.position, 0)
        self.assertEqual(len(env.trades_history), 1)

    def test_trainer_and_predictor_fast_pipeline(self):
        temp_dir = tempfile.mkdtemp()
        try:
            trainer = GoldRLTrainer(model_dir=temp_dir)
            
            processor = GoldRLDataProcessor()
            X, prices, _ = processor.build_features_from_candles(
                self.mock_candles, fit_scaler=True
            )

            # Fast training run with 500 steps
            split = 220
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            
            env = Monitor(GoldTradingEnv(X[:split], prices[:split]))
            model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
            model.learn(total_timesteps=128)

            model_path = os.path.join(temp_dir, "latest_gold_ppo.zip")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")
            model.save(model_path)
            processor.save_scaler(scaler_path)

            predictor = GoldRLPredictor(model_path=model_path, scaler_path=scaler_path)
            self.assertTrue(predictor.is_ready)

            pred = predictor.predict_action(self.mock_candles)
            self.assertIn(pred["action"], ["HOLD", "BUY", "SELL", "CLOSE"])
            self.assertIn("action_probabilities", pred)
            prob_sum = sum(pred["action_probabilities"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=2)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
