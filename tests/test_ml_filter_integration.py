"""
Unit & Integration Tests for Machine Learning Meta-Labeling Engine.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from gold_trader.brokers.base import BaseBroker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.core.ml_filter import GoldMLPredictor, extract_candle_features, label_triple_barrier


class TestMLFilterIntegration(unittest.TestCase):
    def setUp(self):
        # Create 250 synthetic candles
        self.mock_candles = []
        base = 2650.0
        for i in range(250):
            o = base + np.sin(i / 10.0) * 5.0
            c = o + np.cos(i / 10.0) * 3.0
            h = max(o, c) + 2.0
            l = min(o, c) - 2.0
            self.mock_candles.append({
                "date": f"2026-10-01 {i%24:02d}:00",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": 1000.0,
            })

    def test_feature_extraction(self):
        X, names = extract_candle_features(self.mock_candles, macro_score=0.5)
        self.assertEqual(len(X), 250)
        self.assertIn("dist_ema50_atr", names)
        self.assertIn("rsi14", names)
        self.assertIn("hour_sin", names)
        self.assertIn("supertrend_dir", names)
        # Non-warmup rows should have valid finite features
        valid_row = X[-1]
        self.assertFalse(np.isnan(valid_row).any())

    def test_triple_barrier_labeling(self):
        labels_buy = label_triple_barrier(self.mock_candles, direction="BUY", atr_multiplier_sl=1.5, rr_ratio=1.25, min_sl_pts=5.0)
        labels_sell = label_triple_barrier(self.mock_candles, direction="SELL", atr_multiplier_sl=1.5, rr_ratio=1.25, min_sl_pts=5.0)
        self.assertEqual(len(labels_buy), 250)
        self.assertEqual(len(labels_sell), 250)
        # Valid labels should be either 0.0 or 1.0
        valid_vals = set(labels_buy[~np.isnan(labels_buy)])
        self.assertTrue(valid_vals.issubset({0.0, 1.0}))

    @patch("gold_trader.core.ml_filter.fetch_gold_candles")
    def test_predictor_train_and_predict(self, mock_fetch):
        import tempfile
        import os
        mock_fetch.return_value = self.mock_candles * 2  # 500 candles
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
            temp_model_path = tf.name
        try:
            pred = GoldMLPredictor(model_path=temp_model_path, symbol="XAUUSD")
            train_res = pred.train_model(period="3mo", interval="1h", min_sl_pts=4.0)
            self.assertEqual(train_res["status"], "SUCCESS")

            eval_res = pred.predict_trade_probability(self.mock_candles, direction="BUY", min_probability_threshold=0.52)
            self.assertIn("probability_win", eval_res)
            self.assertIn("is_high_conviction", eval_res)
            self.assertIn("recommended_lot_multiplier", eval_res)
            self.assertTrue(0.0 <= eval_res["probability_win"] <= 1.0)
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    @patch("gold_trader.core.auto_trader.fetch_gold_candles")
    @patch("gold_trader.core.ml_filter.GoldMLPredictor.predict_trade_probability")
    @patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup")
    def test_auto_trader_ml_filter_approval_and_rejection(self, mock_setup, mock_ml, mock_candles):
        mock_candles.return_value = self.mock_candles
        mock_setup.return_value = {
            "status": "APPROVED",
            "action": "BUY",
            "symbol": "BTCUSD",
            "level": 1,
            "order_type": "MARKET",
            "market_price": 65000.0,
            "entry_price": 65000.0,
            "lots": 0.02,
            "take_profit": 66000.0,
            "stop_loss": 64000.0,
            "expected_profit_usd": 30.0,
            "expected_loss_usd": 24.0,
            "reward_risk_ratio": 1.25,
            "setup_name": "Donchian 10 Touch",
            "quality_score": 92,
            "candle_time": "2026-10-18T10:00:00Z",
        }

        mock_broker = MagicMock(spec=BaseBroker)
        mock_broker.get_account_info.return_value = {"balance": 1000.0, "equity": 1000.0, "free_margin": 1000.0}
        mock_broker.get_open_positions.return_value = []
        mock_broker.get_pending_orders.return_value = []
        mock_broker.get_symbol_info.return_value = {"status": "SUCCESS", "bid": 65000.0, "ask": 65005.0}

        # Case 1: Low probability trade (< threshold) -> Should be rejected by ML Gate
        mock_ml.return_value = {
            "direction": "BUY",
            "probability_win": 0.44,
            "is_high_conviction": False,
            "recommended_lot_multiplier": 0.0,
            "recommendation": "PASS_LOW_PROBABILITY",
        }

        cfg_ml = AutoTraderConfig(
            symbol="BTCUSD",
            mode="COMPOUNDING_TOUCH",
            enable_ml_filter=True,
            ml_min_probability=0.52,
            session_filter=False,
            auto_execute=False,
            db_path=":memory:",
            enable_news_catalyst_guard=False,
        )
        trader = GoldAutoTrader(broker=mock_broker, config_override=cfg_ml)
        trader.is_market_open = MagicMock(return_value=True)
        res_reject = trader.evaluate_and_trade()
        self.assertIn("ML Filter Gate", res_reject["decision"])

        # Case 2: High probability trade (>= threshold) -> Should be approved
        mock_ml.return_value = {
            "direction": "BUY",
            "probability_win": 0.62,
            "is_high_conviction": True,
            "recommended_lot_multiplier": 1.25,
            "recommendation": "EXECUTE_HIGH_CONVICTION",
        }
        res_approve = trader.evaluate_and_trade()
        self.assertIn("EXECUTE", res_approve["decision"])


if __name__ == "__main__":
    unittest.main()
