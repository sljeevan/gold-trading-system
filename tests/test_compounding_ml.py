"""
Unit tests for the 30-Level Compounding Ladder & Machine Learning Meta-Filter.
"""
import os
import tempfile
import unittest
import numpy as np

from gold_trader.core.compounding_ladder import (
    CompoundingLadderManager,
    LADDER_MATRIX,
    get_ladder_level_info,
    find_closest_level_for_balance,
)
from gold_trader.core.ml_filter import (
    extract_candle_features,
    label_triple_barrier,
    GoldMLPredictor,
)
from gold_trader.core.strategy_engine import generate_compounding_ladder_setup


class TestCompoundingLadderAndML(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_file.close()

        # Build 250 mock candles for testing ML feature extraction
        self.mock_candles = []
        base = 2700.0
        for i in range(250):
            o = base + i * 0.20
            h = o + 2.50
            l = o - 2.00
            c = o + 0.80
            self.mock_candles.append({
                "date": f"2026-03-01T{i % 24:02d}:00:00Z",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": 500.0,
            })

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_ladder_matrix_integrity(self):
        """Ensure all 30 levels from the PDF match exact specifications."""
        self.assertEqual(len(LADDER_MATRIX), 30)

        # Level 1 check
        l1 = LADDER_MATRIX[0]
        self.assertEqual(l1["level"], 1)
        self.assertEqual(l1["starting_balance"], 20.0)
        self.assertEqual(l1["lot_size"], 0.03)
        self.assertEqual(l1["tp_pips"], 20.0)
        self.assertEqual(l1["ending_balance"], 26.0)

        # Level 10 check
        l10 = LADDER_MATRIX[9]
        self.assertEqual(l10["level"], 10)
        self.assertEqual(l10["starting_balance"], 212.0)
        self.assertEqual(l10["lot_size"], 0.32)
        self.assertEqual(l10["ending_balance"], 276.0)

        # Level 30 check
        l30 = LADDER_MATRIX[29]
        self.assertEqual(l30["level"], 30)
        self.assertEqual(l30["starting_balance"], 40296.0)
        self.assertEqual(l30["lot_size"], 60.46)
        self.assertEqual(l30["ending_balance"], 52384.0)

    def test_bracket_order_calculation(self):
        ladder = CompoundingLadderManager(state_file=self.temp_file.name)
        ladder.reset_ladder(20.0)

        buy_order = ladder.calculate_bracket_order(2700.0, "BUY")
        self.assertEqual(buy_order["lots"], 0.03)
        self.assertEqual(buy_order["take_profit"], 2702.0)  # +$2.00 (+20 pips)
        self.assertEqual(buy_order["stop_loss"], 2698.5)   # -$1.50 (-15 pips)
        self.assertEqual(buy_order["expected_profit_usd"], 6.0)
        self.assertEqual(buy_order["expected_loss_usd"], 4.5)

        sell_order = ladder.calculate_bracket_order(2700.0, "SELL")
        self.assertEqual(sell_order["take_profit"], 2698.0)
        self.assertEqual(sell_order["stop_loss"], 2701.5)

    def test_ladder_progression_and_loss(self):
        ladder = CompoundingLadderManager(state_file=self.temp_file.name, loss_policy="RESET_TO_LEVEL_1")
        ladder.reset_ladder(20.0)

        # Win Level 1
        res1 = ladder.record_trade_result(win=True, realized_pnl=6.0, exit_price=2702.0)
        self.assertEqual(res1["new_level"], 2)
        self.assertEqual(res1["new_balance"], 26.0)

        # Win Level 2
        res2 = ladder.record_trade_result(win=True, realized_pnl=8.0, exit_price=2702.0)
        self.assertEqual(res2["new_level"], 3)
        self.assertEqual(res2["new_balance"], 34.0)

        # Loss at Level 3
        res3 = ladder.record_trade_result(win=False, realized_pnl=-8.0, exit_price=2698.5)
        self.assertEqual(res3["new_level"], 1)  # Reset policy
        self.assertEqual(res3["new_balance"], 26.0)

    def test_ladder_step_down_policy(self):
        ladder = CompoundingLadderManager(state_file=self.temp_file.name, loss_policy="STEP_DOWN_BY_BALANCE")
        ladder.set_level(5)
        ladder.state["current_balance"] = 76.0  # Level 6 starting balance

        # If balance falls to $60, closest level with starting_balance <= 60 is Level 5 ($58)
        self.assertEqual(find_closest_level_for_balance(60.0), 5)
        self.assertEqual(find_closest_level_for_balance(15.0), 1)
        self.assertEqual(find_closest_level_for_balance(1000.0), 15)

    def test_ml_feature_extraction(self):
        features, names = extract_candle_features(self.mock_candles, macro_score=1.5)
        self.assertEqual(len(features), len(self.mock_candles))
        self.assertTrue(len(names) >= 14)
        # Latest row should not have NaN
        self.assertFalse(np.isnan(features[-1]).any())

    def test_triple_barrier_labeling(self):
        labels = label_triple_barrier(
            self.mock_candles,
            direction="BUY",
            tp_points=2.0,
            sl_points=1.5,
            max_holding_bars=10,
        )
        self.assertEqual(len(labels), len(self.mock_candles))
        valid_labels = labels[~np.isnan(labels)]
        self.assertTrue(len(valid_labels) > 0)
        # All valid labels must be 0 or 1
        for val in valid_labels:
            self.assertIn(val, (0.0, 1.0))

    def test_ml_predictor_inference(self):
        predictor = GoldMLPredictor()
        # Test inference with mock candles
        pred = predictor.predict_trade_probability(
            candles=self.mock_candles,
            direction="BUY",
            macro_score=1.0,
            min_probability_threshold=0.70,
        )
        self.assertIn("probability_win", pred)
        self.assertTrue(0.0 <= pred["probability_win"] <= 1.0)
        self.assertIn("is_high_conviction", pred)
        self.assertIn("recommendation", pred)


if __name__ == "__main__":
    unittest.main()
