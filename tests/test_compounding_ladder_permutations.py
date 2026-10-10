"""
Unit tests for the Enhanced Compounding Ladder Manager with Permutations & Combinations.
Tests:
- Discrete 30-Level Mode (Backward compatibility)
- Tiered Phased Sprint Mode (Phased risk & Ratchet Floors)
- Dynamic Equity Percentage Mode
- Streak-Adaptive Mode (Anti-Martingale)
- Circuit Breaker De-Risking
- Ratchet Floor Enforcements
"""
import os
import tempfile
import unittest

from gold_trader.core.compounding_ladder import (
    CompoundingLadderManager,
    LADDER_MATRIX,
    find_closest_level_for_balance,
    get_ladder_level_info,
)


class TestCompoundingLadderPermutations(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            try:
                os.unlink(self.temp_file.name)
            except OSError:
                pass

    def test_default_backward_compatibility(self):
        """Ensure standard discrete 30-level behavior remains 100% backward compatible."""
        ladder = CompoundingLadderManager(state_file=self.temp_file.name)
        status = ladder.get_status()
        self.assertEqual(status["current_level"], 1)
        self.assertEqual(status["current_balance"], 20.0)
        self.assertEqual(ladder.ladder_mode, "DISCRETE_30_LEVEL")

        # Bracket check
        bracket = ladder.calculate_bracket_order(2700.0, "BUY", use_atr_stops=False)
        self.assertEqual(bracket["lots"], 0.03)
        self.assertEqual(bracket["take_profit"], 2702.0)
        self.assertEqual(bracket["stop_loss"], 2698.5)

    def test_tiered_phased_sprint_mode(self):
        """Test sprint phase (10%), expansion phase (8%), and preservation phase (5%)."""
        ladder = CompoundingLadderManager(
            state_file=self.temp_file.name,
            ladder_mode="TIERED_PHASED_SPRINT",
            starting_balance=100.0,
            loss_policy="PROGRESSIVE_RATCHET",
        )
        status = ladder.get_status()
        self.assertEqual(status["ladder_mode"], "TIERED_PHASED_SPRINT")
        self.assertEqual(status["current_balance"], 100.0)

        # In Phase 1 ($100 balance < $500), risk is 10% ($10 risk)
        # With SL = $10.00 (1000 points), lots = $10 / ($10 * 100) = 0.01
        bracket = ladder.calculate_bracket_order(
            current_price=2700.0,
            direction="BUY",
            atr=6.67,  # 1.5 * 6.67 = $10.00 SL
            min_sl_pts=10.0,
            rr_ratio=1.8,
        )
        self.assertEqual(bracket["sl_points"], 10.0)
        self.assertEqual(bracket["lots"], 0.01)

        # Simulate reaching Expansion Phase ($600 balance)
        ladder.state["current_balance"] = 600.0
        ladder.state["peak_balance"] = 600.0
        ladder.update_ratchet_floor()
        # Ratchet floor at $500+ is $350
        self.assertEqual(ladder.get_ratchet_floor(), 350.0)

        # Risk capital is balance - floor = $600 - $350 = $250.
        # In Phase 2 ($500-$2000), risk is 8% -> 8% of $250 = $20 risk.
        # Lots with $10 SL: $20 / ($10 * 100) = 0.02
        bracket2 = ladder.calculate_bracket_order(
            current_price=2700.0,
            direction="BUY",
            atr=6.67,
            min_sl_pts=10.0,
            rr_ratio=1.8,
        )
        self.assertEqual(bracket2["lots"], 0.02)

    def test_streak_adaptive_mode(self):
        """Test anti-martingale risk expansion on wins and de-risking on loss."""
        ladder = CompoundingLadderManager(
            state_file=self.temp_file.name,
            ladder_mode="STREAK_ADAPTIVE",
            starting_balance=1000.0,
        )

        # Base risk: 6.0%
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 6.0)

        # 1st win -> consecutive wins = 1 -> risk = 6.0 + 1.5 = 7.5%
        ladder.record_trade_result(win=True, realized_pnl=60.0, exit_price=2710.0)
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 7.5)

        # 2nd win -> consecutive wins = 2 -> risk = 6.0 + 3.0 = 9.0%
        ladder.record_trade_result(win=True, realized_pnl=75.0, exit_price=2715.0)
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 9.0)

        # Loss -> drops immediately to de-risked baseline (3.5%)
        ladder.record_trade_result(win=False, realized_pnl=-90.0, exit_price=2700.0)
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 3.5)

    def test_circuit_breaker_mechanism(self):
        """Test 50% de-risking after 2 consecutive losses."""
        ladder = CompoundingLadderManager(
            state_file=self.temp_file.name,
            ladder_mode="DYNAMIC_EQUITY_PERCENT",
            starting_balance=1000.0,
            base_risk_pct=8.0,
            enable_circuit_breaker=True,
        )
        self.assertFalse(ladder.is_circuit_breaker_active())
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 8.0)

        # 1st loss
        ladder.record_trade_result(win=False, realized_pnl=-80.0, exit_price=2690.0)
        self.assertFalse(ladder.is_circuit_breaker_active())
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 8.0)

        # 2nd loss in a row -> Circuit breaker activates, risk halved to 4.0%
        ladder.record_trade_result(win=False, realized_pnl=-73.6, exit_price=2680.0)
        self.assertTrue(ladder.is_circuit_breaker_active())
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 4.0)

        # Win resets circuit breaker
        ladder.record_trade_result(win=True, realized_pnl=60.0, exit_price=2695.0)
        self.assertFalse(ladder.is_circuit_breaker_active())
        self.assertAlmostEqual(ladder.get_active_risk_pct(), 8.0)

    def test_noise_adaptive_ladder_mode(self):
        """Test the redesigned noise-adaptive compounding ladder ($100 seed, $8+ stops, 1.5R target)."""
        ladder = CompoundingLadderManager(
            state_file=self.temp_file.name,
            ladder_mode="NOISE_ADAPTIVE_LADDER",
            starting_balance=100.0,
            loss_policy="PROGRESSIVE_RATCHET",
        )
        status = ladder.get_status()
        self.assertEqual(status["ladder_mode"], "NOISE_ADAPTIVE_LADDER")
        self.assertEqual(status["current_level"], 1)
        self.assertEqual(status["current_balance"], 100.0)
        self.assertEqual(status["next_trade_specs"]["phase"], "Sprint")
        self.assertEqual(status["next_trade_specs"]["expected_risk_usd"], 10.0)
        self.assertEqual(status["next_trade_specs"]["expected_gain_usd"], 15.0)

        # Test Bracket Calculation with real gold noise (1.5x ATR = $10.00 stop)
        bracket = ladder.calculate_bracket_order(
            current_price=2700.0,
            direction="BUY",
            atr=6.67,  # 1.5 * 6.67 = $10.00 SL
            min_sl_pts=8.0,
        )
        self.assertEqual(bracket["sl_points"], 10.0)
        self.assertEqual(bracket["tp_points"], 15.0)  # 1.5R target
        self.assertEqual(bracket["lots"], 0.01)       # $10 risk / ($10 * 100) = 0.01
        self.assertEqual(bracket["expected_loss_usd"], 10.0)
        self.assertEqual(bracket["expected_profit_usd"], 15.0)

        # Win trade -> advances to Level 2 ($115.00)
        res = ladder.record_trade_result(win=True, realized_pnl=15.0, exit_price=2715.0)
        self.assertEqual(res["new_level"], 2)
        self.assertEqual(res["new_balance"], 115.0)

        # Next trade specs for Level 2
        status_l2 = ladder.get_status()
        self.assertEqual(status_l2["current_level"], 2)
        self.assertEqual(status_l2["next_trade_specs"]["expected_risk_usd"], 11.50)
        self.assertEqual(status_l2["next_trade_specs"]["expected_gain_usd"], 17.25)


if __name__ == "__main__":
    unittest.main()

