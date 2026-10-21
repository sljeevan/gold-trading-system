"""
Unit tests for the Autonomous Gold Live Scanner and Trade Execution Engine.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gold_trader.brokers.base import BaseBroker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader


class DummyMockBroker(BaseBroker):
    def __init__(self):
        self.orders = []
        self.modified = []
        self.open_pos = []

    def connect(self) -> bool:
        return True

    def get_account_info(self):
        return {
            "broker": "MockBroker",
            "account_id": "MOCK123",
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0,
            "margin_level_pct": 0.0,
            "leverage": 100,
            "currency": "USD",
        }

    def get_open_positions(self, symbol=None):
        return list(self.open_pos)

    def get_symbol_info(self, symbol=None):
        return {"status": "SUCCESS", "bid": 4350.0, "ask": 4350.35}

    def place_order(self, symbol, side, lots, sl=None, tp=None, comment="", price=None, order_type="MARKET", **kwargs):
        res = {
            "status": "SUCCESS",
            "ticket": f"TICK-{len(self.orders)+1}",
            "symbol": symbol,
            "action": side,
            "lots": lots,
            "price": 4350.0,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "message": "Filled",
            "timestamp": "2026-09-02T12:00:00Z",
        }
        self.orders.append(res)
        return res

    def modify_position(self, ticket, sl=None, tp=None):
        self.modified.append({"ticket": ticket, "sl": sl, "tp": tp})
        return {"status": "SUCCESS", "ticket": ticket, "sl": sl, "tp": tp, "message": "Modified"}

    def close_position(self, ticket):
        return {"status": "SUCCESS", "ticket": ticket, "message": "Closed"}


class TestGoldAutoTrader(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.broker = DummyMockBroker()
        self.cfg = AutoTraderConfig(
            symbol="XAUUSD",
            mode="STANDARD_SWING",
            min_risk_reward=1.5,
            min_quality_score=80,
            max_open_positions=2,
            auto_execute=True,
            trail_to_breakeven_at_t1=True,
            session_filter=False,
            send_telegram_alerts=False,
        )
        self.mock_notifier = MagicMock()
        self.trader = GoldAutoTrader(
            broker=self.broker,
            config_override=self.cfg,
            notifier=self.mock_notifier,
            db_path=self.temp_db.name,
        )

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)

    @patch("gold_trader.core.auto_trader.generate_gold_trade_setup")
    def test_evaluate_and_trade_skip_on_hold(self, mock_setup):
        mock_setup.return_value = {
            "symbol": "XAUUSD",
            "action": "HOLD",
            "bias": "NEUTRAL",
            "setup_name": "None",
            "entry_price": 4350.0,
            "quality_score": 40,
            "risk_reward_t1": 1.0,
            "suggested_lot_size": 0.01,
            "stop_loss": 4330.0,
            "target_1": 4370.0,
            "target_2": 4400.0,
        }

        result = self.trader.evaluate_and_trade()
        self.assertEqual(result["action"], "HOLD")
        self.assertIn("SKIP", result["decision"])
        self.assertEqual(len(self.broker.orders), 0)

    @patch("gold_trader.core.auto_trader.generate_gold_trade_setup")
    def test_evaluate_and_trade_executes_on_valid_setup(self, mock_setup):
        mock_setup.return_value = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "bias": "BULLISH",
            "setup_name": "Golden Pocket Retracement",
            "entry_price": 4350.0,
            "quality_score": 90,
            "risk_reward_t1": 2.0,
            "suggested_lot_size": 0.05,
            "stop_loss": 4330.0,
            "target_1": 4390.0,
            "target_2": 4420.0,
        }

        result = self.trader.evaluate_and_trade()
        self.assertEqual(result["action"], "BUY")
        self.assertIn("EXECUTE", result["decision"])
        self.assertEqual(len(self.broker.orders), 1)
        self.assertEqual(self.broker.orders[0]["action"], "BUY")
        self.assertEqual(self.broker.orders[0]["lots"], 0.05)

    def test_trailing_stop_to_breakeven(self):
        # Open position with $10 profit
        self.broker.open_pos = [{
            "ticket": "POS-101",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.01,
            "open_price": 4340.0,
            "current_price": 4352.0,  # $12 in profit
            "sl": 4330.0,             # SL below entry
            "tp": 4380.0,
            "pnl": 12.0,
        }]

        actions = self.trader._manage_open_positions(self.broker.open_pos)
        self.assertEqual(len(actions), 1)
        self.assertIn("Trailed SL to Break-Even", actions[0])
        self.assertEqual(len(self.broker.modified), 1)
        self.assertEqual(self.broker.modified[0]["sl"], 4340.5)

    def test_trailing_stop_to_breakeven_deferred_when_under_1r(self):
        # Position with only $4 profit on a $10 risk distance trade: Should NOT trail to BE yet
        self.broker.open_pos = [{
            "ticket": "POS-102",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.01,
            "open_price": 4340.0,
            "current_price": 4344.0,  # Only $4 in profit (< $8 / 1.0R)
            "sl": 4330.0,             # $10 risk distance
            "tp": 4380.0,
            "pnl": 4.0,
        }]
        actions = self.trader._manage_open_positions(self.broker.open_pos)
        self.assertEqual(len(actions), 0)
        self.assertEqual(len(self.broker.modified), 0)

    @patch("gold_trader.core.auto_trader.generate_donchian_breakout_setup")
    def test_evaluate_and_trade_donchian_breakout_mode(self, mock_donchian):
        self.trader.config.mode = "DONCHIAN_BREAKOUT"
        mock_donchian.return_value = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "bias": "BULLISH",
            "setup_name": "Donchian 20-Bar Channel Breakout + SuperTrend",
            "entry_price": 4350.0,
            "quality_score": 90,
            "risk_reward_t1": 1.5,
            "suggested_lot_size": 0.02,
            "stop_loss": 4335.0,
            "target_1": 4372.5,
            "target_2": 4385.0,
        }
        result = self.trader.evaluate_and_trade()
        self.assertEqual(result["action"], "BUY")
        self.assertIn("EXECUTE", result["decision"])
        self.assertEqual(len(self.broker.orders), 1)
        self.assertEqual(self.broker.orders[0]["action"], "BUY")
        self.assertEqual(self.broker.orders[0]["lots"], 0.02)

    @patch("gold_trader.core.auto_trader.generate_donchian_momentum_hybrid_setup")
    def test_evaluate_and_trade_donchian_momentum_hybrid_mode(self, mock_hybrid):
        self.trader.config.mode = "DONCHIAN_MOMENTUM_HYBRID"
        mock_hybrid.return_value = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "bias": "BULLISH",
            "setup_name": "SuperTrend + MACD Momentum Continuation",
            "entry_price": 4350.0,
            "quality_score": 90,
            "risk_reward_t1": 1.8,
            "suggested_lot_size": 0.03,
            "stop_loss": 4335.0,
            "target_1": 4377.0,
            "target_2": 4387.5,
        }
        result = self.trader.evaluate_and_trade()
        self.assertEqual(result["action"], "BUY")
        self.assertIn("EXECUTE", result["decision"])
        self.assertEqual(len(self.broker.orders), 1)
        self.assertEqual(self.broker.orders[0]["action"], "BUY")
        self.assertEqual(self.broker.orders[0]["lots"], 0.03)

    @patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup")
    def test_evaluate_and_trade_compounding_hybrid_mode(self, mock_compounding):
        self.trader.config.mode = "COMPOUNDING_HYBRID"
        mock_compounding.return_value = {
            "status": "APPROVED_HIGH_CONVICTION",
            "action": "BUY",
            "symbol": "XAUUSD",
            "level": 10,
            "order_type": "MARKET",
            "market_price": 4350.0,
            "entry_price": 4350.0,
            "pullback_discount_pips": 0.0,
            "lots": 0.03,
            "take_profit": 4378.8,
            "stop_loss": 4334.0,
            "breakeven_trigger_price": 4366.0,
            "tp_pips": 288.0,
            "sl_pips": 160.0,
            "expected_profit_usd": 86.4,
            "expected_loss_usd": 48.0,
            "reward_risk_ratio": 1.8,
            "setup_name": "1H Donchian + Momentum Hybrid Compounding",
            "quality_score": 90,
            "candle_time": "2026-10-04T12:00:00Z",
        }
        result = self.trader.evaluate_and_trade()
        self.assertEqual(result["action"], "BUY")
        self.assertIn("EXECUTE", result["decision"])
        self.assertEqual(len(self.broker.orders), 1)
        self.assertEqual(self.broker.orders[0]["action"], "BUY")
        self.assertEqual(self.broker.orders[0]["lots"], 0.03)
        self.assertIn("AI-FlatRisk-XAUUSD", self.broker.orders[0]["comment"])

    def test_weekend_market_closed_blocks_execution(self):
        self.trader.config.session_filter = True
        # If currently Sunday before 22 UTC, evaluate_and_trade should block execution
        from datetime import datetime, timezone
        now_dt = datetime.now(timezone.utc)
        if now_dt.weekday() in (5, 6) and (now_dt.weekday() == 5 or now_dt.hour < 22):
            result = self.trader.evaluate_and_trade()
            self.assertIn("Market Closed", result["decision"])

    def test_history_logging(self):
        self.trader.log_scan(
            symbol="XAUUSD",
            price=4355.0,
            bias="BULLISH",
            setup_name="Test Setup",
            action="HOLD",
            quality_score=50,
            rr_ratio=1.0,
            decision="SKIP: Test",
            details={"test": True},
        )
        history = self.trader.get_recent_history(limit=5)
        self.assertEqual(len(history["scans"]), 1)
        self.assertEqual(history["scans"][0]["setup_name"], "Test Setup")

    def test_get_matching_open_positions_and_has_open_position(self):
        # When empty
        self.broker.open_pos = []
        self.assertFalse(self.trader.has_open_position())
        self.assertEqual(self.trader.get_matching_open_positions(), [])

        # When non-matching symbol (BTCUSD while trader is XAUUSD)
        self.broker.open_pos = [{"ticket": "1001", "symbol": "BTCUSD", "lots": 0.1, "open_price": 82000.0, "current_price": 82500.0, "side": "BUY"}]
        self.assertFalse(self.trader.has_open_position())
        self.assertEqual(self.trader.get_matching_open_positions(), [])

        # When matching symbol
        self.broker.open_pos = [{"ticket": "1002", "symbol": "XAUUSD", "lots": 0.03, "open_price": 4150.0, "current_price": 4160.0, "side": "BUY"}]
        self.assertTrue(self.trader.has_open_position())
        matching = self.trader.get_matching_open_positions()
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["ticket"], "1002")

    def test_manage_active_positions_fast_cycle(self):
        self.broker.open_pos = [{
            "ticket": "2001",
            "symbol": "XAUUSD",
            "lots": 0.02,
            "open_price": 4150.0,
            "current_price": 4155.0,
            "side": "BUY",
            "sl": 4140.0,
            "tp": 4180.0,
        }]
        res = self.trader.manage_active_positions()
        self.assertEqual(res["status"], "FAST_MANAGE")
        self.assertEqual(res["open_positions_count"], 1)
        self.assertIn("decision", res)


if __name__ == "__main__":
    unittest.main()
