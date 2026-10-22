"""
Unit tests for Production 0.5R Trailing Stop-Loss Engine and Dynamic Buffer Execution.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader


class MockBroker:
    def __init__(self):
        self.orders = []
        self.modified = []
        self.closed = []
        self.open_pos = []

    def connect(self) -> bool:
        return True

    def get_account_info(self):
        return {"balance": 3000.0, "equity": 3000.0, "margin": 0.0, "free_margin": 3000.0}

    def get_open_positions(self, symbol=None):
        return list(self.open_pos)

    def modify_position(self, ticket, sl=None, tp=None):
        self.modified.append({"ticket": ticket, "sl": sl, "tp": tp})
        return {"status": "SUCCESS", "ticket": ticket, "sl": sl, "tp": tp}

    def close_position(self, ticket, lots=None, **kwargs):
        self.closed.append({"ticket": ticket, "lots": lots})
        return {"status": "SUCCESS", "ticket": ticket, "lots": lots}

    def place_order(self, symbol, side, lots, sl=None, tp=None, comment="", price=None, order_type="MARKET", **kwargs):
        res = {
            "status": "SUCCESS",
            "ticket": f"TICK-{len(self.orders)+1}",
            "symbol": symbol,
            "action": side,
            "lots": lots,
            "price": price or 2700.0,
            "sl": sl,
            "tp": tp,
            "comment": comment,
        }
        self.orders.append(res)
        return res


class Test05RTrailingProduction(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.broker = MockBroker()
        self.cfg = AutoTraderConfig(
            symbol="XAUUSD",
            mode="FLAT_RISK_TOUCH",
            max_open_positions=0,  # Uncapped
            runner_trail_buffer_r=0.50,  # 0.5R Dynamic Trail
            enable_runner_trailing=True,
            send_telegram_alerts=False,
            session_filter=False,
            enable_news_catalyst_guard=False,
            enable_ml_filter=False,
            one_trade_per_bar=False,
        )
        self.trader = GoldAutoTrader(broker=self.broker, config_override=self.cfg, db_path=self.temp_db.name)

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)

    def test_dynamic_05r_trail_buffer_calculation_gold(self):
        # Entry at $2700.00, initial SL at $2690.00 (risk = $10.00 = 1.0R)
        # 0.5R trail buffer should be exactly $5.00
        ticket = "GOLD-101"
        self.trader.tracked_positions[ticket] = {
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "BUY",
            "lots": 0.08,
            "entry_price": 2700.0,
            "stop_loss": 2690.0,  # $10 risk
            "target_1": 2710.0,   # +1.0R
            "target_2": 2730.0,   # +3.0R
            "tp1_taken": 1,       # Target 1 already banked, now trailing
            "peak_price": 2710.0,
        }

        # Price advances to $2720.00 (+2.0R)
        self.broker.open_pos = [{
            "ticket": ticket,
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.04,
            "open_price": 2700.0,
            "current_price": 2720.0,
            "sl": 2700.50,  # Currently at BE + $0.50
            "tp": 2730.0,
        }]

        actions = self.trader._manage_open_positions(self.broker.open_pos)
        self.assertTrue(len(actions) >= 1)
        # Expected trailed SL = peak (2720) - 0.5R ($5.00) = $2715.00
        self.assertEqual(len(self.broker.modified), 1)
        self.assertEqual(self.broker.modified[0]["sl"], 2715.0)

    def test_dynamic_05r_trail_buffer_calculation_bitcoin(self):
        # Bitcoin: Entry $80,000, SL $79,200 (risk = $800 = 1.0R)
        # 0.5R trail buffer should be exactly $400.00
        self.trader.config.symbol = "BTCUSD"
        ticket = "BTC-101"
        self.trader.tracked_positions[ticket] = {
            "ticket": ticket,
            "symbol": "BTCUSD",
            "action": "BUY",
            "lots": 0.10,
            "entry_price": 80000.0,
            "stop_loss": 79200.0,  # $800 risk
            "target_1": 80800.0,
            "target_2": 82400.0,
            "tp1_taken": 1,
            "peak_price": 80800.0,
        }

        # Price advances to $82,000 (+2.5R)
        self.broker.open_pos = [{
            "ticket": ticket,
            "symbol": "BTCUSD",
            "side": "BUY",
            "lots": 0.05,
            "open_price": 80000.0,
            "current_price": 82000.0,
            "sl": 80030.0,  # BE + $30
            "tp": 82400.0,
        }]

        actions = self.trader._manage_open_positions(self.broker.open_pos)
        self.assertTrue(len(actions) >= 1)
        # Expected trailed SL = peak (82000) - 0.5R ($400) = $81,600.00
        self.assertEqual(len(self.broker.modified), 1)
        self.assertEqual(self.broker.modified[0]["sl"], 81600.0)

    def test_dynamic_05r_trail_buffer_calculation_ethereum(self):
        # Ethereum: Entry $2500, SL $2530 (risk = $30 = 1.0R)
        # 0.5R trail buffer should be exactly $15.00
        self.trader.config.symbol = "ETHUSD"
        ticket = "ETH-101"
        self.trader.tracked_positions[ticket] = {
            "ticket": ticket,
            "symbol": "ETHUSD",
            "action": "SELL",
            "lots": 2.50,
            "entry_price": 2500.0,
            "stop_loss": 2530.0,  # $30 risk
            "target_1": 2470.0,
            "target_2": 2410.0,
            "tp1_taken": 1,
            "peak_price": 2470.0,
        }

        # Price drops to $2440 (-2.0R)
        self.broker.open_pos = [{
            "ticket": ticket,
            "symbol": "ETHUSD",
            "side": "SELL",
            "lots": 1.25,
            "open_price": 2500.0,
            "current_price": 2440.0,
            "sl": 2498.0,  # BE - $2.00
            "tp": 2410.0,
        }]

        actions = self.trader._manage_open_positions(self.broker.open_pos)
        self.assertTrue(len(actions) >= 1)
        # Expected trailed SL for SELL = trough (2440) + 0.5R ($15) = $2455.00
        self.assertEqual(len(self.broker.modified), 1)
        self.assertEqual(self.broker.modified[0]["sl"], 2455.0)

    @patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup")
    def test_uncapped_exposure_allows_concurrent_positions(self, mock_setup):
        mock_setup.return_value = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "bias": "BULLISH",
            "setup_name": "1H Donchian 10 Touch",
            "entry_price": 2700.0,
            "quality_score": 90,
            "risk_reward_t1": 1.5,
            "suggested_lot_size": 0.08,
            "stop_loss": 2690.0,
            "target_1": 2715.0,
            "target_2": 2730.0,
            "candle_time": "2026-10-22T13:00:00Z",
        }

        # Simulate 3 already open positions on Gold
        self.broker.open_pos = [
            {"ticket": "1", "symbol": "XAUUSD", "side": "BUY", "lots": 0.04, "open_price": 2680.0, "current_price": 2700.0, "sl": 2680.5, "tp": 2730.0},
            {"ticket": "2", "symbol": "XAUUSD", "side": "BUY", "lots": 0.04, "open_price": 2685.0, "current_price": 2700.0, "sl": 2685.5, "tp": 2730.0},
            {"ticket": "3", "symbol": "XAUUSD", "side": "BUY", "lots": 0.04, "open_price": 2690.0, "current_price": 2700.0, "sl": 2690.5, "tp": 2730.0},
        ]

        # Execution should succeed and place a 4th order because max_open_positions is 0 (uncapped)
        res = self.trader.evaluate_and_trade()
        self.assertEqual(res["action"], "BUY")
        self.assertIn("EXECUTE", res["decision"])
        self.assertEqual(len(self.broker.orders), 1)


if __name__ == "__main__":
    unittest.main()
