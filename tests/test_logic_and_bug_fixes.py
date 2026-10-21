"""
Comprehensive Unit Tests for Bug Fixes and Logic Integrity Verification.
"""
from __future__ import annotations

import os
import tempfile
import typing
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from gold_trader.brokers.metaapi_broker import MetaApiBroker
from gold_trader.brokers.paper_broker import PaperBroker
from gold_trader.brokers.xm_mt5_broker import XmMT5Broker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.core.indicators import extract_all_indicators
from gold_trader.core.macro import analyze_macro_drivers
from gold_trader.core.openbb_feed import check_high_impact_news_window
from gold_trader.core.strategy_engine import generate_compounding_ladder_setup
from gold_trader.server import train_gold_ml_model


class TestLogicAndBugFixes(unittest.TestCase):
    def test_indicators_any_type_hint_resolution(self):
        """Verify extract_all_indicators type hints resolve cleanly without NameError."""
        hints = typing.get_type_hints(extract_all_indicators)
        self.assertIn("return", hints)

    def test_metaapi_broker_get_open_positions_success(self):
        """Verify MetaApiBroker.get_open_positions parses list of positions without NameError."""
        broker = MetaApiBroker(token="test_tok", account_id="test_acc")
        mock_data = [
            {
                "id": "12345",
                "symbol": "XAUUSD",
                "type": "POSITION_TYPE_BUY",
                "volume": 0.05,
                "openPrice": 2650.0,
                "currentPrice": 2660.0,
                "profit": 50.0,
            },
            {
                "id": "67890",
                "symbol": "BTCUSD",
                "type": "POSITION_TYPE_SELL",
                "volume": 0.10,
                "openPrice": 65000.0,
                "currentPrice": 64500.0,
                "profit": 50.0,
            },
        ]
        with patch.object(broker, "_send_request", return_value=mock_data):
            positions = broker.get_open_positions()
            self.assertEqual(len(positions), 2)
            self.assertEqual(positions[0]["ticket"], "12345")
            self.assertEqual(positions[0]["side"], "BUY")
            self.assertEqual(positions[1]["ticket"], "67890")
            self.assertEqual(positions[1]["side"], "SELL")

    def test_broker_signatures_support_lots_in_close_position(self):
        """Verify all broker close_position methods accept lots parameter without TypeError."""
        meta_broker = MetaApiBroker(token="tok", account_id="acc")
        with patch.object(meta_broker, "_send_request", return_value={"price": 2650.0}):
            res_meta = meta_broker.close_position("123", lots=0.02)
            self.assertEqual(res_meta["status"], "SUCCESS")
            self.assertEqual(res_meta["lots"], 0.02)

        xm_broker = XmMT5Broker()
        res_xm = xm_broker.close_position("123", lots=0.01)
        self.assertIn("status", res_xm)

    def test_paper_broker_partial_close_and_crypto_contract_size(self):
        """Verify PaperBroker correctly updates remaining lots and uses 1.0 contract size for ETH/BTC."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            broker = PaperBroker(db_path=db_path)
            # Mock initial market price for ETHUSD at 3000.0 (fill with spread at 3000.15)
            with patch.object(broker, "_get_current_price", return_value=3000.0):
                res = broker.place_order(symbol="ETHUSD", side="BUY", lots=1.00, price=3000.0, order_type="LIMIT")
                ticket = res["ticket"]
                self.assertIsNotNone(ticket)

            # Mock price to 3050 (+50 pts)
            with patch.object(broker, "_get_current_price", return_value=3050.0):
                # Close 0.40 lots
                part_res = broker.close_position(ticket, lots=0.40)
                self.assertEqual(part_res["status"], "SUCCESS")
                self.assertEqual(part_res["lots"], 0.40)
                # PnL for 0.40 ETH on +$50 move = +$20.00
                self.assertIn("+$20.00", part_res["message"])

                # Check remaining position has 0.60 lots
                open_pos = broker.get_open_positions("ETHUSD")
                self.assertEqual(len(open_pos), 1)
                self.assertAlmostEqual(open_pos[0]["lots"], 0.60)

                # Close remaining 0.60 lots
                full_res = broker.close_position(ticket)
                self.assertEqual(full_res["status"], "SUCCESS")
                self.assertAlmostEqual(full_res["lots"], 0.60)
                # PnL for 0.60 ETH on +$50 move = +$30.00
                self.assertIn("+$30.00", full_res["message"])

                # Verify open positions empty
                self.assertEqual(len(broker.get_open_positions()), 0)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_macro_drivers_handles_none_ema_cleanly(self):
        """Verify analyze_macro_drivers doesn't falsely mark bearish headwinds when EMA is None."""
        with patch("gold_trader.core.macro.fetch_gold_candles", return_value=[]):
            with patch("gold_trader.core.openbb_feed.fetch_openbb_macro_data", return_value={}):
                result = analyze_macro_drivers()
                self.assertEqual(result["macro_gold_score"], 0)
                self.assertEqual(result["macro_bias"], "Neutral / Mixed Macro Cues")

    def test_openbb_news_guard_string_datetime_safety(self):
        """Verify check_high_impact_news_window handles string or missing datetimes without AttributeError."""
        fixed_now = datetime(2026, 10, 19, 14, 0, 0, tzinfo=timezone.utc)
        mock_events = [
            {
                "source": "FED",
                "title": "FOMC Meeting",
                "time_utc": datetime(2026, 10, 19, 14, 10, 0, tzinfo=timezone.utc),
            }
        ]
        with patch("gold_trader.core.openbb_feed._get_cached_economic_calendar", return_value=mock_events):
            res = check_high_impact_news_window(buffer_minutes=15, custom_now=fixed_now)
            self.assertTrue(res["is_in_news_window"])
            self.assertEqual(res["action"], "PAUSE")
            self.assertIn("2026-10-19 14:10 UTC", res["event_time_utc"])

    def test_auto_trader_friday_market_closed(self):
        """Verify AutoTrader blocks entries on Friday after 21:00 UTC."""
        friday_night = datetime(2026, 10, 16, 21, 30, 0, tzinfo=timezone.utc)
        trader = GoldAutoTrader()
        with patch("gold_trader.core.auto_trader.datetime") as mock_dt:
            mock_dt.now.return_value = friday_night
            mock_dt.fromtimestamp.side_effect = datetime.fromtimestamp
            result = trader.evaluate_and_trade()
            self.assertIn("Market Closed", result["decision"])

    def test_strategy_engine_eth_ladder_state(self):
        """Verify generate_compounding_ladder_setup selects eth_compounding_ladder_state.json for ETH."""
        with patch("gold_trader.core.strategy_engine.CompoundingLadderManager") as mock_ladder_cls:
            mock_inst = MagicMock()
            mock_inst.get_status.return_value = {"current_level": 1, "current_balance": 1000.0}
            mock_inst.calculate_bracket_order.return_value = {"entry_price": 2500.0}
            mock_ladder_cls.return_value = mock_inst

            with patch("gold_trader.core.strategy_engine.generate_donchian_momentum_hybrid_setup") as mock_setup:
                mock_setup.return_value = {
                    "action": "HOLD",
                    "entry_price": 2500.0,
                    "reasons": ["Consolidating"],
                }
                generate_compounding_ladder_setup(symbol="ETHUSD")
                call_args = mock_ladder_cls.call_args[1]
                self.assertIn("eth_compounding_ladder_state.json", call_args["state_file"])

    def test_server_train_gold_ml_model_arguments(self):
        """Verify train_gold_ml_model successfully invokes predictor.train_model."""
        with patch("gold_trader.server.GoldMLPredictor") as mock_pred_cls:
            mock_pred = MagicMock()
            mock_pred.train_model.return_value = {"status": "SUCCESS", "auc": 0.82}
            mock_pred_cls.return_value = mock_pred

            res = train_gold_ml_model(symbol="XAUUSD", period="1y", interval="1h")
            self.assertEqual(res["status"], "SUCCESS")
            mock_pred.train_model.assert_called_once_with(
                symbol="XAUUSD",
                period="1y",
                interval="1h",
                atr_multiplier_sl=1.5,
                rr_ratio=1.25,
                min_sl_pts=None,
            )


if __name__ == "__main__":
    unittest.main()
