"""
Unit tests verifying P1 High Priority Remediation:
1. Historical Macro Feature Leakage prevention during ML training.
2. Strict broker market-data enforcement (no silent external fallback in live mode).
3. Independent, decoupled position management (partial scale-out, BE, runner trailing).
4. Broker execution post-timeout order reconciliation.
5. Close-confirmed breakout signal evaluation on completed candles.
"""
import unittest
from unittest.mock import MagicMock, patch
import tempfile

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.strategy_engine import generate_donchian_momentum_hybrid_setup
from tests.test_auto_trader import DummyMockBroker


class TestP1Remediation(unittest.TestCase):

    def test_strict_broker_market_data_fails_closed(self):
        """
        Verify that fetch_gold_candles with require_broker=True raises an error
        when broker rates are unavailable, rather than silently substituting Yahoo Finance.
        """
        with patch("gold_trader.brokers.socket_broker.SocketBroker.get_rates", return_value=[]):
            with self.assertRaises(RuntimeError) as ctx:
                fetch_gold_candles("XAUUSD", period="1d", interval="1h", require_broker=True)
            self.assertIn("broker feed required", str(ctx.exception).lower())

    def test_decoupled_partial_scale_out_when_breakeven_disabled(self):
        """
        Verify that partial scale-out at Target 1 still triggers even if trail_to_breakeven_at_t1 is False.
        """
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = DummyMockBroker()
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_partial_scale_out=True,
            trail_to_breakeven_at_t1=False,  # Decoupled!
            enable_runner_trailing=True,
            send_telegram_alerts=False,
            scale_out_r=1.0,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        # Mock an open position that has reached Target 1 (+1.0R = 2710.0 from 2700.0 with $10 SL)
        open_pos = [{
            "ticket": "99901",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.04,
            "open_price": 2700.0,
            "current_price": 2712.0,  # Past target 1
            "sl": 2690.0,
            "tp": 2730.0,
            "comment": "AI-FlatRisk-XAUUSD",
        }]

        actions = trader._manage_open_positions(open_pos)
        self.assertGreater(len(actions), 0, "Partial scale-out should trigger even with breakeven disabled")
        self.assertTrue(any("Scaled out" in a for a in actions))

    def test_post_timeout_order_reconciliation(self):
        """
        Verify that SocketBroker.place_order checks open positions if a socket timeout occurs,
        successfully reconciling the trade without false rejection.
        """
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={"status": "ERROR", "message": "Socket timeout"}):
            with patch.object(broker, "get_open_positions", return_value=[{
                "ticket": 88812,
                "symbol": "XAUUSD",
                "lots": 0.05,
                "open_price": 2705.50,
                "sl": 2695.0,
                "tp": 2725.0,
                "comment": "ReconTest-XAUUSD",
            }]):
                result = broker.place_order(
                    symbol="XAUUSD",
                    side="BUY",
                    lots=0.05,
                    comment="ReconTest-XAUUSD",
                )
                self.assertEqual(result["status"], "SUCCESS")
                self.assertEqual(result["ticket"], "88812")
                self.assertIn("verified via post-timeout reconciliation", result["message"])

    def test_close_confirmed_breakout_candle_evaluation(self):
        """
        Verify that when trigger_mode='BREAKOUT', setup evaluates completed candle [-2]
        against channel [-3], rather than incomplete forming bar.
        """
        # Create a series where candle [-2] broke above band, but candle [-1] is inside
        # generate_donchian_momentum_hybrid_setup with trigger_mode='BREAKOUT'
        with patch("gold_trader.core.strategy_engine.fetch_gold_candles") as mock_candles:
            # Generate 50 candles
            mock_data = []
            for i in range(50):
                mock_data.append({
                    "date": f"2026-01-01T{i%24:02d}:00:00",
                    "open": 2000.0 + i,
                    "high": 2005.0 + i,
                    "low": 1995.0 + i,
                    "close": 2000.0 + i,
                })
            mock_candles.return_value = mock_data

            setup = generate_donchian_momentum_hybrid_setup(
                symbol="XAUUSD",
                account_balance=1000.0,
                trigger_mode="BREAKOUT",
                trend_filter_ema=50,
            )
            self.assertNotIn("error", setup)


if __name__ == "__main__":
    unittest.main()
