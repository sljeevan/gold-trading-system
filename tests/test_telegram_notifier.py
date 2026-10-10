"""
Unit tests for the Telegram Notifier Module.
"""
import unittest
from unittest.mock import MagicMock, patch

from gold_trader.core.telegram_notifier import TelegramNotifier


class TestTelegramNotifier(unittest.TestCase):
    def setUp(self):
        self.notifier = TelegramNotifier(
            bot_token="123456789:TEST_TOKEN",
            chat_id="987654321",
            enabled=True,
        )

    @patch("gold_trader.core.telegram_notifier.requests.post")
    def test_send_message_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        res = self.notifier.send_message("Test message")
        self.assertTrue(res)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["chat_id"], "987654321")
        self.assertIn("Test message", kwargs["json"]["text"])

    @patch("gold_trader.core.telegram_notifier.requests.post")
    def test_send_trade_entry(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        order = {
            "ticket": "1001",
            "symbol": "XAUUSD",
            "action": "BUY",
            "lots": 0.05,
            "price": 4350.0,
        }
        setup = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "stop_loss": 4330.0,
            "target_1": 4380.0,
            "target_2": 4410.0,
            "risk_reward_t1": 1.5,
            "quality_score": 90,
            "setup_name": "Golden Pocket Retracement",
            "reasons": ["Price in 0.618 Fib zone", "RSI is healthy"],
        }

        res = self.notifier.send_trade_entry(order, setup)
        self.assertTrue(res)
        args, kwargs = mock_post.call_args
        text = kwargs["json"]["text"]
        self.assertIn("NEW BUY TRADE EXECUTED", text)
        self.assertIn("#1001", text)
        self.assertIn("0.05 lots", text)
        self.assertIn("$4350.00", text)
        self.assertIn("Golden Pocket Retracement", text)

    @patch("gold_trader.core.telegram_notifier.requests.post")
    def test_send_trade_exit_profit(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        res = self.notifier.send_trade_exit(
            ticket="1001",
            symbol="XAUUSD",
            side="BUY",
            lots=0.05,
            entry_price=4350.0,
            close_price=4380.0,
            pnl=150.0,
            account_balance=10150.0,
        )
        self.assertTrue(res)
        args, kwargs = mock_post.call_args
        text = kwargs["json"]["text"]
        self.assertIn("PROFITABLE TRADE", text)
        self.assertIn("+$150.00", text)
        self.assertIn("$10,150.00", text)

    def test_disabled_notifier_skips(self):
        disabled_notifier = TelegramNotifier(enabled=False)
        self.assertFalse(disabled_notifier.send_message("Should not send"))


if __name__ == "__main__":
    unittest.main()
