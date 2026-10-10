"""
Unit tests for the High-Velocity TOUCH Strategy:
Donchian 10 Touch + SuperTrend (10, 3.0) + 50 EMA + 1.25R Target on the Compounding Ladder.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gold_trader.brokers.base import BaseBroker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.core.compounding_ladder import (
    NOISE_ADAPTIVE_MATRIX,
    CompoundingLadderManager,
    find_closest_level_for_balance,
)
from gold_trader.core.strategy_engine import (
    generate_compounding_ladder_setup,
    generate_donchian_momentum_hybrid_setup,
)


class MockBroker(BaseBroker):
    def __init__(self):
        self.orders = []
        self.modified = []
        self.open_pos = []

    def connect(self) -> bool:
        return True

    def get_account_info(self):
        return {
            "broker": "MockBroker",
            "account_id": "MOCK-XAU",
            "balance": 200.0,
            "equity": 200.0,
            "margin": 0.0,
            "free_margin": 200.0,
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
            "price": price or 4350.0,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "message": "Executed",
        }
        self.orders.append(res)
        return res

    def modify_position(self, ticket, sl=None, tp=None):
        self.modified.append({"ticket": ticket, "sl": sl, "tp": tp})
        return {"status": "SUCCESS", "ticket": ticket, "sl": sl, "tp": tp}

    def close_position(self, ticket):
        return {"status": "SUCCESS", "ticket": ticket}


class TestHighVelocityTouchStrategy(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.temp_state = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_state.close()
        self.broker = MockBroker()

    def tearDown(self):
        for path in (self.temp_db.name, self.temp_state.name):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @patch("gold_trader.core.strategy_engine.fetch_gold_candles")
    def test_donchian_10_touch_bullish_signal(self, mock_fetch):
        # Generate 65 healthy zigzag candles trending upward so 50 EMA and RSI (~60) are valid
        candles = []
        p = 4300.0
        for i in range(65):
            p += 0.8 if (i % 2 == 0) else -0.5
            candles.append({
                "date": f"2026-10-06T12:00:00Z",
                "open": round(p - 0.5, 2),
                "high": round(p + 1.2, 2),
                "low": round(p - 1.0, 2),
                "close": round(p + 0.3, 2),
                "volume": 1000,
            })
        # Last candle penetrates above previous 10-bar Donchian high
        prev_high = max(c["high"] for c in candles[-11:-1])
        candles[-1]["open"] = round(prev_high - 0.2, 2)
        candles[-1]["high"] = round(prev_high + 1.5, 2)  # Touch trigger
        candles[-1]["close"] = round(prev_high + 0.8, 2)
        candles[-1]["low"] = round(prev_high - 0.5, 2)
        mock_fetch.return_value = candles

        setup = generate_donchian_momentum_hybrid_setup(
            symbol="XAUUSD",
            account_balance=200.0,
            timeframe="1h",
            donchian_period=10,
            rr_multiplier=1.25,
            trend_filter_ema=50,
            trigger_mode="TOUCH",
        )

        self.assertNotIn("error", setup)
        self.assertEqual(setup["action"], "BUY")
        self.assertIn("Touch", setup["setup_name"])
        self.assertEqual(setup["risk_reward_t1"], 1.25)
        self.assertGreaterEqual(setup["stop_distance"], 8.00)

    @patch("gold_trader.core.strategy_engine.fetch_gold_candles")
    def test_trend_filter_50_ema_blocks_counter_trend_signal(self, mock_fetch):
        # Generate synthetic downtrend where price is below 50 EMA
        candles = []
        base = 4500.0
        for i in range(40):
            p = base - i * 4.0
            candles.append({
                "date": f"2026-10-06T12:00:00Z",
                "open": p + 1.0,
                "high": p + 2.0,
                "low": p - 2.0,
                "close": p - 1.0,
                "volume": 1000,
            })
        # Try a fake upward spike that touches upper Donchian, but overall price is well below 50 EMA
        candles[-1]["high"] = candles[-1]["open"] + 15.0
        mock_fetch.return_value = candles

        setup = generate_donchian_momentum_hybrid_setup(
            symbol="XAUUSD",
            account_balance=200.0,
            timeframe="1h",
            donchian_period=10,
            rr_multiplier=1.25,
            trend_filter_ema=50,
            trigger_mode="TOUCH",
        )
        # Should be HOLD because price is below 50 EMA and SuperTrend is bearish
        self.assertIn(setup["action"], ("HOLD", "SELL"))

    def test_ladder_manager_200_seed_calibration(self):
        mgr = CompoundingLadderManager(
            state_file=self.temp_state.name,
            ladder_mode="NOISE_ADAPTIVE_LADDER",
            loss_policy="PROGRESSIVE_RATCHET",
        )
        status = mgr.reset_ladder(starting_balance=200.0)

        # $200 maps to Level 5 ($174.90 bracket)
        self.assertEqual(status["current_level"], 5)
        self.assertEqual(status["current_balance"], 200.0)
        self.assertEqual(status["starting_balance"], 200.0)

        # Bracket order calculation for Level 5 ($17.49 planned risk)
        bracket = mgr.calculate_bracket_order(
            current_price=4350.0,
            direction="BUY",
            atr=5.0,  # ATR 5.0 -> 1.5*ATR = 7.50 -> clamped to min $8.00
            min_sl_pts=8.0,
            rr_ratio=1.25,
        )

        self.assertEqual(bracket["sl_pts"], 8.00)
        self.assertEqual(bracket["tp_pts"], 10.00)  # 8.00 * 1.25 = 10.00
        self.assertEqual(bracket["reward_risk_ratio"], 1.25)
        # 17.49 / (8.00 * 100) = 0.0218 -> 0.02 lots
        self.assertEqual(bracket["lots"], 0.02)

    def test_compounding_ladder_progression_and_ratchet_floor(self):
        mgr = CompoundingLadderManager(
            state_file=self.temp_state.name,
            ladder_mode="NOISE_ADAPTIVE_LADDER",
            loss_policy="PROGRESSIVE_RATCHET",
        )
        mgr.reset_ladder(starting_balance=200.0)

        # Win trade #1: advance level
        res1 = mgr.record_trade_result(win=True, realized_pnl=25.0, exit_price=4360.0)
        self.assertEqual(res1["new_level"], 6)
        self.assertEqual(res1["new_balance"], 225.0)

        # Win trade #2: push balance over $250 -> locks $150 floor
        res2 = mgr.record_trade_result(win=True, realized_pnl=35.0, exit_price=4370.0)
        self.assertEqual(res2["new_level"], 7)
        self.assertEqual(res2["new_balance"], 260.0)
        self.assertEqual(res2["ratchet_floor"], 150.0)

        # Simulate subsequent loss: balance drops but cannot breach $150 floor
        res3 = mgr.record_trade_result(win=False, realized_pnl=-20.0, exit_price=4350.0)
        self.assertEqual(res3["new_balance"], 240.0)
        self.assertEqual(res3["ratchet_floor"], 150.0)

    @patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup")
    def test_auto_trader_compounding_touch_execution(self, mock_setup):
        cfg = AutoTraderConfig(
            symbol="XAUUSD",
            mode="COMPOUNDING_TOUCH",
            min_risk_reward=1.25,
            donchian_period=10,
            trend_filter_ema=50,
            trigger_mode="TOUCH",
            send_telegram_alerts=False,
            session_filter=False,
        )
        mock_notifier = MagicMock()
        trader = GoldAutoTrader(
            broker=self.broker,
            config_override=cfg,
            notifier=mock_notifier,
            db_path=self.temp_db.name,
        )

        mock_setup.return_value = {
            "status": "APPROVED_HIGH_CONVICTION",
            "action": "BUY",
            "symbol": "XAUUSD",
            "level": 5,
            "order_type": "MARKET",
            "market_price": 4350.0,
            "entry_price": 4350.0,
            "pullback_discount_pips": 0.0,
            "lots": 0.02,
            "take_profit": 4360.0,
            "stop_loss": 4342.0,
            "breakeven_trigger_price": 4358.0,
            "tp_pips": 100.0,
            "sl_pips": 80.0,
            "expected_profit_usd": 20.0,
            "expected_loss_usd": 16.0,
            "reward_risk_ratio": 1.25,
            "setup_name": "1H Donchian 10 Touch High-Velocity Compounding",
            "quality_score": 92,
            "candle_time": "2026-10-06T12:00:00Z",
        }

        result = trader.evaluate_and_trade()
        self.assertEqual(result["action"], "BUY")
        self.assertIn("EXECUTE", result["decision"])
        self.assertEqual(len(self.broker.orders), 1)
        order = self.broker.orders[0]
        self.assertEqual(order["action"], "BUY")
        self.assertEqual(order["lots"], 0.02)
        self.assertEqual(order["comment"], "AI-Compounding-L5")


if __name__ == "__main__":
    unittest.main()
