"""
Unit tests for the Institutional Price Action & Candlestick Pattern Engine.
"""
import unittest

from gold_trader.core.price_action import (
    detect_engulfing,
    detect_fair_value_gap,
    detect_liquidity_sweep,
    detect_pin_bar,
    extract_price_action_signals,
)


class TestPriceAction(unittest.TestCase):
    def test_bullish_pin_bar(self):
        # Long lower shadow: Low 4300, Open 4340, Close 4345, High 4350 (Range: 50, Lower wick: 40 = 80%)
        candle = {"open": 4340.0, "high": 4350.0, "low": 4300.0, "close": 4345.0}
        pin = detect_pin_bar(candle)
        self.assertIsNotNone(pin)
        self.assertEqual(pin["pattern"], "BULLISH_PIN_BAR")
        self.assertEqual(pin["bias"], "BULLISH")

    def test_bearish_pin_bar(self):
        # Long upper shadow: Low 4300, Open 4310, Close 4305, High 4360 (Range: 60, Upper wick: 50 = 83.3%)
        candle = {"open": 4310.0, "high": 4360.0, "low": 4300.0, "close": 4305.0}
        pin = detect_pin_bar(candle)
        self.assertIsNotNone(pin)
        self.assertEqual(pin["pattern"], "BEARISH_PIN_BAR")
        self.assertEqual(pin["bias"], "BEARISH")

    def test_bullish_engulfing(self):
        prev = {"open": 4330.0, "high": 4335.0, "low": 4315.0, "close": 4320.0}  # Red bar (4330 -> 4320)
        curr = {"open": 4318.0, "high": 4345.0, "low": 4315.0, "close": 4340.0}  # Big Green bar (4318 -> 4340)
        engulf = detect_engulfing(curr, prev)
        self.assertIsNotNone(engulf)
        self.assertEqual(engulf["pattern"], "BULLISH_ENGULFING")
        self.assertEqual(engulf["bias"], "BULLISH")

    def test_liquidity_sweep(self):
        recent_high = 4350.0
        recent_low = 4300.0
        # Spikes to 4358 but closes at 4342
        candle = {"open": 4335.0, "high": 4358.0, "low": 4330.0, "close": 4342.0}
        sweep = detect_liquidity_sweep(candle, recent_high, recent_low)
        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["pattern"], "BEARISH_LIQUIDITY_SWEEP")

    def test_fair_value_gap(self):
        c1 = {"open": 4300.0, "high": 4310.0, "low": 4295.0, "close": 4305.0}
        c2 = {"open": 4305.0, "high": 4340.0, "low": 4305.0, "close": 4338.0}  # Big impulse
        c3 = {"open": 4338.0, "high": 4350.0, "low": 4320.0, "close": 4345.0}  # Low is 4320 > c1.high (4310)
        fvg = detect_fair_value_gap(c1, c2, c3)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg["pattern"], "BULLISH_FVG")
        self.assertEqual(fvg["gap_size"], 10.0)

    def test_extract_signals_integration(self):
        candles = [
            {"date": "2026-09-01 10:00", "open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0},
            {"date": "2026-09-01 11:00", "open": 4305.0, "high": 4315.0, "low": 4300.0, "close": 4310.0},
            {"date": "2026-09-01 12:00", "open": 4310.0, "high": 4320.0, "low": 4305.0, "close": 4315.0},
            {"date": "2026-09-01 13:00", "open": 4315.0, "high": 4325.0, "low": 4300.0, "close": 4322.0},
        ]
        sig = extract_price_action_signals(candles)
        self.assertIn("patterns", sig)
        self.assertIn("primary_bias", sig)


if __name__ == "__main__":
    unittest.main()
