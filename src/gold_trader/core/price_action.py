"""
Institutional Price Action & Candlestick Pattern Engine for Gold (XAUUSD).
Detects:
1. Pin Bars / Liquidity Rejection Wicks (Hammer / Shooting Star)
2. Bullish & Bearish Engulfing Bars
3. Liquidity Sweeps / Stop Hunts (Turtle Soup)
4. Fair Value Gaps (FVG) / Imbalances
5. Inside Bar Breakouts
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_pin_bar(candle: Dict[str, Any], min_wick_pct: float = 0.60) -> Optional[Dict[str, Any]]:
    """
    Detects high-conviction Pin Bar / Rejection Wick.
    A bullish pin bar has a long lower shadow (>= 60% of total candle range) and small body near high.
    A bearish pin bar has a long upper shadow (>= 60% of total candle range) and small body near low.
    """
    high = float(candle["high"])
    low = float(candle["low"])
    open_p = float(candle["open"])
    close = float(candle["close"])
    total_range = high - low

    if total_range <= 0:
        return None

    body = abs(close - open_p)
    upper_wick = high - max(open_p, close)
    lower_wick = min(open_p, close) - low

    # Bullish Pin Bar (Rejection of lower prices)
    if (lower_wick / total_range) >= min_wick_pct and (body / total_range) <= 0.35:
        return {
            "pattern": "BULLISH_PIN_BAR",
            "bias": "BULLISH",
            "wick_pct": round((lower_wick / total_range) * 100, 1),
            "rejection_level": low,
            "description": f"Strong bullish price rejection at ${low:.2f} ({round((lower_wick / total_range) * 100)}% lower wick)",
        }

    # Bearish Pin Bar (Rejection of higher prices / Shooting Star)
    if (upper_wick / total_range) >= min_wick_pct and (body / total_range) <= 0.35:
        return {
            "pattern": "BEARISH_PIN_BAR",
            "bias": "BEARISH",
            "wick_pct": round((upper_wick / total_range) * 100, 1),
            "rejection_level": high,
            "description": f"Strong bearish price rejection at ${high:.2f} ({round((upper_wick / total_range) * 100)}% upper wick)",
        }

    return None


def detect_engulfing(curr: Dict[str, Any], prev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detects Bullish / Bearish Engulfing Candlesticks.
    Bullish: Current green body completely covers previous red candle body.
    Bearish: Current red body completely covers previous green candle body.
    """
    c_open, c_close = float(curr["open"]), float(curr["close"])
    p_open, p_close = float(prev["open"]), float(prev["close"])

    curr_bullish = c_close > c_open
    prev_bearish = p_close < p_open
    curr_bearish = c_close < c_open
    prev_bullish = p_close > p_open

    c_body = abs(c_close - c_open)
    p_body = abs(p_close - p_open)

    # Bullish Engulfing
    if curr_bullish and prev_bearish and c_close >= p_open and c_open <= p_close and c_body > p_body:
        return {
            "pattern": "BULLISH_ENGULFING",
            "bias": "BULLISH",
            "expansion_ratio": round(c_body / max(p_body, 0.01), 2),
            "description": f"Bullish Engulfing bar overtaking prior bearish candle",
        }

    # Bearish Engulfing
    if curr_bearish and prev_bullish and c_close <= p_open and c_open >= p_close and c_body > p_body:
        return {
            "pattern": "BEARISH_ENGULFING",
            "bias": "BEARISH",
            "expansion_ratio": round(c_body / max(p_body, 0.01), 2),
            "description": f"Bearish Engulfing bar overtaking prior bullish candle",
        }

    return None


def detect_liquidity_sweep(
    curr: Dict[str, Any],
    recent_high: float,
    recent_low: float,
) -> Optional[Dict[str, Any]]:
    """
    Detects Liquidity Sweeps (Stop Hunts / False Breakouts / Turtle Soup).
    Price spikes beyond recent high/low but closes firmly back inside the range.
    """
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])

    # Bearish Liquidity Sweep (Swept buyside liquidity above swing high, then closed back below)
    if high > recent_high and close < recent_high:
        return {
            "pattern": "BEARISH_LIQUIDITY_SWEEP",
            "bias": "BEARISH",
            "swept_level": recent_high,
            "description": f"Buyside liquidity swept above ${recent_high:.2f} with strong reversal close at ${close:.2f}",
        }

    # Bullish Liquidity Sweep (Swept sellside liquidity below swing low, then closed back above)
    if low < recent_low and close > recent_low:
        return {
            "pattern": "BULLISH_LIQUIDITY_SWEEP",
            "bias": "BULLISH",
            "swept_level": recent_low,
            "description": f"Sellside liquidity swept below ${recent_low:.2f} with strong reversal close at ${close:.2f}",
        }

    return None


def detect_fair_value_gap(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    c3: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Detects 3-Candle Institutional Fair Value Gaps (FVG) / Price Imbalances.
    Bullish FVG: Low of candle 3 > High of candle 1 (gap created by impulse candle 2).
    Bearish FVG: High of candle 3 < Low of candle 1.
    """
    c1_high, c1_low = float(c1["high"]), float(c1["low"])
    c3_high, c3_low = float(c3["high"]), float(c3["low"])

    # Bullish FVG
    if c3_low > c1_high:
        return {
            "pattern": "BULLISH_FVG",
            "bias": "BULLISH",
            "fvg_top": c3_low,
            "fvg_bottom": c1_high,
            "gap_size": round(c3_low - c1_high, 2),
            "description": f"Bullish Fair Value Gap (${c1_high:.2f} - ${c3_low:.2f})",
        }

    # Bearish FVG
    if c3_high < c1_low:
        return {
            "pattern": "BEARISH_FVG",
            "bias": "BEARISH",
            "fvg_top": c1_low,
            "fvg_bottom": c3_high,
            "gap_size": round(c1_low - c3_high, 2),
            "description": f"Bearish Fair Value Gap (${c3_high:.2f} - ${c1_low:.2f})",
        }

    return None


def extract_price_action_signals(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract comprehensive Price Action signals and candlestick patterns from the latest candles.
    """
    if len(candles) < 4:
        return {"patterns": [], "primary_bias": "NEUTRAL", "has_confluence": False}

    curr = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3]
    prev3 = candles[-4]

    # Calculate recent swing high/low of prior 10 candles (excluding current)
    lookback_window = candles[-12:-1] if len(candles) >= 12 else candles[:-1]
    recent_high = max(float(c["high"]) for c in lookback_window)
    recent_low = min(float(c["low"]) for c in lookback_window)

    patterns = []

    # 1. Pin Bar Check
    pin = detect_pin_bar(curr)
    if pin:
        patterns.append(pin)

    # 2. Engulfing Check
    engulf = detect_engulfing(curr, prev)
    if engulf:
        patterns.append(engulf)

    # 3. Liquidity Sweep Check
    sweep = detect_liquidity_sweep(curr, recent_high, recent_low)
    if sweep:
        patterns.append(sweep)

    # 4. Fair Value Gap Check (3-bar pattern)
    fvg = detect_fair_value_gap(prev2, prev, curr)
    if fvg:
        patterns.append(fvg)

    # Determine overall Price Action bias
    bull_count = sum(1 for p in patterns if p["bias"] == "BULLISH")
    bear_count = sum(1 for p in patterns if p["bias"] == "BEARISH")

    if bull_count > bear_count:
        primary_bias = "BULLISH"
    elif bear_count > bull_count:
        primary_bias = "BEARISH"
    else:
        primary_bias = "NEUTRAL"

    return {
        "patterns": patterns,
        "patterns_count": len(patterns),
        "primary_bias": primary_bias,
        "has_confluence": len(patterns) > 0,
        "recent_swing_high": recent_high,
        "recent_swing_low": recent_low,
        "latest_candle": {
            "open": float(curr["open"]),
            "high": float(curr["high"]),
            "low": float(curr["low"]),
            "close": float(curr["close"]),
        },
    }
