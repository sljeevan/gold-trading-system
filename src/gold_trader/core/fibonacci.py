"""
Fibonacci Retracement and Golden Pocket (0.618 - 0.786) Analyzer for Gold.
Gold algorithms heavily react to institutional Fibonacci retracement and expansion zones.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple


_FIB_RETRACEMENTS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
_FIB_EXTENSIONS = [1.272, 1.618, 2.0, 2.618]


def find_swing_high_low(
    candles: List[Dict[str, Any]], lookback: int = 50
) -> Tuple[float, float, int, int]:
    """Find the swing high and swing low in the given lookback window."""
    window = candles[-lookback:] if len(candles) >= lookback else candles
    highs = [float(c["high"]) for c in window]
    lows = [float(c["low"]) for c in window]

    max_high = max(highs)
    min_low = min(lows)

    idx_high = highs.index(max_high)
    idx_low = lows.index(min_low)

    return max_high, min_low, idx_high, idx_low


def analyze_gold_fibonacci(
    candles: List[Dict[str, Any]], lookback: int = 50
) -> Dict[str, Any]:
    """
    Calculate Fibonacci levels and detect whether Gold is inside the Golden Pocket.
    """
    if not candles:
        return {}

    current_price = float(candles[-1]["close"])
    swing_high, swing_low, idx_high, idx_low = find_swing_high_low(candles, lookback)

    # If swing high occurred after swing low, it's an uptrend pullback; otherwise downtrend
    trend: Literal["uptrend", "downtrend"] = "uptrend" if idx_high > idx_low else "downtrend"
    diff = swing_high - swing_low

    retrace_levels: Dict[str, float] = {}
    for ratio in _FIB_RETRACEMENTS:
        if trend == "uptrend":
            price = swing_high - ratio * diff
        else:
            price = swing_low + ratio * diff
        retrace_levels[str(ratio)] = round(price, 2)

    ext_levels: Dict[str, float] = {}
    for ratio in _FIB_EXTENSIONS:
        if trend == "uptrend":
            price = swing_high + (ratio - 1.0) * diff
        else:
            price = swing_low - (ratio - 1.0) * diff
        ext_levels[str(ratio)] = round(price, 2)

    # Golden Pocket Zone: 0.618 to 0.786
    fib_618 = retrace_levels["0.618"]
    fib_786 = retrace_levels["0.786"]
    golden_min = min(fib_618, fib_786)
    golden_max = max(fib_618, fib_786)

    is_in_golden_pocket = golden_min <= current_price <= golden_max

    # Nearest Level
    all_levels = sorted([(k, v) for k, v in retrace_levels.items()], key=lambda x: x[1])
    nearest = min(all_levels, key=lambda x: abs(x[1] - current_price))
    nearest_dist_pct = round(((current_price - nearest[1]) / current_price) * 100, 2)

    # Current Zone
    current_zone = "Unknown"
    for i in range(len(all_levels) - 1):
        r1, p1 = all_levels[i]
        r2, p2 = all_levels[i + 1]
        if p1 <= current_price <= p2:
            current_zone = f"Between {r1} (${p1}) and {r2} (${p2})"
            break

    if current_zone == "Unknown":
        if current_price < all_levels[0][1]:
            current_zone = f"Below 1.0 (${all_levels[0][1]}) - Breakdown"
        else:
            current_zone = f"Above 0.0 (${all_levels[-1][1]}) - Breakout / Extension"

    fib_supports = [{"ratio": r, "price": p} for r, p in sorted(all_levels, key=lambda x: x[1], reverse=True) if p < current_price]
    fib_resistances = [{"ratio": r, "price": p} for r, p in sorted(all_levels, key=lambda x: x[1]) if p > current_price]

    return {
        "current_price": current_price,
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "trend": trend,
        "retracement_levels": retrace_levels,
        "extension_levels": ext_levels,
        "golden_pocket_zone": {
            "lower": round(golden_min, 2),
            "upper": round(golden_max, 2),
            "label": "0.618 - 0.786 Golden Pocket",
        },
        "is_in_golden_pocket": is_in_golden_pocket,
        "current_zone": current_zone,
        "nearest_level": {
            "ratio": nearest[0],
            "price": nearest[1],
            "distance_pct": nearest_dist_pct,
        },
        "fib_supports": fib_supports[:3],
        "fib_resistances": fib_resistances[:3],
    }
