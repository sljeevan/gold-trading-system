"""
Macro economic confluence analyzer for Gold.
Gold strongly moves inversely to the US Dollar Index (DXY) and US 10-Year Real Yields (^TNX).
"""
from __future__ import annotations

from typing import Any, Dict

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_rsi


def analyze_macro_drivers() -> Dict[str, Any]:
    """
    Evaluate DXY (Dollar Index) and US10Y (Treasury Yield) to determine Macro Gold Bias.
    """
    score = 0  # Negative = Bearish for Gold, Positive = Bullish for Gold
    signals = []

    # 1. US Dollar Index (DXY) Analysis
    dxy_status = "Neutral"
    try:
        dxy_candles = fetch_gold_candles("DX-Y.NYB", period="1mo", interval="1d")
        if dxy_candles and len(dxy_candles) >= 20:
            dxy_closes = [c["close"] for c in dxy_candles]
            dxy_ema20 = calc_ema(dxy_closes, 20)[-1]
            dxy_rsi = calc_rsi(dxy_closes, 14)[-1]
            dxy_price = dxy_closes[-1]

            if dxy_ema20 and dxy_price < dxy_ema20:
                score += 2
                dxy_status = "Bearish (Weak Dollar -> Bullish Gold)"
                signals.append("DXY trading below 20-day EMA (Dollar weakness supports Gold)")
            else:
                score -= 2
                dxy_status = "Bullish (Strong Dollar -> Bearish Gold)"
                signals.append("DXY trading above 20-day EMA (Dollar strength headwinds for Gold)")

            if dxy_rsi and dxy_rsi > 65:
                score += 1
                signals.append("DXY overbought (Potential dollar pullback favors Gold)")
            elif dxy_rsi and dxy_rsi < 35:
                score -= 1
                signals.append("DXY oversold (Potential dollar bounce threatens Gold)")
    except Exception as e:
        dxy_status = f"Error fetching DXY: {e}"

    # 2. US 10-Year Yield Analysis
    yield_status = "Neutral"
    try:
        tnx_candles = fetch_gold_candles("^TNX", period="1mo", interval="1d")
        if tnx_candles and len(tnx_candles) >= 20:
            tnx_closes = [c["close"] for c in tnx_candles]
            tnx_ema20 = calc_ema(tnx_closes, 20)[-1]
            tnx_price = tnx_closes[-1]

            if tnx_ema20 and tnx_price < tnx_ema20:
                score += 2
                yield_status = "Falling Yields (Lower opportunity cost -> Bullish Gold)"
                signals.append("US 10Y Yields falling below 20-day EMA (Bullish for non-yielding Gold)")
            else:
                score -= 2
                yield_status = "Rising Yields (Higher opportunity cost -> Bearish Gold)"
                signals.append("US 10Y Yields rising above 20-day EMA (Bearish pressure on Gold)")
    except Exception as e:
        yield_status = f"Error fetching US10Y: {e}"

    # 3. OpenBB Macro Confluence (Yield Curve Spread & CFTC COT Positioning)
    openbb_data = {}
    try:
        from gold_trader.core.openbb_feed import fetch_openbb_macro_data
        openbb_data = fetch_openbb_macro_data()
        obb_score = openbb_data.get("macro_confluence_score", 0)
        score += obb_score
        obb_signals = openbb_data.get("signals", [])
        signals.extend(obb_signals)
    except Exception as e:
        signals.append(f"OpenBB feed note: {e}")

    if score >= 2:
        macro_bias = "Bullish Gold Tailwinds"
    elif score <= -2:
        macro_bias = "Bearish Gold Headwinds"
    else:
        macro_bias = "Neutral / Mixed Macro Cues"

    return {
        "macro_gold_score": score,
        "macro_bias": macro_bias,
        "dxy_analysis": dxy_status,
        "us10y_analysis": yield_status,
        "openbb_macro": openbb_data,
        "signals": signals,
    }

