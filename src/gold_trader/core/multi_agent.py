"""
Multi-Agent Debate Pipeline for Gold Trading.
Agents:
1. Technical Structure Analyst
2. Macro & US Dollar/Yields Analyst
3. Risk & Volatility Manager
"""
from __future__ import annotations

from typing import Any, Dict

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.fibonacci import analyze_gold_fibonacci
from gold_trader.core.indicators import extract_all_indicators
from gold_trader.core.macro import analyze_macro_drivers


def run_gold_multi_agent_debate(
    symbol: str = "XAUUSD",
    timeframe: str = "1h",
) -> Dict[str, Any]:
    """
    Execute a structured 3-agent debate to reach an institutional trade consensus on Gold.
    """
    candles = fetch_gold_candles(symbol, period="1mo", interval=timeframe)
    if not candles or len(candles) < 30:
        return {"error": f"Failed to retrieve candle data for {symbol}"}

    ind = extract_all_indicators(candles)
    fib = analyze_gold_fibonacci(candles, lookback=40)
    macro = analyze_macro_drivers()

    price = float(ind["price"])
    ema20 = ind["ema20"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    rsi = ind["rsi"]
    st_dir = ind["supertrend_direction"]
    atr = ind["atr"]

    # ── Agent 1: Technical Structure Analyst ─────────────────────────────
    tech_score = 0
    tech_obs = []

    if ema20 and ema50 and ema200:
        if price > ema20 > ema50 > ema200:
            tech_score += 2
            tech_obs.append("Bullish stack: Price > EMA20 > EMA50 > EMA200")
        elif price < ema20 < ema50 < ema200:
            tech_score -= 2
            tech_obs.append("Bearish stack: Price < EMA20 < EMA50 < EMA200")

    if st_dir == 1:
        tech_score += 1
        tech_obs.append("Supertrend is Bullish (Green)")
    elif st_dir == -1:
        tech_score -= 1
        tech_obs.append("Supertrend is Bearish (Red)")

    if fib.get("is_in_golden_pocket"):
        tech_score += 1 if tech_score >= 0 else -1
        tech_obs.append("Price inside 0.618-0.786 Golden Pocket reversal zone")

    if rsi and 50 <= rsi <= 68:
        tech_score += 1
        tech_obs.append(f"RSI {rsi:.1f} shows healthy bullish momentum without exhaustion")
    elif rsi and 32 <= rsi <= 50:
        tech_score -= 1
        tech_obs.append(f"RSI {rsi:.1f} shows bearish control")

    agent_tech = {
        "role": "Technical Structure Analyst",
        "stance": "Bullish" if tech_score > 0 else "Bearish" if tech_score < 0 else "Neutral",
        "score": tech_score,
        "observations": tech_obs,
    }

    # ── Agent 2: Macro & Dollar/Yield Analyst ────────────────────────────
    macro_score = macro.get("macro_gold_score", 0)
    agent_macro = {
        "role": "Macro & Dollar/Yield Analyst",
        "stance": "Bullish" if macro_score > 0 else "Bearish" if macro_score < 0 else "Neutral",
        "score": macro_score,
        "observations": macro.get("signals", []),
    }

    # ── Agent 3: Risk & Volatility Manager ────────────────────────────────
    risk_score = 0
    risk_warnings = []
    atr_pct = (atr / price) * 100.0 if price else 0.0

    if atr_pct > 1.2:
        risk_score -= 1
        risk_warnings.append(f"Elevated ATR volatility ({atr_pct:.2f}% of price) - reduce position size")
    else:
        risk_warnings.append(f"Normal volatility profile (ATR {atr:.2f})")

    if rsi and rsi > 75:
        risk_score -= 2
        risk_warnings.append(f"RSI {rsi:.1f} severely overbought - high pullback risk for longs")
    elif rsi and rsi < 25:
        risk_score -= 2
        risk_warnings.append(f"RSI {rsi:.1f} severely oversold - high bounce risk for shorts")

    agent_risk = {
        "role": "Risk & Volatility Manager",
        "risk_level": "High" if risk_score < -1 else "Medium" if risk_score == -1 else "Low",
        "risk_adjustment": risk_score,
        "warnings": risk_warnings,
    }

    # ── Final Consensus ──────────────────────────────────────────────────
    total_score = tech_score + macro_score + risk_score

    if total_score >= 3 and agent_risk["risk_level"] != "High":
        decision = "STRONG BUY"
        confidence = "High"
    elif total_score > 0:
        decision = "BUY"
        confidence = "Medium"
    elif total_score <= -3 and agent_risk["risk_level"] != "High":
        decision = "STRONG SELL"
        confidence = "High"
    elif total_score < 0:
        decision = "SELL"
        confidence = "Medium"
    else:
        decision = "HOLD / STAND ASIDE"
        confidence = "Low"

    return {
        "framework": "Gold Multi-Agent Consensus Pipeline",
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "agents": {
            "technical_analyst": agent_tech,
            "macro_analyst": agent_macro,
            "risk_manager": agent_risk,
        },
        "consensus": {
            "decision": decision,
            "confidence": confidence,
            "net_score": total_score,
            "summary": (
                f"Technical ({tech_score:+d}) + Macro ({macro_score:+d}) "
                f"+ Risk ({risk_score:+d}) = Net {total_score:+d}"
            ),
        },
    }
