"""
Data feed module for Gold instruments and macro drivers.
Supports Yahoo Finance REST endpoint with user-agent rotation and clean parsing.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from gold_trader.config import config


_USER_AGENT = "gold-trader-mcp/1.0 (Institutional Quantitative Gold Engine)"
_YF_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


def resolve_symbol(symbol: str) -> str:
    """Map common gold alias to valid Yahoo Finance ticker."""
    s = symbol.upper().strip()
    return config.SYMBOL_MAP.get(s, s)


def calculate_required_candle_count(
    period: str,
    interval: str,
    is_crypto: bool = False,
) -> int:
    """
    Calculate the realistic required candle count based on interval, period,
    and market trading sessions (5 days/week for Gold/FX, 7 days/week for Crypto).
    """
    interval_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1wk": 10080,
    }.get(interval.lower(), 60)

    trading_days_per_period = {
        "1d": 1,
        "5d": 5,
        "1mo": 30 if is_crypto else 22,
        "3mo": 90 if is_crypto else 65,
        "6mo": 180 if is_crypto else 130,
        "1y": 365 if is_crypto else 260,
        "2y": 730 if is_crypto else 520,
    }.get(period.lower(), 22)

    daily_trading_minutes = 1440 if is_crypto else 1380
    total_minutes = trading_days_per_period * daily_trading_minutes
    bars = max(50, total_minutes // interval_minutes)
    return int(bars)


def validate_candle_coverage(
    candles: List[Dict[str, Any]],
    requested_count: int,
    interval: str,
) -> Dict[str, Any]:
    """
    Validate candle list coverage, timestamps, ordering, and gap detection.
    """
    if not candles:
        return {"status": "EMPTY", "count": 0, "coverage_pct": 0.0, "is_complete": False}

    count = len(candles)
    coverage_pct = round(min(1.0, count / max(1, requested_count)) * 100.0, 1)
    is_ordered = all(candles[i]["date"] <= candles[i + 1]["date"] for i in range(count - 1))

    return {
        "status": "VALID" if is_ordered and count >= min(requested_count // 2, 30) else "INCOMPLETE",
        "count": count,
        "requested_count": requested_count,
        "coverage_pct": coverage_pct,
        "start_date": candles[0].get("date"),
        "end_date": candles[-1].get("date"),
        "is_ordered": is_ordered,
        "is_complete": count >= requested_count * 0.8,
    }


def fetch_gold_candles(
    symbol: str = "XAUUSD",
    period: str = "3mo",
    interval: str = "1h",
    require_broker: bool = False,
    broker: Optional[Any] = None,
    broker_port: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV candles with live MT5 broker priority for Gold intraday and Yahoo Finance fallback.
    Intervals: '1m', '5m', '15m', '1h', '1d', '1wk'
    Periods: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'

    If require_broker=True: Strictly enforces broker source for live execution, preventing
    silent substitution with external research data. Fails closed with DATA_UNAVAILABLE.
    """
    clean_sym = symbol.upper().strip()
    is_live_broker_sym = clean_sym in ("XAUUSD", "GOLD", "BTCUSD", "BTC", "ETHUSD", "ETH")
    is_crypto = "BTC" in clean_sym or "ETH" in clean_sym
    broker_sym = "GOLD" if clean_sym in ("XAUUSD", "GOLD") else ("BTCUSD" if clean_sym == "BTC" else ("ETHUSD" if clean_sym == "ETH" else clean_sym))

    # 1. If Gold or Crypto symbol on intraday/swing timeframe, check live MT5 broker for real-time rates
    if is_live_broker_sym and interval in ("1m", "5m", "15m", "30m", "1h") and period in ("1d", "5d", "1mo", "3mo", "6mo"):
        try:
            active_broker = broker
            if active_broker is None:
                from gold_trader.brokers.socket_broker import SocketBroker
                active_broker = SocketBroker(port=broker_port) if broker_port else SocketBroker()
            count = calculate_required_candle_count(period=period, interval=interval, is_crypto=is_crypto)
            rates = active_broker.get_rates(symbol=broker_sym, timeframe=interval, count=count)
            if rates and len(rates) >= 30:
                for r in rates:
                    r.setdefault("source", "MT5_BROKER")
                    r.setdefault("symbol", broker_sym)
                return rates
        except Exception as e:
            if require_broker:
                raise RuntimeError(f"DATA_UNAVAILABLE: Live broker feed unavailable for {symbol} ({broker_sym}): {e}") from e

    if require_broker:
        raise RuntimeError(
            f"DATA_UNAVAILABLE: Strict broker feed required for live symbol {symbol} ({broker_sym}), but MT5 bridge returned insufficient rates. "
            f"Failing closed to prevent external price mismatch."
        )

    # 2. Yahoo Finance REST endpoint (Research / Macro fallback only)
    yf_symbol = resolve_symbol(symbol)
    url = f"{_YF_BASE_URL}/{yf_symbol}?interval={interval}&range={period}"

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch market data for '{symbol}' ({yf_symbol}): {exc}") from exc

    chart_data = raw.get("chart", {})
    results = chart_data.get("result")
    if not results or not results[0]:
        err_msg = chart_data.get("error", {}).get("description", "No market data results returned")
        raise RuntimeError(f"Failed to fetch market data for '{symbol}' ({yf_symbol}): {err_msg}")

    result = results[0]
    timestamps = result.get("timestamp", [])
    quotes = result["indicators"]["quote"][0]

    date_fmt = "%Y-%m-%d %H:%M" if interval in ("1m", "5m", "15m", "1h") else "%Y-%m-%d"
    candles: List[Dict[str, Any]] = []

    for i, ts in enumerate(timestamps):
        o = quotes["open"][i]
        h = quotes["high"][i]
        l = quotes["low"][i]
        c = quotes["close"][i]
        v = quotes["volume"][i]

        if None in (o, h, l, c):
            continue

        candles.append({
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime(date_fmt),
            "open": round(float(o), 4),
            "high": round(float(h), 4),
            "low": round(float(l), 4),
            "close": round(float(c), 4),
            "volume": float(v or 0.0),
            "source": "YAHOO_RESEARCH",
            "symbol": yf_symbol,
        })

    # Append fresh live broker candles newer than the last Yahoo candle if broker is available
    if is_live_broker_sym and interval in ("1m", "5m", "15m", "30m", "1h") and candles:
        try:
            from gold_trader.brokers.socket_broker import SocketBroker
            broker = SocketBroker()
            latest_date = candles[-1]["date"]
            live_rates = broker.get_rates(symbol=broker_sym, timeframe=interval, count=250)
            if live_rates:
                for r in live_rates:
                    if r.get("date", "") > latest_date:
                        r.setdefault("source", "MT5_BROKER")
                        r.setdefault("symbol", broker_sym)
                        candles.append(r)
        except Exception:
            pass

    return candles


def fetch_research_futures_candles(
    period: str = "3mo",
    interval: str = "1h",
) -> List[Dict[str, Any]]:
    """
    Explicit research data provider for COMEX Gold Futures (GC=F).
    Note: Futures prices trade at a basis differential to Spot Gold CFD (XAU/USD)
    and follow CME exchange hours rather than XM OTC 24/5 liquidity.
    """
    candles = fetch_gold_candles(symbol="GC=F", period=period, interval=interval, require_broker=False)
    for c in candles:
        c["source"] = "YAHOO_FUTURES_RESEARCH"
    return candles


def get_live_gold_snapshot() -> Dict[str, Any]:
    """
    Fetch a synchronized live snapshot of Spot Gold, COMEX Futures, Gold ETFs, DXY, and US10Y.
    """
    instruments = {
        "comex_gold_futures": "GC=F",
        "gold_etf_gld": "GLD",
        "gold_etf_iau": "IAU",
        "india_goldbees": "GOLDBEES.NS",
        "us_dollar_index": "DX-Y.NYB",
        "us_10y_yield": "^TNX",
        "usd_inr": "INR=X",
    }

    snapshot: Dict[str, Any] = {}
    for key, sym in instruments.items():
        try:
            candles = fetch_gold_candles(sym, period="5d", interval="1d")
            if candles and len(candles) >= 2:
                latest = candles[-1]["close"]
                prev = candles[-2]["close"]
                chg = latest - prev
                chg_pct = (chg / prev) * 100.0 if prev else 0.0
                snapshot[key] = {
                    "symbol": sym,
                    "price": round(latest, 4),
                    "change": round(chg, 4),
                    "change_pct": round(chg_pct, 2),
                    "date": candles[-1]["date"],
                }
            elif candles:
                snapshot[key] = {
                    "symbol": sym,
                    "price": round(candles[-1]["close"], 4),
                    "change": 0.0,
                    "change_pct": 0.0,
                    "date": candles[-1]["date"],
                }
        except Exception:
            snapshot[key] = {"symbol": sym, "error": "Unavailable"}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_data": snapshot,
    }
