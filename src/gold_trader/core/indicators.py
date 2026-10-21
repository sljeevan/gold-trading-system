"""
Technical indicators calculator for Gold Trading — zero external C-dependencies.
Pure Python implementations with Wilder's smoothing and precise ATR/Supertrend logic.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def calc_ema(closes: List[float], period: int) -> List[Optional[float]]:
    """Exponential Moving Average."""
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n < period or period <= 0:
        return result
    k = 2.0 / (period + 1.0)
    sma = sum(closes[:period]) / float(period)
    result[period - 1] = sma
    for i in range(period, n):
        result[i] = closes[i] * k + result[i - 1] * (1.0 - k)
    return result


def calc_sma(closes: List[float], period: int) -> List[Optional[float]]:
    """Simple Moving Average."""
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n < period or period <= 0:
        return result
    for i in range(period - 1, n):
        result[i] = sum(closes[i - period + 1 : i + 1]) / float(period)
    return result


def calc_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index with Wilder's smoothing."""
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n <= period:
        return result

    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains) / float(period)
    avg_loss = sum(losses) / float(period)

    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / float(period)
        avg_loss = (avg_loss * (period - 1) + loss) / float(period)
        if avg_loss == 0.0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


def calc_atr(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> List[Optional[float]]:
    """Average True Range (Wilder's smoothing)."""
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n <= period:
        return result

    trs: List[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    atr = sum(trs[:period]) / float(period)
    result[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + trs[i - 1]) / float(period)
        result[i] = atr

    return result


def calc_supertrend(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> Dict[str, List[Optional[float]]]:
    """
    Supertrend ATR Trend Indicator.
    Returns:
      'direction': 1 (Bullish) or -1 (Bearish)
      'upper': Upper trailing resistance band
      'lower': Lower trailing support band
    """
    n = len(closes)
    atr = calc_atr(highs, lows, closes, atr_period)

    direction: List[Optional[int]] = [None] * n
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n

    prev_upper = None
    prev_lower = None
    prev_dir = None

    for i in range(1, n):
        if atr[i] is None:
            continue

        hl2 = (highs[i] + lows[i]) / 2.0
        u = hl2 + multiplier * atr[i]
        l = hl2 - multiplier * atr[i]

        if prev_upper is not None:
            u = min(u, prev_upper) if closes[i - 1] < prev_upper else u
            l = max(l, prev_lower) if closes[i - 1] > prev_lower else l

        upper[i] = u
        lower[i] = l

        if prev_dir is None:
            direction[i] = 1 if closes[i] > u else -1
        elif prev_dir == 1:
            direction[i] = 1 if closes[i] >= l else -1
        else:
            direction[i] = -1 if closes[i] <= u else 1

        prev_upper = u
        prev_lower = l
        prev_dir = direction[i]

    return {"direction": direction, "upper": upper, "lower": lower}


def calc_donchian(
    highs: List[float], lows: List[float], period: int = 20
) -> Dict[str, List[Optional[float]]]:
    """Donchian Channels (Turtle Breakout)."""
    n = len(highs)
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    middle: List[Optional[float]] = [None] * n

    for i in range(period - 1, n):
        u = max(highs[i - period + 1 : i + 1])
        l = min(lows[i - period + 1 : i + 1])
        upper[i] = u
        lower[i] = l
        middle[i] = (u + l) / 2.0

    return {"upper": upper, "lower": lower, "middle": middle}


def calc_bollinger(
    closes: List[float], period: int = 20, std_mult: float = 2.0
) -> Dict[str, List[Optional[float]]]:
    """Bollinger Bands (Upper, Middle SMA, Lower, and Bandwidth BBW)."""
    n = len(closes)
    middle = calc_sma(closes, period)
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    bbw: List[Optional[float]] = [None] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / float(period)
        std = math.sqrt(variance)
        upper[i] = mean + std_mult * std
        lower[i] = mean - std_mult * std
        if mean and mean > 0:
            bbw[i] = (upper[i] - lower[i]) / mean

    return {"upper": upper, "middle": middle, "lower": lower, "bbw": bbw}


def calc_macd(
    closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, List[Optional[float]]]:
    """MACD Line, Signal Line, and Histogram."""
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    n = len(closes)

    macd_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line: List[Optional[float]] = [None] * n
    histogram: List[Optional[float]] = [None] * n

    valid_macd = [(idx, val) for idx, val in enumerate(macd_line) if val is not None]
    if len(valid_macd) >= signal:
        macd_vals = [val for _, val in valid_macd]
        sig_ema = calc_ema(macd_vals, signal)
        for j, (orig_i, _) in enumerate(valid_macd):
            if sig_ema[j] is not None:
                signal_line[orig_i] = sig_ema[j]
                histogram[orig_i] = macd_line[orig_i] - sig_ema[j]

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calc_adx(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> Dict[str, List[Optional[float]]]:
    """Average Directional Index (ADX) measuring trend strength."""
    n = len(closes)
    adx_res: List[Optional[float]] = [None] * n
    plus_di_res: List[Optional[float]] = [None] * n
    minus_di_res: List[Optional[float]] = [None] * n

    if n < period * 2:
        return {"adx": adx_res, "plus_di": plus_di_res, "minus_di": minus_di_res}

    plus_dm: List[float] = []
    minus_dm: List[float] = []
    tr_list: List[float] = []

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

    smoothed_tr = sum(tr_list[:period])
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])

    dx_list: List[float] = []

    for i in range(period, len(tr_list)):
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_list[i]
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]

        plus_di = (smoothed_plus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0.0
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0.0

        plus_di_res[i + 1] = plus_di
        minus_di_res[i + 1] = minus_di

        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum * 100) if di_sum > 0 else 0.0
        dx_list.append(dx)

        if len(dx_list) == period:
            adx_val = sum(dx_list) / period
            adx_res[i + 1] = adx_val
        elif len(dx_list) > period:
            adx_val = (adx_res[i] * (period - 1) + dx) / period
            adx_res[i + 1] = adx_val

    return {"adx": adx_res, "plus_di": plus_di_res, "minus_di": minus_di_res}


def extract_all_indicators(candles: List[Dict]) -> Dict[str, Any]:
    """Compute the full suite of indicators on a candle series."""
    if not candles:
        return {}

    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]

    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    ema200 = calc_ema(closes, 200)
    rsi14 = calc_rsi(closes, 14)
    atr14 = calc_atr(highs, lows, closes, 14)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    dc = calc_donchian(highs, lows, 20)
    bb = calc_bollinger(closes, 20, 2.0)
    macd = calc_macd(closes, 12, 26, 9)
    adx = calc_adx(highs, lows, closes, 14)

    latest_close = closes[-1]
    latest_atr = atr14[-1] or (latest_close * 0.01)

    return {
        "price": latest_close,
        "ema20": ema20[-1],
        "ema50": ema50[-1],
        "ema200": ema200[-1],
        "rsi": rsi14[-1],
        "atr": latest_atr,
        "supertrend_direction": st["direction"][-1],
        "supertrend_upper": st["upper"][-1],
        "supertrend_lower": st["lower"][-1],
        "donchian_upper": dc["upper"][-1],
        "donchian_middle": dc["middle"][-1],
        "donchian_lower": dc["lower"][-1],
        "bb_upper": bb["upper"][-1],
        "bb_middle": bb["middle"][-1],
        "bb_lower": bb["lower"][-1],
        "bbw": bb["bbw"][-1],
        "macd_line": macd["macd"][-1],
        "macd_signal": macd["signal"][-1],
        "macd_hist": macd["histogram"][-1],
        "adx": adx["adx"][-1],
        "plus_di": adx["plus_di"][-1],
        "minus_di": adx["minus_di"][-1],
    }
