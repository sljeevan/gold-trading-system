"""
Type definitions and data structures for the Gold Trading System.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict


class Candle(TypedDict):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorsResult(TypedDict, total=False):
    rsi: Optional[float]
    ema_fast: Optional[float]
    ema_mid: Optional[float]
    ema_slow: Optional[float]
    supertrend_direction: Optional[int]  # 1 (Bullish) or -1 (Bearish)
    supertrend_upper: Optional[float]
    supertrend_lower: Optional[float]
    donchian_upper: Optional[float]
    donchian_middle: Optional[float]
    donchian_lower: Optional[float]
    bb_upper: Optional[float]
    bb_middle: Optional[float]
    bb_lower: Optional[float]
    bbw: Optional[float]
    atr: Optional[float]
    adx: Optional[float]
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]


class FibonacciAnalysis(TypedDict):
    swing_high: float
    swing_low: float
    trend: Literal["uptrend", "downtrend"]
    retracement_levels: Dict[str, float]
    extension_levels: Dict[str, float]
    current_zone: str
    golden_pocket_zone: Dict[str, float]
    is_in_golden_pocket: bool
    nearest_level: Dict[str, Any]
    fib_supports: List[Dict[str, float]]
    fib_resistances: List[Dict[str, float]]


class TradeSetup(TypedDict):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    bias: str
    setup_name: str
    entry_price: float
    stop_loss: float
    stop_distance: float
    stop_distance_pips: float
    target_1: float
    target_2: float
    risk_reward_t1: float
    risk_reward_t2: float
    quality_score: int
    reasons: List[str]
    suggested_lot_size: float
    timestamp: str


class AccountInfo(TypedDict):
    broker: str
    account_id: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level_pct: float
    leverage: int
    currency: str


class Position(TypedDict):
    ticket: str
    symbol: str
    side: Literal["BUY", "SELL"]
    lots: float
    open_price: float
    current_price: float
    sl: Optional[float]
    tp: Optional[float]
    pnl: float
    open_time: str
    comment: str


class OrderResult(TypedDict):
    status: Literal["SUCCESS", "REJECTED", "ERROR"]
    ticket: Optional[str]
    symbol: str
    action: str
    lots: float
    price: float
    sl: Optional[float]
    tp: Optional[float]
    message: str
    timestamp: str
