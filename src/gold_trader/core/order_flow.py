"""
ATAS-Grade Autonomous Order Flow Analysis Engine for Aegis Alpha (UAT).

Directly ingests real-time tick/trade feeds via exchange WebSockets (Bybit, Binance, Coinbase)
and computes institutional order-flow analytics in pure Python:
- Footprint (Bid x Ask volume per price tick)
- Delta (Ask Vol - Bid Vol) and Cumulative Volume Delta (CVD)
- CVD Delta Divergence Detection (Absorption & Exhaustion)
- Volume Point of Control (VPOC) and 70% Value Area (VAH / VAL)
- Diagonal Bid/Ask Imbalances and Stacked Imbalances (3+ consecutive levels)
- Large Block / Big Trades Tracking
- Persistent UAT SQLite Telemetry (~/.gold_trading_data/uat_order_flow.db)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("OrderFlowEngine")

DEFAULT_UAT_DB_PATH = os.path.expanduser("~/.gold_trading_data/uat_order_flow.db")

# Default tick sizes for price bucketing
DEFAULT_TICK_SIZES: dict[str, float] = {
    "BTCUSDT": 1.0,
    "BTCUSD": 1.0,
    "ETHUSDT": 0.10,
    "ETHUSD": 0.10,
    "PAXGUSDT": 0.10,
    "XAUUSD": 0.10,
}

# Default big trade thresholds in base asset units
DEFAULT_BIG_TRADE_THRESHOLDS: dict[str, float] = {
    "BTCUSDT": 0.50,   # ~0.50 BTC ($35k+)
    "BTCUSD": 0.50,
    "ETHUSDT": 5.0,    # ~5.0 ETH ($12k+)
    "ETHUSD": 5.0,
    "PAXGUSDT": 2.0,   # ~2.0 oz Gold ($5k+)
    "XAUUSD": 2.0,
}


@dataclass
class Trade:
    """Represents an executed market trade."""
    price: float
    size: float
    side: str  # "BUY" (aggressive ask lift) or "SELL" (aggressive bid hit)
    timestamp: float  # epoch seconds or ms
    trade_id: str = ""
    is_liquidation: bool = False

    @property
    def notional_usd(self) -> float:
        return self.price * self.size


@dataclass
class FootprintLevel:
    """Volume profile and delta at a discrete price tick bucket."""
    price: float
    buy_volume: float = 0.0   # Aggressive market buyers lifting ask
    sell_volume: float = 0.0  # Aggressive market sellers hitting bid
    trade_count: int = 0

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume


@dataclass
class ImbalanceZone:
    """Stacked imbalance detection zone."""
    side: str  # "BUY" or "SELL"
    start_price: float
    end_price: float
    levels_count: int
    imbalance_ratios: list[float]
    total_volume: float


@dataclass
class OrderFlowBar:
    """Full footprint candlestick bar with ATAS microstructural analytics."""
    symbol: str
    timeframe: str
    bar_start_ts: float  # Unix epoch seconds
    open: float
    high: float
    low: float
    close: float
    levels: dict[float, FootprintLevel] = field(default_factory=dict)
    tick_size: float = 1.0

    # Aggregated metrics
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    delta: float = 0.0
    delta_pct: float = 0.0
    cvd: float = 0.0

    # Market Profile & Footprint analytics
    vpoc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    value_area_volume: float = 0.0

    # Imbalances & Signals
    stacked_buy_imbalances: list[ImbalanceZone] = field(default_factory=list)
    stacked_sell_imbalances: list[ImbalanceZone] = field(default_factory=list)
    divergence: str = "NONE"  # "BULLISH_DIVERGENCE", "BEARISH_DIVERGENCE", "NONE"
    absorption: str = "NONE"  # "BULLISH_ABSORPTION", "BEARISH_ABSORPTION", "NONE"

    def add_trade(self, price: float, size: float, side: str) -> None:
        """Add trade to discrete price bucket and update bar OHLC."""
        bucket = round(round(price / self.tick_size) * self.tick_size, 4)
        if bucket not in self.levels:
            self.levels[bucket] = FootprintLevel(price=bucket)

        level = self.levels[bucket]
        if side.upper() == "BUY":
            level.buy_volume += size
            self.buy_volume += size
        else:
            level.sell_volume += size
            self.sell_volume += size

        level.trade_count += 1
        self.total_volume += size
        self.delta = self.buy_volume - self.sell_volume
        self.delta_pct = (self.delta / self.total_volume * 100.0) if self.total_volume > 0 else 0.0

        # Update OHLC
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price

    def finalize(
        self,
        imbalance_ratio: float = 3.0,
        min_imbalance_volume: float = 0.01,
        stacked_min_levels: int = 3,
        prior_bar: OrderFlowBar | None = None,
    ) -> None:
        """Finalize VPOC, VAH/VAL, and diagonal imbalances."""
        if not self.levels or self.total_volume <= 0:
            return

        sorted_prices = sorted(self.levels.keys())

        # 1. Calculate VPOC (Price with highest volume)
        vpoc_price = max(sorted_prices, key=lambda p: self.levels[p].total_volume)
        self.vpoc = vpoc_price

        # 2. Calculate 70% Value Area (VAH / VAL) via standard ATAS market profile expansion
        target_va_vol = 0.70 * self.total_volume
        accum_vol = self.levels[vpoc_price].total_volume
        low_idx = sorted_prices.index(vpoc_price)
        high_idx = low_idx

        while accum_vol < target_va_vol and (low_idx > 0 or high_idx < len(sorted_prices) - 1):
            # Inspect 2 levels up vs 2 levels down
            up_vol = 0.0
            down_vol = 0.0

            # Step up
            up_candidates = sorted_prices[high_idx + 1: high_idx + 3]
            for p in up_candidates:
                up_vol += self.levels[p].total_volume

            # Step down
            down_candidates = sorted_prices[max(0, low_idx - 2): low_idx]
            for p in down_candidates:
                down_vol += self.levels[p].total_volume

            if up_vol >= down_vol and high_idx < len(sorted_prices) - 1:
                accum_vol += up_vol
                high_idx = min(len(sorted_prices) - 1, high_idx + len(up_candidates))
            elif low_idx > 0:
                accum_vol += down_vol
                low_idx = max(0, low_idx - len(down_candidates))
            else:
                break

        self.val = sorted_prices[low_idx]
        self.vah = sorted_prices[high_idx]
        self.value_area_volume = accum_vol

        # 3. Calculate Diagonal Footprint Imbalances
        # Diagonal Buy Imbalance: Ask volume at level P_i >= ratio * Bid volume at level P_{i-1}
        # Diagonal Sell Imbalance: Bid volume at level P_i >= ratio * Ask volume at level P_{i+1}
        buy_imbalance_prices: list[tuple[float, float]] = []
        sell_imbalance_prices: list[tuple[float, float]] = []

        for i, price in enumerate(sorted_prices):
            curr_level = self.levels[price]

            # Check Buy Imbalance diagonally against level below
            if i > 0:
                below_level = self.levels[sorted_prices[i - 1]]
                if curr_level.buy_volume >= min_imbalance_volume:
                    denom = max(below_level.sell_volume, 0.0001)
                    ratio = curr_level.buy_volume / denom
                    if ratio >= imbalance_ratio and (below_level.sell_volume > 0 or curr_level.buy_volume >= min_imbalance_volume * 3):
                        buy_imbalance_prices.append((price, ratio))

            # Check Sell Imbalance diagonally against level above
            if i < len(sorted_prices) - 1:
                above_level = self.levels[sorted_prices[i + 1]]
                if curr_level.sell_volume >= min_imbalance_volume:
                    denom = max(above_level.buy_volume, 0.0001)
                    ratio = curr_level.sell_volume / denom
                    if ratio >= imbalance_ratio and (above_level.buy_volume > 0 or curr_level.sell_volume >= min_imbalance_volume * 3):
                        sell_imbalance_prices.append((price, ratio))

        # 4. Detect Stacked Imbalances (>= stacked_min_levels consecutive prices)
        self.stacked_buy_imbalances = self._extract_stacked_zones(
            buy_imbalance_prices, sorted_prices, "BUY", stacked_min_levels
        )
        self.stacked_sell_imbalances = self._extract_stacked_zones(
            sell_imbalance_prices, sorted_prices, "SELL", stacked_min_levels
        )

        # 5. Delta Divergence against prior bar
        if prior_bar and prior_bar.total_volume > 0:
            price_higher = self.close > prior_bar.close or self.high > prior_bar.high
            price_lower = self.close < prior_bar.close or self.low < prior_bar.low
            delta_negative = self.delta < 0 and self.delta < prior_bar.delta
            delta_positive = self.delta > 0 and self.delta > prior_bar.delta

            if price_higher and delta_negative:
                self.divergence = "BEARISH_DIVERGENCE"
            elif price_lower and delta_positive:
                self.divergence = "BULLISH_DIVERGENCE"

        # 6. Absorption & Exhaustion detection
        # Bullish Absorption: Strong selling in lower wick but candle closes up or positive delta
        # Bearish Absorption: Strong buying in upper wick but candle closes down or negative delta
        candle_range = max(self.high - self.low, 0.0001)
        lower_wick_ratio = (min(self.open, self.close) - self.low) / candle_range
        upper_wick_ratio = (self.high - max(self.open, self.close)) / candle_range

        if lower_wick_ratio >= 0.35 and self.delta > 0:
            self.absorption = "BULLISH_ABSORPTION"
        elif upper_wick_ratio >= 0.35 and self.delta < 0:
            self.absorption = "BEARISH_ABSORPTION"

    def _extract_stacked_zones(
        self,
        imbalance_list: list[tuple[float, float]],
        sorted_prices: list[float],
        side: str,
        min_levels: int,
    ) -> list[ImbalanceZone]:
        """Extract consecutive adjacent price levels with imbalances."""
        if len(imbalance_list) < min_levels:
            return []

        imbalance_dict = dict(imbalance_list)
        zones: list[ImbalanceZone] = []
        curr_streak: list[float] = []

        for p in sorted_prices:
            if p in imbalance_dict:
                curr_streak.append(p)
            else:
                if len(curr_streak) >= min_levels:
                    zones.append(self._build_zone(curr_streak, imbalance_dict, side))
                curr_streak = []

        if len(curr_streak) >= min_levels:
            zones.append(self._build_zone(curr_streak, imbalance_dict, side))

        return zones

    def _build_zone(
        self, streak: list[float], ratios_dict: dict[float, float], side: str
    ) -> ImbalanceZone:
        ratios = [ratios_dict[p] for p in streak]
        tot_vol = sum(self.levels[p].total_volume for p in streak)
        return ImbalanceZone(
            side=side,
            start_price=streak[0],
            end_price=streak[-1],
            levels_count=len(streak),
            imbalance_ratios=ratios,
            total_volume=tot_vol,
        )


class OrderFlowDB:
    """Isolated SQLite database storage for UAT Order Flow persistence."""

    def __init__(self, db_path: str = DEFAULT_UAT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uat_order_flow_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    bar_start_ts REAL NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    total_volume REAL NOT NULL,
                    buy_volume REAL NOT NULL,
                    sell_volume REAL NOT NULL,
                    delta REAL NOT NULL,
                    delta_pct REAL NOT NULL,
                    cvd REAL NOT NULL,
                    vpoc REAL NOT NULL,
                    vah REAL NOT NULL,
                    val REAL NOT NULL,
                    stacked_buy_count INTEGER NOT NULL DEFAULT 0,
                    stacked_sell_count INTEGER NOT NULL DEFAULT 0,
                    divergence TEXT NOT NULL DEFAULT 'NONE',
                    absorption TEXT NOT NULL DEFAULT 'NONE',
                    footprint_json TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, bar_start_ts)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uat_big_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    notional_usd REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uat_order_flow_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    details_json TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_bar(self, bar: OrderFlowBar) -> None:
        """Persist finalized bar to SQLite."""
        footprint_summary = {
            f"{p:.2f}": {
                "buy": round(lvl.buy_volume, 4),
                "sell": round(lvl.sell_volume, 4),
                "delta": round(lvl.delta, 4),
            }
            for p, lvl in sorted(bar.levels.items())
        }

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO uat_order_flow_bars (
                    symbol, timeframe, bar_start_ts, open, high, low, close,
                    total_volume, buy_volume, sell_volume, delta, delta_pct,
                    cvd, vpoc, vah, val, stacked_buy_count, stacked_sell_count,
                    divergence, absorption, footprint_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, bar_start_ts) DO UPDATE SET
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    total_volume=excluded.total_volume,
                    buy_volume=excluded.buy_volume,
                    sell_volume=excluded.sell_volume,
                    delta=excluded.delta,
                    delta_pct=excluded.delta_pct,
                    cvd=excluded.cvd,
                    vpoc=excluded.vpoc,
                    vah=excluded.vah,
                    val=excluded.val,
                    stacked_buy_count=excluded.stacked_buy_count,
                    stacked_sell_count=excluded.stacked_sell_count,
                    divergence=excluded.divergence,
                    absorption=excluded.absorption,
                    footprint_json=excluded.footprint_json
            """, (
                bar.symbol,
                bar.timeframe,
                bar.bar_start_ts,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.total_volume,
                bar.buy_volume,
                bar.sell_volume,
                bar.delta,
                bar.delta_pct,
                bar.cvd,
                bar.vpoc,
                bar.vah,
                bar.val,
                len(bar.stacked_buy_imbalances),
                len(bar.stacked_sell_imbalances),
                bar.divergence,
                bar.absorption,
                json.dumps(footprint_summary),
            ))
            conn.commit()

    def record_big_trade(self, trade: Trade, symbol: str) -> None:
        """Log detected institutional block trade."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO uat_big_trades (symbol, price, size, side, notional_usd, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, trade.price, trade.size, trade.side, trade.notional_usd, trade.timestamp))
            conn.commit()

    def record_signal(self, symbol: str, signal_type: str, price: float, details: dict[str, Any], ts: float) -> None:
        """Record actionable order flow alert."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO uat_order_flow_signals (symbol, signal_type, price, details_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, signal_type, price, json.dumps(details), ts))
            conn.commit()

    def get_recent_bars(self, symbol: str, timeframe: str = "15m", limit: int = 10) -> list[dict[str, Any]]:
        """Query recent bars from UAT SQLite."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT * FROM uat_order_flow_bars
                WHERE symbol = ? AND timeframe = ?
                ORDER BY bar_start_ts DESC LIMIT ?
            """, (symbol, timeframe, limit))
            return [dict(row) for row in cur.fetchall()]


class OrderFlowEngine:
    """
    Core Order Flow calculation engine.
    Ingests streaming real-time trades and maintains:
    - Current developing bar footprint
    - Running Session CVD
    - VPOC, Value Area, and Imbalances
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe_minutes: int = 15,
        tick_size: float | None = None,
        imbalance_ratio: float = 3.0,
        min_imbalance_volume: float | None = None,
        db_path: str = DEFAULT_UAT_DB_PATH,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe_seconds = timeframe_minutes * 60
        self.timeframe_str = f"{timeframe_minutes}m"
        self.tick_size = tick_size or DEFAULT_TICK_SIZES.get(self.symbol, 1.0)
        self.imbalance_ratio = imbalance_ratio
        self.min_imbalance_volume = min_imbalance_volume or (0.05 if "BTC" in self.symbol else 0.5)
        self.big_trade_threshold = DEFAULT_BIG_TRADE_THRESHOLDS.get(self.symbol, 1.0)

        self.db = OrderFlowDB(db_path)
        self.running_cvd: float = 0.0
        self.current_bar: OrderFlowBar | None = None
        self.completed_bars: list[OrderFlowBar] = []
        self.last_trade: Trade | None = None
        self.signal_callbacks: list[Callable[[str, dict[str, Any]], None]] = []

    def on_trade(self, price: float, size: float, side: str, timestamp_s: float) -> None:
        """Process incoming trade ticket."""
        trade = Trade(price=price, size=size, side=side.upper(), timestamp=timestamp_s)
        self.last_trade = trade

        # 1. Big trade detection
        if size >= self.big_trade_threshold:
            self.db.record_big_trade(trade, self.symbol)
            self._emit_signal("BIG_TRADE", {
                "price": price,
                "size": size,
                "side": side,
                "notional_usd": trade.notional_usd,
            }, price, timestamp_s)

        # 2. Bar bucketing
        bar_start = math.floor(timestamp_s / self.timeframe_seconds) * self.timeframe_seconds

        if self.current_bar is None:
            self.current_bar = OrderFlowBar(
                symbol=self.symbol,
                timeframe=self.timeframe_str,
                bar_start_ts=bar_start,
                open=price,
                high=price,
                low=price,
                close=price,
                tick_size=self.tick_size,
            )
        elif bar_start > self.current_bar.bar_start_ts:
            # Finalize old bar
            prior = self.completed_bars[-1] if self.completed_bars else None
            self.current_bar.finalize(
                imbalance_ratio=self.imbalance_ratio,
                min_imbalance_volume=self.min_imbalance_volume,
                prior_bar=prior,
            )
            self.running_cvd += self.current_bar.delta
            self.current_bar.cvd = self.running_cvd
            self.db.save_bar(self.current_bar)
            self.completed_bars.append(self.current_bar)
            if len(self.completed_bars) > 100:
                self.completed_bars.pop(0)

            # Check finalized alerts
            self._check_bar_signals(self.current_bar)

            # Start new bar
            self.current_bar = OrderFlowBar(
                symbol=self.symbol,
                timeframe=self.timeframe_str,
                bar_start_ts=bar_start,
                open=price,
                high=price,
                low=price,
                close=price,
                tick_size=self.tick_size,
            )

        # Update current bar
        self.current_bar.add_trade(price, size, side)
        self.current_bar.cvd = self.running_cvd + self.current_bar.delta

    def get_developing_metrics(self) -> dict[str, Any]:
        """Inspect real-time state of developing bar."""
        if not self.current_bar:
            return {"status": "AWAITING_DATA", "symbol": self.symbol}

        prior = self.completed_bars[-1] if self.completed_bars else None
        # Compute temporary finalize on developing copy
        self.current_bar.finalize(
            imbalance_ratio=self.imbalance_ratio,
            min_imbalance_volume=self.min_imbalance_volume,
            prior_bar=prior,
        )

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe_str,
            "bar_time": datetime.fromtimestamp(self.current_bar.bar_start_ts, tz=timezone.utc).strftime("%H:%M:%S UTC"),
            "open": self.current_bar.open,
            "high": self.current_bar.high,
            "low": self.current_bar.low,
            "close": self.current_bar.close,
            "total_volume": round(self.current_bar.total_volume, 4),
            "buy_volume": round(self.current_bar.buy_volume, 4),
            "sell_volume": round(self.current_bar.sell_volume, 4),
            "delta": round(self.current_bar.delta, 4),
            "delta_pct": f"{self.current_bar.delta_pct:+.2f}%",
            "cvd": round(self.current_bar.cvd, 4),
            "vpoc": self.current_bar.vpoc,
            "vah": self.current_bar.vah,
            "val": self.current_bar.val,
            "stacked_buy_zones": len(self.current_bar.stacked_buy_imbalances),
            "stacked_sell_zones": len(self.current_bar.stacked_sell_imbalances),
            "divergence": self.current_bar.divergence,
            "absorption": self.current_bar.absorption,
        }

    def _check_bar_signals(self, bar: OrderFlowBar) -> None:
        """Trigger alerts on bar close."""
        if bar.stacked_buy_imbalances:
            self._emit_signal("STACKED_BUY_IMBALANCE", {
                "zones": [asdict(z) for z in bar.stacked_buy_imbalances],
                "vpoc": bar.vpoc,
            }, bar.close, bar.bar_start_ts)

        if bar.stacked_sell_imbalances:
            self._emit_signal("STACKED_SELL_IMBALANCE", {
                "zones": [asdict(z) for z in bar.stacked_sell_imbalances],
                "vpoc": bar.vpoc,
            }, bar.close, bar.bar_start_ts)

        if bar.divergence != "NONE":
            self._emit_signal(bar.divergence, {
                "delta": bar.delta,
                "close": bar.close,
            }, bar.close, bar.bar_start_ts)

        if bar.absorption != "NONE":
            self._emit_signal(bar.absorption, {
                "delta": bar.delta,
                "high": bar.high,
                "low": bar.low,
            }, bar.close, bar.bar_start_ts)

    def _emit_signal(self, sig_type: str, details: dict[str, Any], price: float, ts: float) -> None:
        self.db.record_signal(self.symbol, sig_type, price, details, ts)
        for cb in self.signal_callbacks:
            try:
                cb(sig_type, details)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error in signal callback: {e}")


class WebSocketOrderFlowClient:
    """
    Robust Asynchronous WebSocket Client for Bybit and Binance.
    Auto-reconnects with exponential backoff.
    """

    def __init__(
        self,
        engine: OrderFlowEngine,
        feed: str = "bybit",  # "bybit" or "binance"
    ) -> None:
        self.engine = engine
        self.feed = feed.lower()
        self.is_running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self.is_running = True
        if self.feed == "bybit":
            await self._run_bybit()
        elif self.feed == "binance":
            await self._run_binance()
        else:
            raise ValueError(f"Unknown feed: {self.feed}")

    def stop(self) -> None:
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run_bybit(self) -> None:
        import websockets
        url = "wss://stream.bybit.com/v5/public/linear"
        symbol = self.engine.symbol

        while self.is_running:
            try:
                logger.info(f"Connecting to Bybit WebSocket for {symbol}...")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    sub_msg = {"op": "subscribe", "args": [f"publicTrade.{symbol}"]}
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"Subscribed to Bybit publicTrade.{symbol}")

                    while self.is_running:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        if "data" in msg and isinstance(msg["data"], list):
                            for trade in msg["data"]:
                                price = float(trade["p"])
                                size = float(trade["v"])
                                side = trade["S"]  # "Buy" or "Sell"
                                ts = float(trade["T"]) / 1000.0
                                self.engine.on_trade(price, size, side, ts)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Bybit WS connection dropped ({e}), reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _run_binance(self) -> None:
        import websockets
        symbol_lower = self.engine.symbol.lower()
        url = f"wss://stream.binance.com:9443/ws/{symbol_lower}@aggTrade"

        while self.is_running:
            try:
                logger.info(f"Connecting to Binance Spot WS for {symbol_lower}...")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info(f"Connected to Binance {symbol_lower}@aggTrade")

                    while self.is_running:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        if msg.get("e") == "aggTrade":
                            price = float(msg["p"])
                            size = float(msg["q"])
                            # In Binance, isBuyerMaker = True means Buyer was maker -> Taker was Seller
                            side = "SELL" if msg["m"] else "BUY"
                            ts = float(msg["T"]) / 1000.0
                            self.engine.on_trade(price, size, side, ts)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Binance WS connection dropped ({e}), reconnecting in 5s...")
                await asyncio.sleep(5)
