"""
Abstract Base Broker interface for Gold Execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from gold_trader.types import AccountInfo, OrderResult, Position


class BaseBroker(ABC):
    """Abstract broker interface ensuring interchangeable execution (Paper, XM MT5, MetaAPI)."""

    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Fetch balance, equity, margin, leverage."""
        pass

    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Fetch list of open positions."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        lots: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "Gold-Trader-MCP",
        price: Optional[float] = None,
        order_type: Literal["MARKET", "LIMIT"] = "MARKET",
    ) -> OrderResult:
        """Execute a market or limit order with optional SL/TP."""
        pass

    @abstractmethod
    def modify_position(
        self,
        ticket: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP of an existing open position."""
        pass

    @abstractmethod
    def close_position(self, ticket: str, lots: Optional[float] = None) -> OrderResult:
        """Close an open position (or partial volume) by ticket."""
        pass

    def get_symbol_info(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Fetch current live symbol specification and tick from broker."""
        return {}

    def get_rates(self, symbol: str = "XAUUSD", timeframe: str = "15m", count: int = 100) -> List[Dict[str, Any]]:
        """Fetch live OHLCV candle rates directly from broker."""
        return []

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch active pending orders (limits, stops)."""
        return []

    def cancel_order(self, ticket: str) -> Dict[str, Any]:
        """Cancel an active pending order."""
        return {"status": "ERROR", "message": "Not implemented"}

    def get_deal_history(self, ticket: Optional[str | int] = None, days: int = 7) -> Dict[str, Any]:
        """Fetch historical deals/receipts from broker."""
        return {"status": "ERROR", "message": "Not implemented"}

