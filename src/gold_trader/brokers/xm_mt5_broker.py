"""
Native MetaTrader 5 Broker integration for XM Global (Live / Demo Accounts).
Operates directly on Windows or VPS hosting the XM MT5 terminal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from gold_trader.brokers.base import BaseBroker
from gold_trader.config import config
from gold_trader.types import AccountInfo, OrderResult, Position

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False


class XmMT5Broker(BaseBroker):
    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        account_type: Optional[str] = None,
    ):
        self.account_type = (account_type or config.XM_ACCOUNT_TYPE).lower().strip()
        default_login, default_pwd, default_server = config.get_xm_credentials(self.account_type)
        self.login = login or default_login
        self.password = password or default_pwd
        self.server = server or default_server
        self.path = path or config.XM_PATH
        self.connected = False

    def connect(self) -> bool:
        if not _MT5_AVAILABLE:
            return False

        init_kwargs: Dict[str, Any] = {}
        if self.path:
            init_kwargs["path"] = self.path

        if not mt5.initialize(**init_kwargs):
            return False

        if self.login and self.password:
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
            )
            if not authorized:
                mt5.shutdown()
                return False

        self.connected = True
        return True

    def _resolve_xm_symbol(self, requested_symbol: str) -> Optional[str]:
        """Resolves the broker-specific gold symbol format on XM (Standard, Micro, Ultra-Low)."""
        candidates = [
            requested_symbol,
            requested_symbol.upper(),
            "XAUUSD",
            "GOLD",
            "XAUUSD.micro",
            "GOLD.micro",
            "XAUUSD#",
            "GOLD#",
            "XAUUSD.",
            "XAUUSDraw",
        ]
        # Remove duplicates while preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        for sym in ordered:
            tick = mt5.symbol_info_tick(sym)
            if tick is not None:
                # Ensure symbol is selected in Market Watch
                mt5.symbol_select(sym, True)
                return sym
        return None

    def get_account_info(self) -> AccountInfo:
        badge = "LIVE ⚠️ REAL FUNDS" if self.account_type == "live" else "DEMO"
        if not self.connected and not self.connect():
            return {
                "broker": f"XM MT5 [{badge}] ({self.server}) [OFFLINE - MetaTrader5 package or login not active]",
                "account_id": str(self.login),
                "balance": 0.0,
                "equity": 0.0,
                "margin": 0.0,
                "free_margin": 0.0,
                "margin_level_pct": 0.0,
                "leverage": 0,
                "currency": "USD",
            }

        info = mt5.account_info()
        if info is None:
            return {
                "broker": f"XM MT5 [{badge}] ({self.server}) [UNAVAILABLE]",
                "account_id": str(self.login),
                "balance": 0.0,
                "equity": 0.0,
                "margin": 0.0,
                "free_margin": 0.0,
                "margin_level_pct": 0.0,
                "leverage": 0,
                "currency": "USD",
            }

        trade_mode = getattr(info, "trade_mode", None)
        if trade_mode == 0:
            actual_mode = "DEMO"
        elif trade_mode == 2:
            actual_mode = "LIVE ⚠️ REAL CAPITAL"
        else:
            actual_mode = badge

        return {
            "broker": f"XM MT5 [{actual_mode}] ({self.server})",
            "account_id": str(info.login),
            "balance": round(info.balance, 2),
            "equity": round(info.equity, 2),
            "margin": round(info.margin, 2),
            "free_margin": round(info.margin_free, 2),
            "margin_level_pct": round(info.margin_level, 2) if info.margin_level else 0.0,
            "leverage": int(info.leverage),
            "currency": str(info.currency),
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if not self.connected and not self.connect():
            return []

        sym = symbol.upper() if symbol else None
        raw_positions = mt5.positions_get(symbol=sym) if sym else mt5.positions_get()
        if not raw_positions:
            return []

        positions: List[Position] = []
        for p in raw_positions:
            positions.append({
                "ticket": str(p.ticket),
                "symbol": p.symbol,
                "side": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "lots": float(p.volume),
                "open_price": float(p.price_open),
                "current_price": float(p.price_current),
                "sl": float(p.sl) if p.sl else None,
                "tp": float(p.tp) if p.tp else None,
                "pnl": round(float(p.profit), 2),
                "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                "comment": p.comment or "",
            })
        return positions

    def place_order(
        self,
        symbol: str = "XAUUSD",
        side: Literal["BUY", "SELL"] = "BUY",
        lots: float = 0.01,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "Gold-Trader-MCP",
    ) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        if not self.connected and not self.connect():
            return {
                "status": "ERROR",
                "ticket": None,
                "symbol": symbol,
                "action": side,
                "lots": lots,
                "price": 0.0,
                "sl": sl,
                "tp": tp,
                "message": "MetaTrader 5 terminal not connected or MT5 library missing.",
                "timestamp": now_str,
            }

        resolved_symbol = self._resolve_xm_symbol(symbol)
        if not resolved_symbol:
            return {
                "status": "ERROR",
                "ticket": None,
                "symbol": symbol,
                "action": side,
                "lots": lots,
                "price": 0.0,
                "sl": sl,
                "tp": tp,
                "message": f"Failed to retrieve price tick for {symbol} on XM MT5 (checked XAUUSD, GOLD, .micro, # variants)",
                "timestamp": now_str,
            }

        symbol = resolved_symbol
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {
                "status": "ERROR",
                "ticket": None,
                "symbol": symbol,
                "action": side,
                "lots": lots,
                "price": 0.0,
                "sl": sl,
                "tp": tp,
                "message": f"Could not fetch tick for resolved XM symbol: {symbol}",
                "timestamp": now_str,
            }

        order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if side.upper() == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": 20,
            "magic": 987001,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = result.comment if result else f"Error code: {mt5.last_error()}"
            return {
                "status": "REJECTED",
                "ticket": None,
                "symbol": symbol,
                "action": side,
                "lots": lots,
                "price": price,
                "sl": sl,
                "tp": tp,
                "message": f"XM Order execution failed: {err_msg}",
                "timestamp": now_str,
            }

        return {
            "status": "SUCCESS",
            "ticket": str(result.order),
            "symbol": symbol,
            "action": side,
            "lots": lots,
            "price": round(result.price, 2),
            "sl": sl,
            "tp": tp,
            "message": f"XM MT5 order #{result.order} executed successfully at ${result.price:.2f}",
            "timestamp": now_str,
        }

    def modify_position(
        self,
        ticket: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        if not self.connected and not self.connect():
            return {
                "status": "ERROR",
                "ticket": ticket,
                "symbol": "XAUUSD",
                "action": "MODIFY",
                "lots": 0.0,
                "price": 0.0,
                "sl": sl,
                "tp": tp,
                "message": "XM MT5 not connected",
                "timestamp": now_str,
            }

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
        }
        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "status": "ERROR",
                "ticket": ticket,
                "symbol": "XAUUSD",
                "action": "MODIFY",
                "lots": 0.0,
                "price": 0.0,
                "sl": sl,
                "tp": tp,
                "message": f"Failed to modify position #{ticket}: {res.comment if res else mt5.last_error()}",
                "timestamp": now_str,
            }

        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "MODIFY",
            "lots": 0.0,
            "price": 0.0,
            "sl": sl,
            "tp": tp,
            "message": f"Position #{ticket} modified successfully (SL={sl}, TP={tp})",
            "timestamp": now_str,
        }

    def close_position(self, ticket: str) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        if not self.connected and not self.connect():
            return {
                "status": "ERROR",
                "ticket": ticket,
                "symbol": "XAUUSD",
                "action": "CLOSE",
                "lots": 0.0,
                "price": 0.0,
                "sl": None,
                "tp": None,
                "message": "XM MT5 not connected",
                "timestamp": now_str,
            }

        positions = mt5.positions_get(ticket=int(ticket))
        if not positions:
            return {
                "status": "ERROR",
                "ticket": ticket,
                "symbol": "XAUUSD",
                "action": "CLOSE",
                "lots": 0.0,
                "price": 0.0,
                "sl": None,
                "tp": None,
                "message": f"Position #{ticket} not found on XM",
                "timestamp": now_str,
            }

        p = positions[0]
        tick = mt5.symbol_info_tick(p.symbol)
        close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": p.ticket,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "price": close_price,
            "deviation": 20,
            "magic": 987001,
            "comment": "Close by Gold-Trader-MCP",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "status": "ERROR",
                "ticket": ticket,
                "symbol": p.symbol,
                "action": "CLOSE",
                "lots": float(p.volume),
                "price": close_price,
                "sl": None,
                "tp": None,
                "message": f"Failed to close position #{ticket}: {res.comment if res else mt5.last_error()}",
                "timestamp": now_str,
            }

        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": p.symbol,
            "action": "CLOSE",
            "lots": float(p.volume),
            "price": round(res.price, 2),
            "sl": None,
            "tp": None,
            "message": f"Position #{ticket} closed at ${res.price:.2f}",
            "timestamp": now_str,
        }
