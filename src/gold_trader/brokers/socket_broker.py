"""
Free, 100% Open-Source Native Socket Bridge Broker for XM MetaTrader 5.
Communicates directly with the GoldTraderBridge MQL5 Expert Advisor over a local TCP socket.
Zero third-party SaaS subscriptions, zero monthly fees.
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from gold_trader.brokers.base import BaseBroker
from gold_trader.config import config
from gold_trader.types import AccountInfo, OrderResult, Position


class SocketBroker(BaseBroker):
    """
    Direct TCP Socket Broker for XM MT5.
    Connects to the GoldTraderBridge.mq5 Expert Advisor running in your MT5 terminal.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[float] = None,
        account_type: Optional[str] = None,
    ):
        self.host = host or config.MT5_BRIDGE_HOST
        self.port = port or config.MT5_BRIDGE_PORT
        self.timeout = timeout or config.MT5_BRIDGE_TIMEOUT
        self.account_type = (account_type or config.XM_ACCOUNT_TYPE).lower().strip()

    def _send_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON payload to the MT5 Socket Bridge and return the parsed JSON response."""
        payload["account_type"] = self.account_type
        message = json.dumps(payload) + "\n"
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.port))
                s.sendall(message.encode("utf-8"))
                
                # Read response until newline or EOF
                buffer = []
                while True:
                    chunk = s.recv(16384)
                    if not chunk:
                        break
                    buffer.append(chunk.decode("utf-8", errors="replace"))
                    if buffer[-1].endswith("\n"):
                        break

                raw_resp = "".join(buffer).strip()
                if not raw_resp:
                    return {"status": "ERROR", "message": "Empty response from MT5 Bridge EA"}
                return json.loads(raw_resp)
        except ConnectionRefusedError:
            return {
                "status": "ERROR",
                "message": f"Could not connect to MT5 Bridge at {self.host}:{self.port}. Ensure GoldTraderBridge.mq5 EA is running and attached to a chart in XM MT5.",
            }
        except socket.timeout:
            return {
                "status": "ERROR",
                "message": f"Timeout waiting for response from MT5 Bridge at {self.host}:{self.port}",
            }
        except Exception as exc:
            return {
                "status": "ERROR",
                "message": f"Socket communication error: {exc}",
            }

    def connect(self) -> bool:
        resp = self._send_command({"command": "PING"})
        return resp.get("status") == "SUCCESS" or "balance" in resp

    def get_account_info(self) -> AccountInfo:
        badge = "LIVE ⚠️ REAL FUNDS" if self.account_type == "live" else "DEMO"
        resp = self._send_command({"command": "ACCOUNT_INFO"})
        
        if resp.get("status") == "ERROR" or "balance" not in resp:
            err_msg = resp.get("message", "Bridge Offline")
            return {
                "broker": f"XM MT5 Bridge [{badge}] ({self.host}:{self.port}) [OFFLINE - {err_msg}]",
                "account_id": "OFFLINE",
                "balance": 0.0,
                "equity": 0.0,
                "margin": 0.0,
                "free_margin": 0.0,
                "margin_level_pct": 0.0,
                "leverage": 0,
                "currency": "USD",
            }

        server_name = resp.get("server", "XMGlobal-MT5")
        trade_mode = resp.get("trade_mode", badge)
        return {
            "broker": f"XM MT5 [{trade_mode}] ({server_name})",
            "account_id": str(resp.get("login", "0")),
            "balance": round(float(resp.get("balance", 0.0)), 2),
            "equity": round(float(resp.get("equity", 0.0)), 2),
            "margin": round(float(resp.get("margin", 0.0)), 2),
            "free_margin": round(float(resp.get("free_margin", 0.0)), 2),
            "margin_level_pct": round(float(resp.get("margin_level_pct", 0.0)), 2),
            "leverage": int(resp.get("leverage", 100)),
            "currency": str(resp.get("currency", "USD")),
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        resp = self._send_command({
            "command": "GET_POSITIONS",
            "symbol": symbol.upper() if symbol else "",
        })
        
        if resp.get("status") != "SUCCESS" or "positions" not in resp:
            return []

        raw_list = resp.get("positions", [])
        positions: List[Position] = []
        for p in raw_list:
            positions.append({
                "ticket": str(p.get("ticket")),
                "symbol": str(p.get("symbol")),
                "side": "BUY" if p.get("side", "").upper() == "BUY" else "SELL",
                "lots": float(p.get("lots", 0.01)),
                "open_price": float(p.get("open_price", 0.0)),
                "current_price": float(p.get("current_price", 0.0)),
                "sl": float(p.get("sl")) if p.get("sl") else None,
                "tp": float(p.get("tp")) if p.get("tp") else None,
                "pnl": round(float(p.get("pnl", 0.0)), 2),
                "open_time": str(p.get("open_time", "")),
                "comment": str(p.get("comment", "")),
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
        price: Optional[float] = None,
        order_type: Literal["MARKET", "LIMIT"] = "MARKET",
    ) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        payload = {
            "command": "PLACE_ORDER",
            "symbol": symbol,
            "action": side.upper(),
            "lots": float(lots),
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "comment": comment,
            "order_type": order_type.upper(),
            "price": float(price) if price is not None else 0.0,
        }
        
        resp = self._send_command(payload)
        status = resp.get("status", "ERROR")
        ticket = str(resp.get("ticket")) if resp.get("ticket") else None
        filled_or_target_price = float(resp.get("price", price or 0.0))
        msg = resp.get("message", f"Order {status}")
        actual_symbol = resp.get("symbol", symbol)
        actual_sl = resp.get("sl", sl)
        actual_tp = resp.get("tp", tp)

        return {
            "status": "SUCCESS" if status == "SUCCESS" else "REJECTED",
            "ticket": ticket,
            "symbol": actual_symbol,
            "action": side,
            "lots": lots,
            "price": filled_or_target_price,
            "sl": actual_sl,
            "tp": actual_tp,
            "message": msg,
            "timestamp": now_str,
        }

    def modify_position(
        self,
        ticket: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        payload = {
            "command": "MODIFY_POSITION",
            "ticket": int(ticket),
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
        }
        resp = self._send_command(payload)
        status = resp.get("status", "ERROR")
        return {
            "status": "SUCCESS" if status == "SUCCESS" else "ERROR",
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "MODIFY",
            "lots": 0.0,
            "price": 0.0,
            "sl": sl,
            "tp": tp,
            "message": resp.get("message", f"Modify {status}"),
            "timestamp": now_str,
        }

    def close_position(self, ticket: str, lots: Optional[float] = None) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        payload = {
            "command": "CLOSE_POSITION",
            "ticket": int(ticket),
        }
        if lots is not None:
            payload["lots"] = float(lots)
        resp = self._send_command(payload)
        status = resp.get("status", "ERROR")
        price = float(resp.get("price", 0.0))
        closed_lots = float(resp.get("lots", lots or 0.0))
        return {
            "status": "SUCCESS" if status == "SUCCESS" else "ERROR",
            "ticket": ticket,
            "symbol": str(resp.get("symbol", "XAUUSD")),
            "action": "CLOSE",
            "lots": closed_lots,
            "price": price,
            "sl": None,
            "tp": None,
            "message": resp.get("message", f"Close {status}"),
            "timestamp": now_str,
        }

    def get_symbol_info(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Fetch live ticker and broker specs for symbol."""
        return self._send_command({"command": "SYMBOL_INFO", "symbol": symbol})

    def get_rates(self, symbol: str = "XAUUSD", timeframe: str = "15m", count: int = 100) -> List[Dict[str, Any]]:
        """Fetch live OHLCV candle rates directly from MT5 bridge."""
        resp = self._send_command({
            "command": "GET_RATES",
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count,
        })
        if resp.get("status") == "SUCCESS" and "candles" in resp:
            return resp["candles"]
        return []

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch active pending orders (limits, stops)."""
        resp = self._send_command({
            "command": "GET_PENDING_ORDERS",
            "symbol": symbol or "",
        })
        if resp.get("status") == "SUCCESS" and "orders" in resp:
            return resp["orders"]
        return []

    def cancel_order(self, ticket: str) -> Dict[str, Any]:
        """Cancel an active pending order."""
        try:
            ticket_int = int(ticket)
        except (ValueError, TypeError):
            return {"status": "ERROR", "message": f"Invalid ticket format: {ticket}"}
        return self._send_command({
            "command": "CANCEL_ORDER",
            "ticket": ticket_int,
        })

    def get_deal_history(self, ticket: Optional[str | int] = None, days: int = 7) -> Dict[str, Any]:
        """Fetch historical deal receipt directly from MT5 terminal."""
        payload: Dict[str, Any] = {"command": "GET_DEAL_HISTORY"}
        if ticket is not None:
            try:
                payload["ticket"] = int(ticket)
            except (ValueError, TypeError):
                payload["ticket"] = str(ticket)
        else:
            payload["days"] = days
        return self._send_command(payload)

