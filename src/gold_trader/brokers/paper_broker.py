"""
SQLite-backed Paper Trading Broker for testing Gold strategies with realistic execution.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from gold_trader.brokers.base import BaseBroker
from gold_trader.config import config
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.types import AccountInfo, OrderResult, Position


class PaperBroker(BaseBroker):
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.PAPER_DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id TEXT PRIMARY KEY,
                    balance REAL NOT NULL,
                    currency TEXT NOT NULL,
                    leverage INTEGER NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    ticket TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    lots REAL NOT NULL,
                    open_price REAL NOT NULL,
                    sl REAL,
                    tp REAL,
                    open_time TEXT NOT NULL,
                    comment TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    ticket TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    lots REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    comment TEXT
                )
            """)
            cursor.execute("SELECT COUNT(*) FROM account")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO account (id, balance, currency, leverage) VALUES (?, ?, ?, ?)",
                    ("PAPER_MAIN", 10000.0, "USD", 100),
                )
            conn.commit()

    def connect(self) -> bool:
        return True

    def _get_current_price(self, symbol: str) -> float:
        try:
            candles = fetch_gold_candles(symbol, period="5d", interval="1h")
            if candles:
                return float(candles[-1]["close"])
        except Exception:
            pass
        return 2650.0  # Realistic default Gold spot price if offline

    def get_account_info(self) -> AccountInfo:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT balance, currency, leverage FROM account WHERE id = 'PAPER_MAIN'")
            row = cursor.fetchone()
            balance = row[0] if row else 10000.0
            currency = row[1] if row else "USD"
            leverage = row[2] if row else 100

        open_positions = self.get_open_positions()
        floating_pnl = sum(p["pnl"] for p in open_positions)
        equity = round(balance + floating_pnl, 2)
        used_margin = sum(p["lots"] * 100 * p["open_price"] / leverage for p in open_positions)
        free_margin = max(0.0, equity - used_margin)
        margin_level = round((equity / used_margin * 100.0), 2) if used_margin > 0 else 1000.0

        return {
            "broker": "Paper Trading (Simulated)",
            "account_id": "PAPER_MAIN",
            "balance": round(balance, 2),
            "equity": equity,
            "margin": round(used_margin, 2),
            "free_margin": round(free_margin, 2),
            "margin_level_pct": margin_level,
            "leverage": leverage,
            "currency": currency,
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT ticket, symbol, side, lots, open_price, sl, tp, open_time, comment FROM positions"
            params: list = []
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol.upper())
            cursor.execute(query, params)
            rows = cursor.fetchall()

        positions: List[Position] = []
        for r in rows:
            ticket, sym, side, lots, open_p, sl, tp, open_t, comm = r
            curr_p = self._get_current_price(sym)
            pnl = (curr_p - open_p) * lots * 100.0 if side == "BUY" else (open_p - curr_p) * lots * 100.0
            positions.append({
                "ticket": ticket,
                "symbol": sym,
                "side": side,
                "lots": lots,
                "open_price": open_p,
                "current_price": round(curr_p, 2),
                "sl": sl,
                "tp": tp,
                "pnl": round(pnl, 2),
                "open_time": open_t,
                "comment": comm or "",
            })
        return positions

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
        now_str = datetime.now(timezone.utc).isoformat()
        current_price = self._get_current_price(symbol)
        if order_type.upper() == "LIMIT" and price is not None:
            fill_price = float(price)
        else:
            # Apply simulated bid-ask spread ($0.25 spread on XAUUSD)
            fill_price = current_price + 0.15 if side == "BUY" else current_price - 0.15

        ticket = f"P-{uuid.uuid4().hex[:8].upper()}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO positions (ticket, symbol, side, lots, open_price, sl, tp, open_time, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticket, symbol.upper(), side.upper(), lots, round(fill_price, 2), sl, tp, now_str, comment),
            )
            conn.commit()

        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": symbol.upper(),
            "action": side.upper(),
            "lots": lots,
            "price": round(fill_price, 2),
            "sl": sl,
            "tp": tp,
            "message": f"Simulated {side.upper()} order filled at ${fill_price:.2f}",
            "timestamp": now_str,
        }

    def modify_position(
        self,
        ticket: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE positions SET sl = ?, tp = ? WHERE ticket = ?", (sl, tp, ticket))
            if cursor.rowcount == 0:
                return {
                    "status": "ERROR",
                    "ticket": ticket,
                    "symbol": "XAUUSD",
                    "action": "MODIFY",
                    "lots": 0.0,
                    "price": 0.0,
                    "sl": sl,
                    "tp": tp,
                    "message": f"Position ticket '{ticket}' not found",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            conn.commit()

        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "MODIFY",
            "lots": 0.0,
            "price": 0.0,
            "sl": sl,
            "tp": tp,
            "message": f"Position {ticket} updated with SL={sl}, TP={tp}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def close_position(self, ticket: str) -> OrderResult:
        now_str = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, side, lots, open_price, open_time, comment FROM positions WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            if not row:
                return {
                    "status": "ERROR",
                    "ticket": ticket,
                    "symbol": "XAUUSD",
                    "action": "CLOSE",
                    "lots": 0.0,
                    "price": 0.0,
                    "sl": None,
                    "tp": None,
                    "message": f"Position '{ticket}' not found",
                    "timestamp": now_str,
                }

            sym, side, lots, open_p, open_t, comm = row
            close_p = self._get_current_price(sym)
            realized_pnl = (close_p - open_p) * lots * 100.0 if side == "BUY" else (open_p - close_p) * lots * 100.0

            # Delete from positions and insert into history
            cursor.execute("DELETE FROM positions WHERE ticket = ?", (ticket,))
            cursor.execute(
                """
                INSERT INTO trade_history (ticket, symbol, side, lots, open_price, close_price, realized_pnl, open_time, close_time, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticket, sym, side, lots, open_p, close_p, realized_pnl, open_t, now_str, comm),
            )
            # Update balance
            cursor.execute("UPDATE account SET balance = balance + ? WHERE id = 'PAPER_MAIN'", (realized_pnl,))
            conn.commit()

        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": sym,
            "action": "CLOSE",
            "lots": lots,
            "price": round(close_p, 2),
            "sl": None,
            "tp": None,
            "message": f"Closed {ticket} at ${close_p:.2f} (Realized PnL: ${realized_pnl:+.2f})",
            "timestamp": now_str,
        }
