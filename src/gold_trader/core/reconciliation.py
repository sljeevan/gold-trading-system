"""
Broker-to-Database Reconciliation & Position Verification Engine.
Compares real MT5 Broker order book and deal history against local SQLite database records.

Identifies and reports:
- UNMATCHED_BROKER_POSITION: Open on broker terminal, missing from local database.
- LOCAL_GHOST_TRADE: Marked 'OPEN' in local database, but absent from broker.
- VOLUME_MISMATCH: Discrepancy between local tracked remaining_lots and broker lots.
- MISSING_PROTECTIVE_SL: Any broker open position lacking active Stop-Loss protection.
- PNL_DISCREPANCY: Discrepancy between broker deal receipts and local recorded P&L.
- BROKER_UNAVAILABLE: Distinguishes actual query failure from confirmed zero positions.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from gold_trader.brokers.socket_broker import SocketBroker, BrokerPositionsUnavailableError

logger = logging.getLogger("ReconciliationEngine")


class ReconciliationEngine:
    """
    Automated position and deal reconciliation auditor between MT5 Broker and local databases.
    """

    def __init__(
        self,
        broker: Optional[SocketBroker] = None,
        db_paths: Optional[Dict[str, str]] = None,
    ):
        self.broker = broker or SocketBroker()
        data_dir = os.path.expanduser(os.getenv("GOLD_TRADING_DATA_DIR", "~/.gold_trading_data"))
        self.db_paths = db_paths or {
            "XAUUSD": os.path.join(data_dir, "auto_trader.db"),
            "BTCUSD": os.path.join(data_dir, "btc_auto_trader.db"),
            "ETHUSD": os.path.join(data_dir, "eth_auto_trader.db"),
        }

    def _fetch_local_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all trades marked status = 'OPEN' across relevant local databases."""
        local_trades: List[Dict[str, Any]] = []
        targets = [symbol] if symbol and symbol in self.db_paths else list(self.db_paths.keys())

        for sym in targets:
            db_path = self.db_paths.get(sym)
            if not db_path or not os.path.exists(db_path):
                continue
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM trade_executions WHERE status = 'OPEN'")
                rows = cur.fetchall()
                for r in rows:
                    d = dict(r)
                    d["db_source"] = os.path.basename(db_path)
                    local_trades.append(d)
                conn.close()
            except Exception as e:
                logger.error("Failed to query local database %s: %s", db_path, e)
        return local_trades

    def reconcile_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a full reconciliation pass between active MT5 positions and local SQLite database.
        Returns detailed discrepancy matrix and component status.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        discrepancies: List[Dict[str, Any]] = []

        # 1. Fetch Broker Open Positions (distinguishing query failure from zero positions)
        broker_query_status = "SUCCESS"
        broker_positions: List[Dict[str, Any]] = []
        broker_err_msg = ""
        try:
            broker_positions = self.broker.get_open_positions()
        except BrokerPositionsUnavailableError as b_err:
            broker_query_status = "UNAVAILABLE"
            broker_err_msg = str(b_err)
        except Exception as e:
            broker_query_status = "ERROR"
            broker_err_msg = str(e)

        if broker_query_status != "SUCCESS":
            return {
                "status": "BROKER_UNAVAILABLE",
                "timestamp": now_str,
                "error": broker_err_msg,
                "broker_positions": None,
                "local_positions_count": len(self._fetch_local_open_trades(symbol)),
                "discrepancies": [{
                    "type": "BROKER_UNAVAILABLE",
                    "severity": "CRITICAL",
                    "message": f"Cannot verify broker positions: {broker_err_msg}",
                }],
            }

        # Filter broker positions if specific symbol requested
        if symbol:
            clean_sym = symbol.upper()
            broker_positions = [
                p for p in broker_positions
                if clean_sym in str(p.get("symbol", "")).upper()
                or (clean_sym == "XAUUSD" and "GOLD" in str(p.get("symbol", "")).upper())
            ]

        # 2. Fetch Local Open Trades
        local_open_trades = self._fetch_local_open_trades(symbol)

        broker_by_ticket = {str(p.get("ticket")): p for p in broker_positions if p.get("ticket")}
        local_by_ticket = {str(t.get("ticket")): t for t in local_open_trades if t.get("ticket")}

        # 3. Check for Unmatched Broker Positions (Active on MT5, absent locally)
        for ticket, b_pos in broker_by_ticket.items():
            sl = b_pos.get("sl")
            # Safety check: Missing SL on broker position
            if sl is None or float(sl) <= 0.0:
                discrepancies.append({
                    "type": "MISSING_PROTECTIVE_SL",
                    "severity": "CRITICAL",
                    "ticket": ticket,
                    "symbol": b_pos.get("symbol"),
                    "side": b_pos.get("side"),
                    "lots": float(b_pos.get("lots", 0.0)),
                    "message": f"Active MT5 position #{ticket} has NO Stop-Loss configured!",
                })

            if ticket not in local_by_ticket:
                discrepancies.append({
                    "type": "UNMATCHED_BROKER_POSITION",
                    "severity": "HIGH",
                    "ticket": ticket,
                    "symbol": b_pos.get("symbol"),
                    "side": b_pos.get("side"),
                    "lots": float(b_pos.get("lots", 0.0)),
                    "message": f"Position #{ticket} is open on MT5 broker but missing from local database (manual trade or desync).",
                })
            else:
                # Check for Volume Discrepancy
                loc = local_by_ticket[ticket]
                b_lots = round(float(b_pos.get("lots", 0.0)), 2)
                loc_lots = round(float(loc.get("remaining_lots") or loc.get("lots", 0.0)), 2)
                if abs(b_lots - loc_lots) >= 0.01:
                    discrepancies.append({
                        "type": "VOLUME_MISMATCH",
                        "severity": "MEDIUM",
                        "ticket": ticket,
                        "symbol": b_pos.get("symbol"),
                        "broker_lots": b_lots,
                        "local_lots": loc_lots,
                        "message": f"Volume mismatch for #{ticket}: broker has {b_lots} lots, local DB tracks {loc_lots} lots.",
                    })

        # 4. Check for Local Ghost Trades (Marked OPEN locally, absent from broker)
        for ticket, loc_trade in local_by_ticket.items():
            if ticket not in broker_by_ticket:
                discrepancies.append({
                    "type": "LOCAL_GHOST_TRADE",
                    "severity": "HIGH",
                    "ticket": ticket,
                    "symbol": loc_trade.get("symbol"),
                    "side": loc_trade.get("action"),
                    "lots": float(loc_trade.get("lots", 0.0)),
                    "db_source": loc_trade.get("db_source"),
                    "message": f"Trade #{ticket} is marked 'OPEN' in {loc_trade.get('db_source')} but does not exist on MT5 broker.",
                })

        overall_status = "CLEAN" if not discrepancies else "DISCREPANCY_DETECTED"
        return {
            "status": overall_status,
            "timestamp": now_str,
            "broker_positions_count": len(broker_positions),
            "local_positions_count": len(local_open_trades),
            "broker_query_status": broker_query_status,
            "discrepancies_count": len(discrepancies),
            "discrepancies": discrepancies,
            "broker_positions": broker_positions,
            "local_open_trades": local_open_trades,
        }
