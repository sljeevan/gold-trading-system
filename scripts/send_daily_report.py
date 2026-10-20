#!/usr/bin/env python3
"""
Multi-Asset Daily Trading Session Summary Reporter.
Compiles daily realized PnL, active Flat Risk execution architecture,
multi-asset telemetry for Gold (XAUUSD), Bitcoin (BTCUSD), and Ethereum (ETHUSD),
open exposure, and market context, then dispatches directly to Telegram.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

# Ensure project root and src/ are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.compounding_ladder import CompoundingLadderManager
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("DailySessionReporter")

ASSETS_METADATA = [
    {
        "symbol": "XAUUSD",
        "broker_sym": "GOLD",
        "name": "Spot Gold",
        "icon": "🟡",
        "state_file": os.path.expanduser("~/.gold_trading_data/compounding_ladder_state.json"),
        "db_path": os.path.expanduser("~/.gold_trading_data/auto_trader.db"),
    },
    {
        "symbol": "BTCUSD",
        "broker_sym": "BTCUSD",
        "name": "Bitcoin",
        "icon": "🟠",
        "state_file": os.path.expanduser("~/.gold_trading_data/btc_compounding_ladder_state.json"),
        "db_path": os.path.expanduser("~/.gold_trading_data/btc_auto_trader.db"),
    },
    {
        "symbol": "ETHUSD",
        "broker_sym": "ETHUSD",
        "name": "Ethereum",
        "icon": "🔷",
        "state_file": os.path.expanduser("~/.gold_trading_data/eth_compounding_ladder_state.json"),
        "db_path": os.path.expanduser("~/.gold_trading_data/eth_auto_trader.db"),
    },
]


def compile_and_send_report(
    target_date: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    date_str = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info(f"📊 Compiling multi-asset daily trading report for {date_str} (Dry-Run: {dry_run})...")

    # 1. Connect to MT5 Bridge for Live Account Info & Open Positions
    broker = SocketBroker()
    broker_connected = broker.connect()
    account_info: Dict[str, Any] = {}
    open_positions: List[Dict[str, Any]] = []

    if broker_connected:
        try:
            account_info = broker.get_account_info()
            open_positions = broker.get_open_positions()
        except Exception as e:
            logger.warning(f"Error querying MT5 account/positions: {e}")
    else:
        logger.warning("MT5 Bridge not connected; using cached/offline fallback info.")
        account_info = {
            "broker": "XM MT5 [OFFLINE/CACHED]",
            "balance": 1000.0,
            "equity": 1000.0,
            "free_margin": 1000.0,
        }

    # 2. Gather Multi-Asset Telemetry
    multi_asset_data: Dict[str, Dict[str, Any]] = {}
    all_today_trades: List[Dict[str, Any]] = []

    for meta in ASSETS_METADATA:
        sym = meta["symbol"]
        bsym = meta["broker_sym"]
        st_file = meta["state_file"]
        db_file = meta["db_path"]

        # Read ladder / risk state
        state: Dict[str, Any] = {}
        if os.path.exists(st_file):
            try:
                with open(st_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception as e:
                logger.error(f"Error reading state file {st_file}: {e}")

        # Query closed trades for target_date with guaranteed connection close
        today_trades: List[Dict[str, Any]] = []
        if os.path.exists(db_file):
            try:
                with closing(sqlite3.connect(db_file)) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    rows = cur.execute(
                        """
                        SELECT ticket, symbol, action, lots, entry_price, close_price, pnl, close_time
                        FROM trade_executions
                        WHERE (close_time LIKE ? OR timestamp LIKE ?) AND status = 'CLOSED'
                        ORDER BY id ASC
                        """,
                        (f"{date_str}%", f"{date_str}%"),
                    ).fetchall()
                    today_trades = [dict(r) for r in rows]
            except Exception as e:
                logger.error(f"Error querying {db_file} for daily trades: {e}")

        for t in today_trades:
            if not t.get("symbol"):
                t["symbol"] = sym
            all_today_trades.append(t)

        # Market Context (Day Range)
        m_stats: Dict[str, Any] = {}
        if broker_connected:
            try:
                rates = broker.get_rates(bsym, timeframe="1d", count=2)
                if rates:
                    latest = rates[-1]
                    m_stats = {
                        "high": latest.get("high"),
                        "low": latest.get("low"),
                        "close": latest.get("close"),
                        "open": latest.get("open"),
                    }
            except Exception as e:
                logger.debug(f"Could not fetch live 1D rates for {bsym}: {e}")

        # Fallback to fetch_gold_candles if broker rates unavailable for Gold
        if not m_stats and sym == "XAUUSD":
            try:
                candles_1d = fetch_gold_candles("XAUUSD", period="5d", interval="1d")
                if candles_1d:
                    latest_day = candles_1d[-1]
                    m_stats = {
                        "high": latest_day.get("high"),
                        "low": latest_day.get("low"),
                        "close": latest_day.get("close"),
                    }
            except Exception:
                pass

        today_pnl = sum(float(t.get("pnl", 0.0)) for t in today_trades)
        today_wins = sum(1 for t in today_trades if float(t.get("pnl", 0.0)) > 0)
        today_losses = sum(1 for t in today_trades if float(t.get("pnl", 0.0)) < 0)
        today_scratches = sum(1 for t in today_trades if float(t.get("pnl", 0.0)) == 0)

        multi_asset_data[sym] = {
            "name": meta["name"],
            "icon": meta["icon"],
            "symbol": sym,
            "mode": state.get("ladder_mode", "FLAT_RISK"),
            "current_balance": float(state.get("current_balance", 1000.0)),
            "starting_balance": float(state.get("starting_balance", 1000.0)),
            "fixed_risk_usd": float(state.get("fixed_risk_usd", 75.44)),
            "compounding_enabled": bool(state.get("compounding_enabled", False)),
            "current_level": state.get("current_level", 18),
            "total_trades": state.get("total_trades", 0),
            "total_wins": state.get("wins", 0),
            "total_losses": state.get("losses", 0),
            "today_trades": today_trades,
            "today_pnl": today_pnl,
            "today_wins": today_wins,
            "today_losses": today_losses,
            "today_scratches": today_scratches,
            "market": m_stats,
        }

    # 3. Primary ladder status (Gold fallback for legacy single-asset callers)
    gold_status = multi_asset_data.get("XAUUSD", {})
    legacy_ladder_status = {
        "ladder_mode": gold_status.get("mode", "FLAT_RISK"),
        "compounding_enabled": gold_status.get("compounding_enabled", False),
        "fixed_risk_usd": gold_status.get("fixed_risk_usd", 75.44),
        "current_balance": gold_status.get("current_balance", 1000.0),
        "current_level": gold_status.get("current_level", 18),
        "wins": gold_status.get("total_wins", 0),
        "losses": gold_status.get("total_losses", 0),
    }

    # 4. Dispatch Telegram Notification (or print if Dry-Run)
    notifier = TelegramNotifier()
    if dry_run:
        logger.info("Dry-run requested: Rendering message without sending to Telegram...")
        # Override send_message temporarily to print
        original_send = notifier.send_message
        notifier.send_message = lambda text, parse_mode="HTML": print("\n--- RENDERED TELEGRAM REPORT ---\n" + text + "\n--------------------------------\n") or True
        success = notifier.send_daily_summary(
            date_str=date_str,
            account_info=account_info,
            ladder_status=legacy_ladder_status,
            today_trades=all_today_trades,
            open_positions=open_positions,
            market_stats=gold_status.get("market"),
            multi_asset_data=multi_asset_data,
        )
        notifier.send_message = original_send
        return success
    else:
        success = notifier.send_daily_summary(
            date_str=date_str,
            account_info=account_info,
            ladder_status=legacy_ladder_status,
            today_trades=all_today_trades,
            open_positions=open_positions,
            market_stats=gold_status.get("market"),
            multi_asset_data=multi_asset_data,
        )
        if success:
            logger.info("✅ Multi-asset daily session report dispatched successfully to Telegram.")
        else:
            logger.warning("⚠️ Daily session report delivery failed or Telegram is disabled.")
        return success


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Daily Session Reporter for Telegram")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format (default: today UTC)")
    parser.add_argument("--dry-run", action="store_true", help="Compile and print report without sending to Telegram")
    parser.add_argument("--send", action="store_true", help="Send report to Telegram (default)")
    args = parser.parse_args()

    compile_and_send_report(target_date=args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
