#!/usr/bin/env python3
"""
Daily Trading Session Summary Reporter.
Compiles daily realized PnL, 30-level compounding ladder status,
open exposure, and market context, then dispatches directly to Telegram.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.compounding_ladder import CompoundingLadderManager, get_ladder_level_info
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("DailySessionReporter")


def compile_and_send_report() -> bool:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info(f"📊 Compiling daily trading session report for {today_str}...")

    # 1. Broker Account Info & Open Positions
    broker = SocketBroker()
    account_info = broker.get_account_info()
    open_positions = broker.get_open_positions()

    # 2. Compounding Ladder State
    ladder = CompoundingLadderManager()
    ladder_status = ladder.get_status()
    current_lvl = ladder_status.get("current_level", 1)
    lvl_info = get_ladder_level_info(current_lvl)
    ladder_status["next_lot_size"] = lvl_info["lot_size"]
    ladder_status["next_level_balance"] = lvl_info["ending_balance"]

    # 3. Query Today's Closed Trades from SQLite Database
    db_path = os.path.expanduser("~/.gold_trading_data/auto_trader.db")
    today_trades = []
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                rows = cursor.execute("""
                    SELECT ticket, symbol, action, lots, entry_price, close_price, pnl, close_time
                    FROM trade_executions
                    WHERE (close_time LIKE ? OR timestamp LIKE ?) AND status = 'CLOSED'
                    ORDER BY id ASC
                """, (f"{today_str}%", f"{today_str}%")).fetchall()
                today_trades = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error querying database for daily trades: {e}")

    # 4. Market Context (Gold Day Range)
    market_stats = {}
    try:
        candles_1d = fetch_gold_candles("XAUUSD", period="5d", interval="1d")
        if candles_1d:
            latest_day = candles_1d[-1]
            market_stats = {
                "high": latest_day.get("high"),
                "low": latest_day.get("low"),
                "close": latest_day.get("close"),
            }
    except Exception as e:
        logger.warning(f"Could not fetch day market stats: {e}")

    # 5. Dispatch Telegram Notification
    notifier = TelegramNotifier()
    success = notifier.send_daily_summary(
        date_str=today_str,
        account_info=account_info,
        ladder_status=ladder_status,
        today_trades=today_trades,
        open_positions=open_positions,
        market_stats=market_stats if market_stats else None,
    )

    if success:
        logger.info("✅ Daily session report dispatched successfully to Telegram.")
    else:
        logger.warning("⚠️ Daily session report delivery failed or Telegram is disabled.")

    return success


if __name__ == "__main__":
    compile_and_send_report()
