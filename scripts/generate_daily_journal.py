#!/usr/bin/env python3
"""
CLI & Operational Script to Generate Daily Trading Journals for Aegis Alpha.

Usage:
  # Generate today's journal
  python3 scripts/generate_daily_journal.py

  # Generate for a specific date
  python3 scripts/generate_daily_journal.py --date 2026-10-30

  # Backfill all historical dates
  python3 scripts/generate_daily_journal.py --backfill

  # List existing daily journals
  python3 scripts/generate_daily_journal.py --list

  # Generate and dispatch executive digest to Telegram
  python3 scripts/generate_daily_journal.py --telegram
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
import sys

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from gold_trader.core.daily_journal import DailyJournalEngine
from gold_trader.core.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("JournalGeneratorCLI")


def main():
    parser = argparse.ArgumentParser(description="Aegis Alpha Autonomous Daily Trading Journal Generator")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format (default: today UTC)")
    parser.add_argument("--backfill", action="store_true", help="Backfill journals for all dates found in database")
    parser.add_argument("--list", action="store_true", help="List summary of all existing journals")
    parser.add_argument("--stdout", action="store_true", help="Print generated markdown journal to standard output")
    parser.add_argument("--telegram", action="store_true", help="Dispatch executive digest to Telegram bot")
    args = parser.parse_args()

    engine = DailyJournalEngine()

    if args.list:
        journals = engine.list_journals()
        if not journals:
            print("No daily journals found in database.")
            return
        print(f"\n{'Date':12s} | {'P&L ($)':10s} | {'ROI %':8s} | {'Trades':6s} | {'Win %':7s} | {'PF':6s} | {'Net R':7s} | {'Markdown File'}")
        print("-" * 95)
        for j in journals:
            pnl_s = f"{'+' if j['total_pnl']>=0 else ''}${j['total_pnl']:.2f}"
            roi_s = f"{j['roi_pct']:+.1f}%"
            print(f"{j['date']:12s} | {pnl_s:10s} | {roi_s:8s} | {j['trades_count']:6d} | {j['win_rate']:5.1f}%  | {j['profit_factor']:5.2f} | {j['net_r_multiple']:+5.2f}R | {os.path.basename(j['markdown_file'])}")
        print("")
        return

    if args.backfill:
        logger.info("Starting historical journal backfill across all recorded database sessions...")
        files = engine.backfill_all()
        logger.info(f"✅ Successfully backfilled {len(files)} daily journals:")
        for f in files:
            logger.info(f"  • {f}")
        return

    target_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = engine.generate_journal(target_date=target_date)

    if args.stdout:
        print("\n" + result["markdown_content"] + "\n")
    else:
        logger.info(f"✅ Daily journal compiled successfully -> {result['markdown_path']}")
        perf = result["performance"]
        pnl_s = f"{'+' if perf['net_pnl_usd']>=0 else ''}${perf['net_pnl_usd']:.2f}"
        logger.info(f"   Summary: {perf['total_trades']} trades | Net P&L: {pnl_s} ({perf['net_r_multiple']:+.2f}R) | Win Rate: {perf['win_rate_pct']}%")

    if args.telegram:
        logger.info("Dispatching daily journal digest to Telegram...")
        notifier = TelegramNotifier()
        text = (
            f"<b>📔 AEGIS ALPHA DAILY TRADING JOURNAL</b>\n"
            f"<b>Session Date:</b> <code>{target_date}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Status:</b> {result['status']}\n"
            f"• <b>Net P&L:</b> <code>{'+' if perf['net_pnl_usd']>=0 else ''}${perf['net_pnl_usd']:.2f}</code> ({perf['net_r_multiple']:+.2f}R)\n"
            f"• <b>ROI:</b> <code>{perf['roi_pct']:+.2f}%</code>\n"
            f"• <b>Closed Trades:</b> <code>{perf['total_trades']}</code> ({perf['wins']}W - {perf['losses']}L - {perf['breakevens']}BE)\n"
            f"• <b>Win Rate:</b> <code>{perf['win_rate_pct']:.1f}%</code> | <b>PF:</b> <code>{perf['profit_factor']:.2f}</code>\n"
            f"• <b>Account Equity:</b> <code>${result['account']['ending_equity']:,.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📄 Full Markdown Journal saved to:\n<code>{result['markdown_path']}</code>"
        )
        success = notifier.send_message(text, parse_mode="HTML")
        if success:
            logger.info("✅ Telegram digest dispatched successfully.")
        else:
            logger.warning("⚠️ Failed to dispatch Telegram digest.")


if __name__ == "__main__":
    main()
