#!/usr/bin/env python3
"""
Monthly Cashflow & Treasury Yield Harvest Monitor for Profile A Crypto Engines.
Tracks:
- Current calendar month realized P&L across YieldA BTC & ETH engines
- Monthly Watermark Anchor ($1,500.00)
- Current Month Cash Harvest amount (Withdrawal Surplus)
- Any Deficit requiring Top-Up
- Complete Historical Payout Ledger
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

DB_PATHS = {
    "BTCUSD": os.path.expanduser("~/.gold_trading_data/yield_a_btc_auto_trader.db"),
    "ETHUSD": os.path.expanduser("~/.gold_trading_data/yield_a_eth_auto_trader.db"),
}

ANCHOR_CAPITAL = 1500.0


def get_monthly_deals_from_db() -> Dict[str, Dict[str, float]]:
    monthly_pnl = {}
    for sym, db in DB_PATHS.items():
        if not os.path.exists(db):
            continue
        try:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT strftime('%Y-%m', timestamp) as yr_mo, SUM(pnl) 
                FROM trade_executions 
                WHERE status = 'CLOSED' 
                GROUP BY yr_mo
                """
            )
            for yr_mo, pnl in cursor.fetchall():
                if yr_mo not in monthly_pnl:
                    monthly_pnl[yr_mo] = {"BTCUSD": 0.0, "ETHUSD": 0.0, "Total": 0.0}
                val = float(pnl or 0.0)
                monthly_pnl[yr_mo][sym] += val
                monthly_pnl[yr_mo]["Total"] += val
            conn.close()
        except Exception as e:
            print(f"Error reading DB {db}: {e}")

    return monthly_pnl


def main():
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    print("=" * 75)
    print(f"💰 PROFILE A: MONTHLY TREASURY YIELD HARVEST MONITOR ({current_month})")
    print(f"   Base Capital Anchor: ${ANCHOR_CAPITAL:,.2f}")
    print("=" * 75)

    monthly_pnl = get_monthly_deals_from_db()
    current_data = monthly_pnl.get(current_month, {"BTCUSD": 0.0, "ETHUSD": 0.0, "Total": 0.0})
    current_net = current_data["Total"]

    print(f"\n▶ Current Month ({current_month}) Realized P&L:")
    print(f"  • Bitcoin (BTCUSD):  ${current_data['BTCUSD']:>+9,.2f}")
    print(f"  • Ethereum (ETHUSD): ${current_data['ETHUSD']:>+9,.2f}")
    print(f"  • Combined Total:    ${current_net:>+9,.2f}")

    print("\n▶ Treasury Cashflow Action for This Month:")
    if current_net > 0:
        print(f"  ✅ SURPLUS AVAILABLE FOR WITHDRAWAL: +${current_net:,.2f}")
        print(f"     Action: Withdraw ${current_net:,.2f} into personal account.")
        print(f"     Trading balance will reset to the ${ANCHOR_CAPITAL:,.2f} anchor.")
    elif current_net < 0:
        deficit = abs(current_net)
        print(f"  ⚠️ DEFICIT DETECTED: -${deficit:,.2f}")
        print(f"     Action: Inject ${deficit:,.2f} top-up to restore anchor back to ${ANCHOR_CAPITAL:,.2f}.")
    else:
        print(f"  ⚖️ BREAK-EVEN: Balance is exactly at ${ANCHOR_CAPITAL:,.2f}. No action needed.")

    if len(monthly_pnl) > 1:
        print("\n▶ Historical Monthly Cashflow Ledger:")
        print(f"  {'Month':<8} | {'BTC ($)':<10} | {'ETH ($)':<10} | {'Combined ($)':<12} | {'Action':<15}")
        print("  " + "-" * 65)
        for mo in sorted(monthly_pnl.keys()):
            d = monthly_pnl[mo]
            act = f"WITHDRAW +${d['Total']:,.2f}" if d['Total'] > 0 else f"TOP-UP -${abs(d['Total']):,.2f}"
            print(f"  {mo:<8} | ${d['BTCUSD']:>9,.2f} | ${d['ETHUSD']:>9,.2f} | ${d['Total']:>11,.2f} | {act}")

    print("=" * 75)


if __name__ == "__main__":
    main()
