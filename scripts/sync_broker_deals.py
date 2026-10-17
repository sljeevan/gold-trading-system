#!/usr/bin/env python3
"""
Broker Deals Reconciliation Script for XM MetaTrader 5.
Synchronizes SQLite database trade_executions and Compounding Ladder state
with exact closed deal records directly from the XM MT5 terminal.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BrokerDealReconciler")

# Ensure project root in python path
sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.compounding_ladder import (
    CompoundingLadderManager,
    NOISE_ADAPTIVE_MATRIX,
    find_closest_level_for_balance,
)


def fetch_all_mt5_deals_via_wine() -> dict[str, dict]:
    """Execute Wine Python script to pull complete deal history directly from MT5."""
    cmd = [
        "xvfb-run",
        "-a",
        "/home/mcpuser/bin/wine",
        r"C:\python64\python.exe",
        "-c",
        """
import MetaTrader5 as mt5
import datetime, json

if not mt5.initialize():
    print(json.dumps({"error": str(mt5.last_error())}))
    exit(1)

date_from = datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc)
date_to = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
deals = mt5.history_deals_get(date_from, date_to)

positions = {}
if deals:
    for d in deals:
        pos_id = str(d.position_id)
        if pos_id not in positions:
            positions[pos_id] = {
                'position_id': pos_id,
                'symbol': d.symbol,
                'total_profit': 0.0,
                'total_commission': 0.0,
                'total_swap': 0.0,
                'entry_price': None,
                'close_price': None,
                'open_time': None,
                'close_time': None,
                'comment': '',
                'type': 'BUY' if d.type == 0 else 'SELL',
            }
        p = positions[pos_id]
        p['total_profit'] += float(d.profit)
        p['total_commission'] += float(getattr(d, 'commission', 0.0))
        p['total_swap'] += float(getattr(d, 'swap', 0.0))
        
        dt_str = datetime.datetime.fromtimestamp(d.time, tz=datetime.timezone.utc).isoformat()
        if d.entry == 0:
            p['entry_price'] = float(d.price)
            p['open_time'] = dt_str
            if d.comment:
                p['comment'] = d.comment
        elif d.entry in (1, 2, 3):
            p['close_price'] = float(d.price)
            p['close_time'] = dt_str
            if d.comment:
                p['comment'] = d.comment

mt5.shutdown()
print(json.dumps(positions))
""",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        output = res.stdout.strip()
        # Find json object in stdout
        start_idx = output.find("{")
        if start_idx != -1:
            data = json.loads(output[start_idx:])
            return data
    except Exception as e:
        logger.error("Failed to query MT5 deals via Wine: %s", e)
    return {}


def sync_database_and_ladder(apply_fixes: bool = True) -> dict:
    db_path = os.path.expanduser("~/.gold_trading_data/auto_trader.db")
    if not os.path.exists(db_path):
        logger.error("DB not found at %s", db_path)
        return {"status": "ERROR", "message": "DB not found"}

    mt5_deals = fetch_all_mt5_deals_via_wine()
    if not mt5_deals:
        logger.error("No MT5 deals retrieved.")
        return {"status": "ERROR", "message": "No MT5 deals"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, ticket, entry_price, close_price, pnl, timestamp, close_time FROM trade_executions")
    rows = cur.fetchall()

    reconciled_trades = []
    for db_id, ticket, cur_entry, cur_close, cur_pnl, open_ts, close_ts in rows:
        ticket_str = str(ticket)
        if ticket_str in mt5_deals:
            m = mt5_deals[ticket_str]
            mt5_profit = round(m["total_profit"], 2)
            mt5_comm = round(m["total_commission"], 2)
            mt5_swap = round(m["total_swap"], 2)
            mt5_net_pnl = round(mt5_profit + mt5_comm + mt5_swap, 2)
            mt5_close_price = m["close_price"] or cur_close
            mt5_close_time = m["close_time"] or close_ts

            diff = round(mt5_net_pnl - (cur_pnl or 0.0), 2)
            if abs(diff) > 0.05 or (mt5_close_price and abs(mt5_close_price - (cur_close or 0.0)) > 0.05):
                reconciled_trades.append({
                    "id": db_id,
                    "ticket": ticket_str,
                    "old_close": cur_close,
                    "new_close": mt5_close_price,
                    "old_pnl": cur_pnl,
                    "new_pnl": mt5_net_pnl,
                    "diff": diff,
                    "close_time": mt5_close_time,
                })

                if apply_fixes:
                    cur.execute(
                        """
                        UPDATE trade_executions 
                        SET close_price = ?, pnl = ?, close_time = ?, status = 'CLOSED'
                        WHERE id = ?
                        """,
                        (mt5_close_price, mt5_net_pnl, mt5_close_time, db_id),
                    )

    if apply_fixes:
        conn.commit()
    conn.close()

    logger.info("Reconciled %d trade records in SQLite database.", len(reconciled_trades))

    # Recalibrate compounding ladder state based on true verified trades post-Oct 10
    ladder = CompoundingLadderManager()
    ladder_status = ladder.get_status()

    # Calculate true post-Oct 10 performance from MT5 deals
    # Trades post Oct 10 00:00 UTC:
    # 2341508531 (+14.69), 2342451369 (-71.17), 2344573123 (-82.35), 2344734155 (+95.36)
    post_oct10_tickets = ["2341508531", "2342451369", "2344573123", "2344734155"]
    
    current_bal = 1000.0  # starting baseline
    ladder_history = []
    consec_w = 0
    consec_l = 0
    wins = 0
    losses = 0
    scratches = 0
    cur_level = find_closest_level_for_balance(current_bal, matrix=NOISE_ADAPTIVE_MATRIX)

    for idx, t in enumerate(post_oct10_tickets, 1):
        if t in mt5_deals:
            d = mt5_deals[t]
            net_pnl = round(d["total_profit"] + d["total_commission"] + d["total_swap"], 2)
            is_win = net_pnl > 0.35
            is_be = abs(net_pnl) <= 0.35
            is_loss = net_pnl < -0.35

            prev_level = cur_level
            current_bal = round(current_bal + net_pnl, 2)

            if is_win:
                wins += 1
                consec_w += 1
                consec_l = 0
                cur_level = min(cur_level + 1, len(NOISE_ADAPTIVE_MATRIX))
                outcome = "WIN"
            elif is_loss:
                losses += 1
                consec_l += 1
                consec_w = 0
                cur_level = max(cur_level - 1, 1)
                outcome = "LOSS"
            else:
                scratches += 1
                outcome = "BREAKEVEN"

            ladder_history.append({
                "trade_no": idx,
                "ticket": t,
                "level": prev_level,
                "outcome": outcome,
                "win": is_win,
                "is_breakeven": is_be,
                "pnl": net_pnl,
                "exit_price": d["close_price"],
                "balance_after": current_bal,
                "ratchet_floor": 700.0,
                "new_level": cur_level,
                "timestamp": d["close_time"],
                "notes": f"Verified XM Deal #{t}: {outcome} (${net_pnl:+.2f})",
            })

    if apply_fixes:
        ladder.state["starting_balance"] = 1000.0
        ladder.state["current_balance"] = current_bal
        ladder.state["peak_balance"] = 1014.69
        ladder.state["current_level"] = cur_level
        ladder.state["peak_level"] = 19
        ladder.state["total_trades"] = len(ladder_history)
        ladder.state["wins"] = wins
        ladder.state["losses"] = losses
        ladder.state["scratches"] = scratches
        ladder.state["consecutive_wins"] = consec_w
        ladder.state["consecutive_losses"] = consec_l
        ladder.state["ratchet_floor"] = 700.0
        ladder.state["history"] = ladder_history
        ladder.state["status"] = "ACTIVE"
        ladder._save_state()
        logger.info("Recalibrated compounding ladder: Balance=$%.2f, Level=%d, Wins=%d, Losses=%d", current_bal, cur_level, wins, losses)

    return {
        "status": "SUCCESS",
        "reconciled_trades_count": len(reconciled_trades),
        "reconciled_trades": reconciled_trades,
        "new_balance": current_bal,
        "new_level": cur_level,
    }


if __name__ == "__main__":
    result = sync_database_and_ladder(apply_fixes=True)
    print(json.dumps(result, indent=2))
