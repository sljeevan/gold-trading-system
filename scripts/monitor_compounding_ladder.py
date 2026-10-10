#!/usr/bin/env python3
"""
Live Monitor & Health Inspector for the Gold Noise-Adaptive Compounding Ladder.
Usage:
    python3 scripts/monitor_compounding_ladder.py
"""

import os
import sqlite3
import subprocess
from datetime import datetime, timezone

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.compounding_ladder import CompoundingLadderManager

def main():
    print("=" * 75)
    print(" 🌟 GOLD NOISE-ADAPTIVE COMPOUNDING LADDER — LIVE MONITOR & TELEMETRY")
    print("=" * 75)

    # 1. Ladder Manager State
    ladder = CompoundingLadderManager()
    status = ladder.get_status()
    specs = status.get("next_trade_specs", {})

    print(f"\n📈 [COMPOUNDING LADDER PROGRESSION]")
    print(f"   • Archetype:         {status.get('ladder_mode')}")
    print(f"   • Loss Policy:       {status.get('loss_policy')}")
    print(f"   • Active Level:      Level {status.get('current_level')} of 37 ({specs.get('phase', 'Sprint')} Phase)")
    print(f"   • Tracked Balance:   ${status.get('current_balance', 100.0):,.2f}")
    print(f"   • Peak Balance:      ${status.get('peak_balance', 100.0):,.2f}")
    print(f"   • Ratchet Floor:     ${status.get('ratchet_floor', 0.0):,.2f} (Locked Irreversible Principal)")
    print(f"   • Win / Loss Record: {status.get('wins', 0)}W - {status.get('losses', 0)}L (Streak: {status.get('consecutive_wins', 0)}W / {status.get('consecutive_losses', 0)}L)")
    print(f"   • Circuit Breaker:   {'ACTIVE (Risk Halved)' if status.get('circuit_breaker_active') else 'OFF (Normal Operations)'}")

    print(f"\n🎯 [UPCOMING TRADE PARAMETERS]")
    print(f"   • Risk Allocation:   ${specs.get('expected_risk_usd', 0.0):.2f} ({specs.get('risk_pct', 0.0)}% of capital)")
    print(f"   • Target Reward:     +${specs.get('expected_gain_usd', 0.0):.2f} ({specs.get('reward_risk_ratio', 1.5):.1f}R Payoff)")
    print(f"   • Stop Clearance:    Dynamic ATR (Min ${specs.get('min_sl_dollars', 8.0):.2f} / 80 pips)")
    print(f"   • Target Equity:     ${specs.get('projected_balance_on_win', 0.0):,.2f} (Advances to Level {status.get('current_level') + 1})")

    # 2. MT5 Broker Status
    print(f"\n🏦 [MT5 EXECUTION BRIDGE TELEMETRY]")
    broker = SocketBroker()
    connected = broker.connect()
    if connected:
        acct = broker.get_account_info()
        print(f"   • Connection:        ONLINE (TCP 127.0.0.1:8888)")
        print(f"   • Environment:       {acct.get('broker')} | Account: #{acct.get('account_id')}")
        print(f"   • Account Balance:   ${acct.get('balance', 0.0):,.2f} {acct.get('currency', 'USD')}")
        print(f"   • Account Equity:    ${acct.get('equity', 0.0):,.2f} | Leverage: 1:{acct.get('leverage')}")
        
        open_pos = broker.get_open_positions()
        print(f"   • Open Positions:    {len(open_pos)}")
        if open_pos:
            for p in open_pos:
                print(f"     -> #{p.get('ticket')}: {p.get('side')} {p.get('lots')} {p.get('symbol')} @ ${p.get('open_price')} (SL: ${p.get('sl')}, TP: ${p.get('tp')}, Profit: ${p.get('profit')})")
    else:
        print("   • Connection:        OFFLINE (MT5 Bridge Server not reachable)")

    # 3. AutoTrader Daemon Process
    print(f"\n🤖 [AUTO-TRADER DAEMON LIFECYCLE]")
    try:
        ps_out = subprocess.check_output(["pgrep", "-f", "run_auto_trader"], text=True).strip().splitlines()
        pids = ", ".join(ps_out) if ps_out else "None"
        print(f"   • Status:            RUNNING (PID: {pids} under tmux session 'gold-auto-trader')")
    except Exception:
        print("   • Status:            STOPPED")

    # 4. Recent Database Trade Executions
    db_path = os.path.expanduser("~/.gold_trading_data/auto_trader.db")
    if os.path.exists(db_path):
        print(f"\n📜 [RECENT COMPLETED & ACTIVE TRADES]")
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                rows = cur.execute(
                    "SELECT id, timestamp, ticket, action, lots, entry_price, stop_loss, target_1, status, pnl "
                    "FROM trade_executions ORDER BY id DESC LIMIT 5"
                ).fetchall()
                if rows:
                    print(f"   {'ID':<4} {'Timestamp':<20} {'Ticket':<10} {'Side':<5} {'Lots':<6} {'Entry':<9} {'SL':<9} {'TP':<9} {'Status':<7} {'PnL ($)'}")
                    print("   " + "-" * 88)
                    for r in rows:
                        pnl_str = f"${r[9]:+.2f}" if r[9] is not None else "-"
                        print(f"   {r[0]:<4} {r[1][:19]:<20} {r[2]:<10} {r[3]:<5} {r[4]:<6.2f} {r[5]:<9.2f} {r[6]:<9.2f} {r[7]:<9.2f} {r[8]:<7} {pnl_str}")
                else:
                    print("   • No executed trades recorded yet. Scanning market for next confluence setup.")
        except Exception as db_err:
            print(f"   • Error reading trade executions: {db_err}")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    main()
