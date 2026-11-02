#!/usr/bin/env python3
"""
UAT Runner & Real-Time Monitor for ATAS-Grade Order Flow Analytics.

Streams live trades from Bybit / Binance WebSockets, builds footprint candlesticks,
computes CVD, VPOC, VAH/VAL, stacked imbalances, and renders a live terminal dashboard.

Usage:
    python3 scripts/run_order_flow_uat.py --symbol BTCUSDT --feed bybit --timeframe 15
    python3 scripts/run_order_flow_uat.py --symbol ETHUSDT --feed bybit --duration 60
    python3 scripts/run_order_flow_uat.py --symbol BTCUSDT --status
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gold_trader.core.order_flow import (
    DEFAULT_UAT_DB_PATH,
    OrderFlowDB,
    OrderFlowEngine,
    WebSocketOrderFlowClient,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OrderFlowUAT")


def print_dashboard(engine: OrderFlowEngine, feed: str) -> None:
    """Render a clean ASCII order-flow dashboard to stdout."""
    m = engine.get_developing_metrics()
    if m.get("status") == "AWAITING_DATA":
        print(f"\r⏳ [{engine.symbol} | {feed.upper()}] Awaiting incoming trades...", end="", flush=True)
        return

    # ANSI color helpers
    C_GREEN = "\033[92m"
    C_RED = "\033[91m"
    C_YELLOW = "\033[93m"
    C_CYAN = "\033[96m"
    C_BOLD = "\033[1m"
    C_RESET = "\033[0m"

    delta_val = m["delta"]
    d_color = C_GREEN if delta_val >= 0 else C_RED
    cvd_color = C_GREEN if m["cvd"] >= 0 else C_RED

    print("\033[2J\033[H")  # Clear screen and move cursor home
    print(f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║     AEGIS ALPHA — ATAS ORDER FLOW ENGINE (UAT MICROSTRUCTURE MONITOR)            ║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f" Asset: {C_BOLD}{m['symbol']}{C_RESET} | Feed: {C_BOLD}{feed.upper()}{C_RESET} | TF: {m['timeframe']} | Bar Time: {m['bar_time']}")
    print("─" * 82)
    print(f" OHLC:  O: {m['open']:.2f} | H: {m['high']:.2f} | L: {m['low']:.2f} | C: {m['close']:.2f}")
    print(f" Volume: Total: {m['total_volume']:.4f} (Buy: {m['buy_volume']:.4f} | Sell: {m['sell_volume']:.4f})")
    print(f" Delta:  {d_color}{m['delta']:+.4f} ({m['delta_pct']}){C_RESET} | Session CVD: {cvd_color}{m['cvd']:+.4f}{C_RESET}")
    print("─" * 82)
    print(f" Market Profile:  {C_YELLOW}VPOC: {m['vpoc']:.2f}{C_RESET} | VAH: {m['vah']:.2f} | VAL: {m['val']:.2f}")

    # Imbalance badges
    b_imbal = f"{C_GREEN}{m['stacked_buy_zones']} STACKED BUY ZONES{C_RESET}" if m['stacked_buy_zones'] > 0 else "None"
    s_imbal = f"{C_RED}{m['stacked_sell_zones']} STACKED SELL ZONES{C_RESET}" if m['stacked_sell_zones'] > 0 else "None"
    print(f" Imbalances:      Buy: {b_imbal} | Sell: {s_imbal}")

    # Microstructure alerts
    div_badge = f"{C_YELLOW}{m['divergence']}{C_RESET}" if m['divergence'] != "NONE" else "None"
    abs_badge = f"{C_CYAN}{m['absorption']}{C_RESET}" if m['absorption'] != "NONE" else "None"
    print(f" Micro Alerts:    Divergence: {div_badge} | Absorption: {abs_badge}")
    print("─" * 82)

    # Print top 8 footprint price levels centered around VPOC
    if engine.current_bar and engine.current_bar.levels:
        print(f"{C_BOLD} Developing Footprint (Price x Bid/Ask Volume):{C_RESET}")
        print("  Price Level     |  Sell Vol (Bid)  |  Buy Vol (Ask)   |  Level Delta")
        print("  ────────────────┼──────────────────┼──────────────────┼──────────────")
        sorted_p = sorted(engine.current_bar.levels.keys(), reverse=True)
        # Select up to 10 levels
        disp_p = sorted_p[:10]
        for p in disp_p:
            lvl = engine.current_bar.levels[p]
            poc_marker = f"{C_YELLOW}★ POC{C_RESET}" if abs(p - m['vpoc']) < 1e-4 else "     "
            lvl_color = C_GREEN if lvl.delta >= 0 else C_RED
            print(f"  {p:>10.2f} {poc_marker} | {lvl.sell_volume:>16.4f} | {lvl.buy_volume:>16.4f} | {lvl_color}{lvl.delta:>+12.4f}{C_RESET}")

    print("─" * 82)
    print(" Press Ctrl+C to stop monitor and persist session data.")


def show_status(db_path: str, symbol: str) -> None:
    """Display historical UAT SQLite summary."""
    db = OrderFlowDB(db_path)
    bars = db.get_recent_bars(symbol=symbol.upper(), timeframe="15m", limit=10)
    print(f"\n=== UAT Order Flow History for {symbol.upper()} (DB: {db_path}) ===")
    if not bars:
        print(f"No order-flow bars recorded yet for {symbol.upper()}.")
        return

    print("Time (UTC)           | Close     | Volume      | Delta        | CVD          | VPOC      | VAH       | VAL")
    print("─────────────────────┼───────────┼─────────────┼──────────────┼──────────────┼───────────┼───────────┼──────────")
    for b in bars:
        ts_str = datetime.fromtimestamp(b["bar_start_ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"{ts_str} | {b['close']:>9.2f} | {b['total_volume']:>11.4f} | {b['delta']:>+12.4f} | {b['cvd']:>+12.4f} | {b['vpoc']:>9.2f} | {b['vah']:>9.2f} | {b['val']:>9.2f}")


async def async_main(args: argparse.Namespace) -> None:
    if args.status:
        show_status(args.db_path, args.symbol)
        return

    engine = OrderFlowEngine(
        symbol=args.symbol,
        timeframe_minutes=args.timeframe,
        tick_size=args.tick_size,
        imbalance_ratio=args.imbalance_ratio,
        db_path=args.db_path,
    )

    client = WebSocketOrderFlowClient(engine=engine, feed=args.feed)

    # Signal callbacks
    def on_signal(sig_type: str, details: dict):
        logger.info(f"🚨 ORDER FLOW SIGNAL: {sig_type} -> {json.dumps(details)}")

    engine.signal_callbacks.append(on_signal)

    # Start WS worker in background
    ws_task = asyncio.create_task(client.start())

    # Dashboard loop
    start_time = time.time()
    try:
        while True:
            await asyncio.sleep(args.refresh)
            print_dashboard(engine, args.feed)

            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration limit of {args.duration}s reached. Exiting...")
                break
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        client.stop()
        ws_task.cancel()
        if engine.current_bar and engine.current_bar.total_volume > 0:
            engine.current_bar.finalize()
            engine.db.save_bar(engine.current_bar)
            print(f"\n[UAT] Finalized and persisted {engine.symbol} order-flow bar to SQLite.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATAS Order Flow UAT System")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (BTCUSDT, ETHUSDT, PAXGUSDT)")
    parser.add_argument("--feed", type=str, default="bybit", choices=["bybit", "binance"], help="WebSocket Feed")
    parser.add_argument("--timeframe", type=int, default=15, help="Candle timeframe in minutes (default 15)")
    parser.add_argument("--tick-size", type=float, default=None, help="Discrete tick size for footprint price bucketing")
    parser.add_argument("--imbalance-ratio", type=float, default=3.0, help="Diagonal imbalance ratio (default 3.0 = 300%)")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--refresh", type=float, default=1.0, help="Dashboard UI refresh interval in seconds")
    parser.add_argument("--db-path", type=str, default=DEFAULT_UAT_DB_PATH, help="Isolated UAT SQLite path")
    parser.add_argument("--status", action="store_true", help="Print recent historical UAT records and exit")

    args = parser.parse_args()

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nExiting Order Flow UAT monitor.")


if __name__ == "__main__":
    main()
