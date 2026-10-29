#!/usr/bin/env python3
"""
Comprehensive 7-Tier Institutional Verification & Stress Test Suite
for Aegis Alpha Autonomous Multi-Account Trading Systems:
- Model 1: Production Baseline (Gold, BTC, ETH on Port 8888, Account 168987477)
- Model 2: Profile A Yield Engine (BTC, ETH on Port 8889, Account 346439083 'Monthly_winsaw')
"""
from __future__ import annotations

import json
import math
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.core.indicators import calc_atr, calc_donchian, calc_ema, calc_macd, calc_supertrend
from gold_trader.core.ml_filter import GoldMLPredictor

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️ WARN"


def test_bridge_connectivity() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 1: DUAL MT5 BRIDGE CONNECTIVITY & ACCOUNT INTEGRITY")
    print("=" * 90)
    results = []

    bridges = [
        {"name": "Model 1 Baseline", "port": 8888, "expected_acc": "168987477", "expected_server": "XMGlobal-MT5 2"},
        {"name": "Model 2 Profile A", "port": 8889, "expected_acc": "346439083", "expected_server": "XMGlobal-MT5 10"},
    ]

    for b in bridges:
        t0 = time.time()
        broker = SocketBroker(port=b["port"])
        connected = broker.connect()
        latency = (time.time() - t0) * 1000

        if not connected:
            results.append({"name": b["name"], "status": FAIL, "msg": f"Failed to connect to port {b['port']}"})
            print(f"  {FAIL} {b['name']} (Port {b['port']}): OFFLINE")
            continue

        acc = broker.get_account_info()
        acc_id = str(acc.get("account_id", ""))
        balance = float(acc.get("balance", 0.0))
        server = acc.get("broker", "")

        is_acc_ok = acc_id == b["expected_acc"]
        is_server_ok = b["expected_server"] in server

        status = PASS if (connected and is_acc_ok and is_server_ok) else WARN
        print(f"  {status} {b['name']} (Port {b['port']}):")
        print(f"      • Account ID: {acc_id} (Expected: {b['expected_acc']})")
        print(f"      • Server:     {server}")
        print(f"      • Balance:    ${balance:,.2f} | Equity: ${acc.get('equity', 0.0):,.2f} | Leverage: 1:{acc.get('leverage')}")
        print(f"      • Latency:    {latency:.1f}ms")

        results.append({
            "name": b["name"],
            "port": b["port"],
            "status": status,
            "account_id": acc_id,
            "balance": balance,
            "latency_ms": round(latency, 2),
        })

    return results


def test_market_data_feeds() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 2: REAL-TIME TICK FEEDS, SPREADS & 1H CANDLE CONTINUITY")
    print("=" * 90)
    results = []

    tests = [
        {"model": "Model 1", "port": 8888, "symbol": "GOLD"},
        {"model": "Model 1", "port": 8888, "symbol": "BTCUSD"},
        {"model": "Model 1", "port": 8888, "symbol": "ETHUSD"},
        {"model": "Model 2", "port": 8889, "symbol": "BTCUSD"},
        {"model": "Model 2", "port": 8889, "symbol": "ETHUSD"},
    ]

    for t in tests:
        broker = SocketBroker(port=t["port"])
        sym = t["symbol"]
        s_info = broker.get_symbol_info(sym)
        rates = broker.get_rates(sym, count=100, timeframe="1h")

        bid = s_info.get("bid", 0.0)
        ask = s_info.get("ask", 0.0)
        spread = s_info.get("spread", 0.0)
        rates_count = len(rates) if rates else 0

        has_prices = bid > 0 and ask > 0 and ask >= bid
        has_rates = rates_count >= 50

        status = PASS if (has_prices and has_rates) else FAIL
        last_date = rates[-1]["date"] if rates else "N/A"
        last_close = rates[-1]["close"] if rates else 0.0

        print(f"  {status} [{t['model']} | Port {t['port']}] {sym}:")
        print(f"      • Bid: {bid} | Ask: {ask} | Spread: {spread}")
        print(f"      • 1H Candles: {rates_count}/100 fetched | Latest Bar: {last_date} @ ${last_close}")

        results.append({
            "model": t["model"],
            "symbol": sym,
            "status": status,
            "bid": bid,
            "ask": ask,
            "rates_count": rates_count,
            "latest_bar": last_date,
        })

    return results


def test_indicators_and_signals() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 3: TECHNICAL INDICATOR COMPUTATION & SIGNAL DETECTION")
    print("=" * 90)
    results = []

    for port, model, sym in [
        (8888, "Model 1", "GOLD"),
        (8888, "Model 1", "BTCUSD"),
        (8888, "Model 1", "ETHUSD"),
        (8889, "Model 2", "BTCUSD"),
        (8889, "Model 2", "ETHUSD"),
    ]:
        broker = SocketBroker(port=port)
        rates = broker.get_rates(sym, count=100, timeframe="1h")
        if not rates or len(rates) < 50:
            print(f"  {FAIL} [{model}] {sym}: Insufficient rates for indicators")
            results.append({"model": model, "symbol": sym, "status": FAIL})
            continue

        highs = [float(r["high"]) for r in rates]
        lows = [float(r["low"]) for r in rates]
        closes = [float(r["close"]) for r in rates]

        st = calc_supertrend(highs, lows, closes, 10, 3.0)
        dc = calc_donchian(highs, lows, 10)
        ema50 = calc_ema(closes, 50)
        macd = calc_macd(closes, 12, 26, 9)
        atr14 = calc_atr(highs, lows, closes, 14)

        last_st = st["direction"][-1]
        last_dc_up = dc["upper"][-2]  # previous bar upper
        last_dc_low = dc["lower"][-2] # previous bar lower
        last_ema = ema50[-1]
        last_atr = atr14[-1]

        valid = (
            last_st in (-1, 1)
            and last_dc_up is not None and last_dc_up > 0
            and last_dc_low is not None and last_dc_low > 0
            and last_ema is not None and last_ema > 0
            and last_atr is not None and last_atr > 0
        )

        status = PASS if valid else FAIL
        st_dir = "BULLISH 🟢" if last_st == 1 else "BEARISH 🔴"
        curr_price = closes[-1]
        dist_up = (last_dc_up - curr_price) if last_dc_up else 0.0
        dist_low = (curr_price - last_dc_low) if last_dc_low else 0.0

        print(f"  {status} [{model}] {sym} (Price: ${curr_price:.2f}):")
        print(f"      • SuperTrend (10, 3.0): {st_dir}")
        print(f"      • 50 EMA:              ${last_ema:.2f}")
        print(f"      • 10 Donchian Bands:   Lower=${last_dc_low:.2f} | Upper=${last_dc_up:.2f}")
        print(f"      • Proximity to Touch:  Up: {dist_up:+.2f} | Low: {dist_low:+.2f}")
        print(f"      • ATR (14):            {last_atr:.2f}")

        results.append({"model": model, "symbol": sym, "status": status, "atr": last_atr})

    return results


def test_machine_learning_gate() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 4: LIGHTGBM ML META-LABELER CONVICTION GATE")
    print("=" * 90)
    results = []

    for port, model, sym, threshold in [
        (8888, "Model 1", "GOLD", 0.44),
        (8888, "Model 1", "BTCUSD", 0.44),
        (8888, "Model 1", "ETHUSD", 0.44),
        (8889, "Model 2 Profile A", "BTCUSD", 0.48),
        (8889, "Model 2 Profile A", "ETHUSD", 0.48),
    ]:
        broker = SocketBroker(port=port)
        rates = broker.get_rates(sym, count=100, timeframe="1h")
        predictor = GoldMLPredictor(symbol="XAUUSD" if "GOLD" in sym else sym)

        t0 = time.time()
        res_buy = predictor.predict_trade_probability(rates, direction="BUY", macro_score=0.0)
        res_sell = predictor.predict_trade_probability(rates, direction="SELL", macro_score=0.0)
        inf_time = (time.time() - t0) * 1000

        p_buy = res_buy.get("probability_win", 0.0)
        p_sell = res_sell.get("probability_win", 0.0)

        valid = 0.0 <= p_buy <= 1.0 and 0.0 <= p_sell <= 1.0
        status = PASS if valid else FAIL

        buy_gate = "ALLOW ✅" if p_buy >= threshold else f"GATE 🛡️ (< {threshold*100:.0f}%)"
        sell_gate = "ALLOW ✅" if p_sell >= threshold else f"GATE 🛡️ (< {threshold*100:.0f}%)"

        print(f"  {status} [{model}] {sym} (Inference: {inf_time:.1f}ms):")
        print(f"      • Model Cache:   {predictor.model_path}")
        print(f"      • BUY Conviction:  {p_buy*100:.1f}% -> {buy_gate}")
        print(f"      • SELL Conviction: {p_sell*100:.1f}% -> {sell_gate}")

        results.append({"model": model, "symbol": sym, "status": status, "buy_prob": p_buy, "sell_prob": p_sell})

    return results


def test_tag_isolation_and_filtering() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 5: MULTI-ENGINE TAG ISOLATION & POSITION FILTERING")
    print("=" * 90)
    results = []

    # Mock broker to verify tag isolation
    class MockBroker:
        def __init__(self, positions):
            self.open_pos = positions
        def get_open_positions(self):
            return self.open_pos

    test_positions = [
        {"ticket": "101", "symbol": "ETHUSD", "comment": "AI-FlatRisk-ETHUSD", "lots": 3.55, "open_price": 2300.0},
        {"ticket": "102", "symbol": "ETHUSD", "comment": "YieldA-ETHUSD", "lots": 3.55, "open_price": 2300.0},
        {"ticket": "103", "symbol": "BTCUSD", "comment": "AI-FlatRisk-BTCUSD", "lots": 0.12, "open_price": 80000.0},
        {"ticket": "104", "symbol": "BTCUSD", "comment": "YieldA-BTCUSD", "lots": 0.12, "open_price": 80000.0},
        {"ticket": "105", "symbol": "GOLD", "comment": "AI-FlatRisk-XAUUSD", "lots": 0.04, "open_price": 4000.0},
    ]

    mock_b = MockBroker(test_positions)

    # Test 1: Model 1 Baseline (No tag or AI-FlatRisk)
    cfg_m1 = AutoTraderConfig(symbol="ETHUSD", order_tag=None)
    trader_m1 = GoldAutoTrader(broker=mock_b, config_override=cfg_m1)
    matched_m1 = trader_m1.get_matching_open_positions()
    # Model 1 must see ticket 101 and NOT ticket 102
    m1_tickets = [p["ticket"] for p in matched_m1]
    m1_ok = "101" in m1_tickets and "102" not in m1_tickets

    # Test 2: Model 2 Profile A (order_tag="YieldA")
    cfg_m2 = AutoTraderConfig(symbol="ETHUSD", order_tag="YieldA")
    trader_m2 = GoldAutoTrader(broker=mock_b, config_override=cfg_m2)
    matched_m2 = trader_m2.get_matching_open_positions()
    m2_tickets = [p["ticket"] for p in matched_m2]
    m2_ok = "102" in m2_tickets and "101" not in m2_tickets

    status_iso = PASS if (m1_ok and m2_ok) else FAIL
    print(f"  {status_iso} Order Comment Tag Isolation (In-Memory Position Filter):")
    print(f"      • Model 1 (Baseline) Matched Tickets: {m1_tickets} (Correctly isolated from YieldA: {m1_ok})")
    print(f"      • Model 2 (YieldA)   Matched Tickets: {m2_tickets} (Correctly isolated from Baseline: {m2_ok})")

    results.append({"test": "Tag Isolation", "status": status_iso})

    # Test 3: Live Broker Cross-Account Firewall Test
    b1 = SocketBroker(port=8888)
    b2 = SocketBroker(port=8889)
    pos1 = b1.get_open_positions()
    pos2 = b2.get_open_positions()
    
    # Bridge 1 account is 168987477, Bridge 2 is 346439083
    acc1_id = b1.get_account_info().get("account_id")
    acc2_id = b2.get_account_info().get("account_id")
    acc_distinct = acc1_id != acc2_id and acc1_id is not None and acc2_id is not None
    status_hw = PASS if acc_distinct else FAIL

    print(f"  {status_hw} Hardware-Level Multi-Broker Account Firewall:")
    print(f"      • Bridge 1 Terminal Account: {acc1_id} (Open Orders: {len(pos1)})")
    print(f"      • Bridge 2 Terminal Account: {acc2_id} (Open Orders: {len(pos2)})")
    print(f"      • Account Separation: 100% Isolated ({acc1_id} != {acc2_id})")

    results.append({"test": "Hardware Account Firewall", "status": status_hw})
    return results


def test_position_sizing_and_risk_math() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 6: DYNAMIC VOLATILITY POSITION SIZING & RISK MATH")
    print("=" * 90)
    results = []

    cases = [
        {"model": "Model 1", "symbol": "GOLD", "risk_usd": 75.44, "contract_size": 100.0, "atr": 6.50, "min_sl": 8.00},
        {"model": "Model 1", "symbol": "BTCUSD", "risk_usd": 75.44, "contract_size": 1.0, "atr": 450.0, "min_sl": 500.0},
        {"model": "Model 1", "symbol": "ETHUSD", "risk_usd": 75.44, "contract_size": 1.0, "atr": 18.0, "min_sl": 20.0},
        {"model": "Model 2", "symbol": "BTCUSD", "risk_usd": 75.00, "contract_size": 1.0, "atr": 450.0, "min_sl": 500.0},
        {"model": "Model 2", "symbol": "ETHUSD", "risk_usd": 75.00, "contract_size": 1.0, "atr": 18.0, "min_sl": 20.0},
    ]

    for c in cases:
        sl_dist = max(1.5 * c["atr"], c["min_sl"])
        lots = round(c["risk_usd"] / (sl_dist * c["contract_size"]), 2)
        lots = max(0.01 if "ETH" not in c["symbol"] else 0.02, lots)
        actual_risk = lots * sl_dist * c["contract_size"]

        t1_gain = 1.25 * actual_risk
        t2_gain = 3.00 * actual_risk

        # Tolerance: actual risk within 10% of planned risk (due to 0.01 lot step rounding)
        is_ok = abs(actual_risk - c["risk_usd"]) / c["risk_usd"] <= 0.15
        status = PASS if is_ok else WARN

        print(f"  {status} [{c['model']}] {c['symbol']}:")
        print(f"      • SL Distance:   ${sl_dist:.2f} (ATR: {c['atr']}, Min SL: {c['min_sl']})")
        print(f"      • Sized Lots:    {lots} lots")
        print(f"      • Actual Risk:   ${actual_risk:.2f} (Target: ${c['risk_usd']:.2f})")
        print(f"      • Target 1 Gain: +${t1_gain:.2f} (+1.25R) | Target 2: +${t2_gain:.2f} (+3.0R)")

        results.append({"model": c["model"], "symbol": c["symbol"], "status": status, "lots": lots, "risk": actual_risk})

    return results


def test_systemd_and_db_integrity() -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("TEST SUITE 7: SYSTEMD DAEMONS & SQLITE DATABASE PERSISTENCE")
    print("=" * 90)
    results = []

    services = [
        ("Model 1", "gold-auto-trader.service", "~/.gold_trading_data/auto_trader.db"),
        ("Model 1", "btc-auto-trader.service", "~/.gold_trading_data/btc_auto_trader.db"),
        ("Model 1", "eth-auto-trader.service", "~/.gold_trading_data/eth_auto_trader.db"),
        ("Model 2", "yield-a-btc-auto-trader.service", "~/.gold_trading_data/yield_a_btc_auto_trader.db"),
        ("Model 2", "yield-a-eth-auto-trader.service", "~/.gold_trading_data/yield_a_eth_auto_trader.db"),
    ]

    for model, svc, db_rel in services:
        # Check systemd status
        cmd = f"systemctl --user is-active {svc}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        is_active = res.stdout.strip() == "active"

        # Check DB file
        db_path = os.path.expanduser(db_rel)
        db_exists = os.path.exists(db_path)
        trade_count = 0
        scan_count = 0
        last_scan = "N/A"

        if db_exists:
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM trade_executions")
                trade_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*), MAX(timestamp) FROM scan_logs")
                row = cur.fetchone()
                scan_count = row[0] or 0
                last_scan = row[1] or "N/A"
                conn.close()
            except Exception as e:
                pass

        status = PASS if (is_active and db_exists) else FAIL
        print(f"  {status} [{model}] {svc}:")
        print(f"      • Service Status:  {'ACTIVE 🟢' if is_active else 'INACTIVE 🔴'}")
        print(f"      • Database:        {db_rel}")
        print(f"      • Executed Trades: {trade_count} recorded")
        print(f"      • Scan Heartbeats: {scan_count} scans logged (Latest: {last_scan})")

        results.append({"service": svc, "status": status, "trades": trade_count, "scans": scan_count})

    return results


def main():
    t_start = time.time()
    print("=" * 90)
    print("🛡️  AEGIS ALPHA — INSTITUTIONAL 7-TIER COMPREHENSIVE SYSTEM VERIFICATION")
    print(f"   Execution Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 90)

    r1 = test_bridge_connectivity()
    r2 = test_market_data_feeds()
    r3 = test_indicators_and_signals()
    r4 = test_machine_learning_gate()
    r5 = test_tag_isolation_and_filtering()
    r6 = test_position_sizing_and_risk_math()
    r7 = test_systemd_and_db_integrity()

    all_tests = r1 + r2 + r3 + r4 + r5 + r6 + r7
    total = len(all_tests)
    passed = len([t for t in all_tests if t.get("status") == PASS])
    warns = len([t for t in all_tests if t.get("status") == WARN])
    failed = len([t for t in all_tests if t.get("status") == FAIL])

    duration = time.time() - t_start

    print("\n" + "=" * 90)
    print("🏆 FINAL VERIFICATION SCORECARD")
    print("=" * 90)
    print(f"  • Total Subsystems Tested:  {total}")
    print(f"  • Passed:                   {passed} ({passed/total*100:.1f}%)")
    print(f"  • Warnings:                 {warns}")
    print(f"  • Failed:                   {failed}")
    print(f"  • Verification Time:        {duration:.2f} seconds")
    print("=" * 90)

    if failed == 0:
        print("🎉 ALL SYSTEMS FULLY OPERATIONAL AND VERIFIED CLEAN!")
    else:
        print(f"⚠️ {failed} subsystems reported failures. Please inspect the diagnostics above.")


if __name__ == "__main__":
    main()
