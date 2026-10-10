#!/usr/bin/env python3
"""
Forensic Audit of Option 3B (Fast-Track 15m Compounding Ladder)
Identifies and diagnoses:
1. Intrabar Lookahead Bias in Same-Bar Limit Fills (High before Low vs Low before High)
2. Premature / Optimistic TP triggering on entry bar
3. Worst-Case vs Best-Case Intrabar Pathing
4. Spread Expansion & Execution Slippage Impact
5. Live Broker Constraints (Max Lot Sizes, Margin Requirements, Slippage)
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.indicators import calc_ema, calc_rsi

def run_forensic_audit():
    print("=" * 80)
    print("🔬 RUNNING FORENSIC AUDIT ON OPTION 3B (XM MT5 15M CANDLES)...")
    broker = SocketBroker(timeout=60.0)
    broker.connect()
    candles = broker.get_rates("XAUUSD", "15m", 25000)
    n = len(candles)
    print(f"Loaded {n} candles.")

    highs = np.array([float(c["high"]) for c in candles])
    lows = np.array([float(c["low"]) for c in candles])
    closes = np.array([float(c["close"]) for c in candles])
    opens = np.array([float(c["open"]) for c in candles])
    dates = [c["date"] for c in candles]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    rsi = calc_rsi(closes.tolist(), 14)

    disc_usd = 0.80  # 8 pips
    spread = 0.35

    # We will test 3 distinct execution models:
    # Model A: Original (Optimistic - Checks TP before SL on red bars, uses entire entry bar high)
    # Model B: Conservative / Realistic (Evaluates TP/SL strictly AFTER the fill bar, k >= k_fill + 1)
    # Model C: Pessimistic Intrabar (If both SL and TP could be touched, SL ALWAYS assumed first)

    models = {
        "Model A (Original Backtest - Same-bar high credited)": {"wins": 0, "losses": 0, "scratches": 0},
        "Model B (Strict Realism - Evaluation starts bar k_fill + 1)": {"wins": 0, "losses": 0, "scratches": 0},
        "Model C (Pessimistic Intrabar - Adverse Stop First)": {"wins": 0, "losses": 0, "scratches": 0},
    }

    # Track how many trades were decided ON bar k_fill itself
    same_bar_wins = 0
    same_bar_losses = 0
    total_fills = 0

    for i in range(200, n - 36):
        dt = dates[i]
        hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
        if not (7 <= hour <= 19):
            continue

        e9, e21, e50, r = ema9[i], ema21[i], ema50[i], rsi[i]
        if None in (e9, e21, e50, r):
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        dir_ = None
        if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
            dir_ = "BUY"
        elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
            dir_ = "SELL"

        if not dir_:
            continue

        k_fill = i + 1
        limit_p = c - disc_usd if dir_ == "BUY" else c + disc_usd
        if dir_ == "BUY" and lows[k_fill] <= limit_p:
            fill_p = limit_p
        elif dir_ == "SELL" and highs[k_fill] >= limit_p:
            fill_p = limit_p
        else:
            continue

        total_fills += 1
        eff_entry = fill_p + spread if dir_ == "BUY" else fill_p
        tp = eff_entry + 2.00 if dir_ == "BUY" else eff_entry - 2.00 - spread
        sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + spread
        be_p = eff_entry + 1.00 if dir_ == "BUY" else eff_entry - 1.00

        # Check if same bar k_fill reached TP
        if dir_ == "BUY":
            if highs[k_fill] >= tp:
                same_bar_wins += 1
            if lows[k_fill] <= sl:
                same_bar_losses += 1
        else:
            if lows[k_fill] <= tp:
                same_bar_wins += 1
            if highs[k_fill] >= sl:
                same_bar_losses += 1

        # ---------------------------------------------------------------------
        # Simulate Model A (Original)
        # ---------------------------------------------------------------------
        res_a = "LOSS"
        is_be_a = False
        for k in range(k_fill, min(k_fill + 25, n)):
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be_a else sl
            if dir_ == "BUY":
                if c_k >= o_k:
                    if l_k <= curr_sl:
                        res_a = "SCRATCH" if is_be_a else "LOSS"
                        break
                    if h_k >= be_p: is_be_a = True
                    curr_sl = eff_entry if is_be_a else sl
                    if h_k >= tp: res_a = "WIN"; break
                else:
                    if h_k >= tp: res_a = "WIN"; break
                    if h_k >= be_p: is_be_a = True
                    curr_sl = eff_entry if is_be_a else sl
                    if l_k <= curr_sl:
                        res_a = "SCRATCH" if is_be_a else "LOSS"
                        break
            else:
                if c_k <= o_k:
                    if h_k >= curr_sl:
                        res_a = "SCRATCH" if is_be_a else "LOSS"
                        break
                    if l_k <= be_p: is_be_a = True
                    curr_sl = eff_entry if is_be_a else sl
                    if l_k <= tp: res_a = "WIN"; break
                else:
                    if l_k <= tp: res_a = "WIN"; break
                    if l_k <= be_p: is_be_a = True
                    curr_sl = eff_entry if is_be_a else sl
                    if h_k >= curr_sl:
                        res_a = "SCRATCH" if is_be_a else "LOSS"
                        break
        if res_a == "WIN": models["Model A (Original Backtest - Same-bar high credited)"]["wins"] += 1
        elif res_a == "SCRATCH": models["Model A (Original Backtest - Same-bar high credited)"]["scratches"] += 1
        else: models["Model A (Original Backtest - Same-bar high credited)"]["losses"] += 1

        # ---------------------------------------------------------------------
        # Simulate Model B (Strict Realism - Evaluation starts from next bar k_fill + 1)
        # On bar k_fill itself, only check if SL was breached after fill.
        # ---------------------------------------------------------------------
        res_b = None
        is_be_b = False
        # On bar k_fill: Did it drop further and hit SL on the entry bar?
        if dir_ == "BUY" and lows[k_fill] <= sl:
            res_b = "LOSS"
        elif dir_ == "SELL" and highs[k_fill] >= sl:
            res_b = "LOSS"

        if res_b is None:
            for k in range(k_fill + 1, min(k_fill + 25, n)):
                o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
                curr_sl = eff_entry if is_be_b else sl
                if dir_ == "BUY":
                    # Realistic path: check worst case (Low) before Best case (High)
                    if l_k <= curr_sl:
                        res_b = "SCRATCH" if is_be_b else "LOSS"
                        break
                    if h_k >= be_p: is_be_b = True
                    if h_k >= tp: res_b = "WIN"; break
                else:
                    if h_k >= curr_sl:
                        res_b = "SCRATCH" if is_be_b else "LOSS"
                        break
                    if l_k <= be_p: is_be_b = True
                    if l_k <= tp: res_b = "WIN"; break
            if res_b is None:
                res_b = "SCRATCH" if is_be_b else "LOSS"

        if res_b == "WIN": models["Model B (Strict Realism - Evaluation starts bar k_fill + 1)"]["wins"] += 1
        elif res_b == "SCRATCH": models["Model B (Strict Realism - Evaluation starts bar k_fill + 1)"]["scratches"] += 1
        else: models["Model B (Strict Realism - Evaluation starts bar k_fill + 1)"]["losses"] += 1

        # ---------------------------------------------------------------------
        # Simulate Model C (Pessimistic Intrabar: Always assumes SL hit before TP if both touched)
        # ---------------------------------------------------------------------
        res_c = None
        is_be_c = False
        for k in range(k_fill, min(k_fill + 25, n)):
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be_c else sl
            if dir_ == "BUY":
                # Check adverse move first
                if l_k <= curr_sl:
                    res_c = "SCRATCH" if is_be_c else "LOSS"
                    break
                if h_k >= be_p: is_be_c = True
                if h_k >= tp: res_c = "WIN"; break
            else:
                if h_k >= curr_sl:
                    res_c = "SCRATCH" if is_be_c else "LOSS"
                    break
                if l_k <= be_p: is_be_c = True
                if l_k <= tp: res_c = "WIN"; break
        if res_c is None:
            res_c = "SCRATCH" if is_be_c else "LOSS"

        if res_c == "WIN": models["Model C (Pessimistic Intrabar - Adverse Stop First)"]["wins"] += 1
        elif res_c == "SCRATCH": models["Model C (Pessimistic Intrabar - Adverse Stop First)"]["scratches"] += 1
        else: models["Model C (Pessimistic Intrabar - Adverse Stop First)"]["losses"] += 1

    print("\n" + "=" * 95)
    print("📊 FORENSIC AUDIT RESULTS: COMPARISON OF INTRABAR EXECUTION MODELS")
    print("=" * 95)
    print(f"{'Execution Model':<55} | {'Wins':<6} | {'Losses':<7} | {'Scratch':<8} | {'Win Rate':<9} | {'PF'}")
    print("-" * 95)
    for name, data in models.items():
        w, l, s = data["wins"], data["losses"], data["scratches"]
        decided = w + l
        wr = (w / decided * 100.0) if decided > 0 else 0.0
        # Win is 2.00, loss is 1.54
        gross_p = w * 2.00
        gross_l = l * 1.54
        pf = (gross_p / gross_l) if gross_l > 0 else 0.0
        print(f"{name:<55} | {w:<6} | {l:<7} | {s:<8} | {wr:>7.1f}% | {pf:>5.2f}")
    print("=" * 95)

    print(f"\n🔍 INTRABAR ANOMALY DIAGNOSIS:")
    print(f"Total Limit Order Fills           : {total_fills}")
    print(f"Trades marked Win on Fill Bar (k) : {same_bar_wins} ({same_bar_wins/total_fills*100:.1f}% of all fills!)")
    print(f"Trades marked Loss on Fill Bar (k): {same_bar_losses} ({same_bar_losses/total_fills*100:.1f}%)")

if __name__ == "__main__":
    run_forensic_audit()
