#!/usr/bin/env python3
"""
Order Flow Enhanced Strategy Backtester for Aegis Alpha (UAT).

Directly fetches historical 1H candles with institutional Taker Buy/Sell Volume
from Binance API (BTCUSDT & ETHUSDT) and evaluates:
1. BASELINE: Current Production Flat Risk Touch Strategy (Donchian 10 + SuperTrend + 50 EMA + Rejection Wick)
2. ORDER_FLOW_DELTA: Baseline + Real-time Delta Confirmation (Delta > 0 for BUY, < 0 for SELL)
3. ORDER_FLOW_FULL: Delta Confirmation + 5-Bar CVD Momentum + Microstructure Exhaustion Runner Exit

Metrics Compared:
- Total Trades & Trade Frequency
- Win Rate %
- False Breakouts Avoided
- Net Realized P&L ($) & Total R
- Profit Factor (PF)
- Maximum Drawdown ($ and %)
- Average Win / Average Loss Ratio
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gold_trader.core.indicators import (
    calc_adx,
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)


def fetch_binance_candles_with_order_flow(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    total_candles: int = 3500,
) -> list[dict[str, Any]]:
    """Fetch historical klines containing real Taker Buy Volume from Binance."""
    print(f"📥 Fetching {total_candles} {interval} candles with Taker Order Flow for {symbol}...")
    all_raw: list[list] = []
    end_time = None

    while len(all_raw) < total_candles:
        limit = min(1000, total_candles - len(all_raw))
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time:
            url += f"&endTime={end_time}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        all_raw = batch + all_raw if end_time else batch
        end_time = batch[0][0] - 1
        time.sleep(0.04)

    # Clean, parse, and calculate Delta & CVD
    seen = set()
    cleaned = []
    for c in all_raw:
        ts = int(c[0])
        if ts in seen:
            continue
        seen.add(ts)

        open_p = float(c[1])
        high_p = float(c[2])
        low_p = float(c[3])
        close_p = float(c[4])
        tot_vol = float(c[5])
        buy_vol = float(c[9])  # Field 9: Taker buy base asset volume
        sell_vol = max(0.0, tot_vol - buy_vol)
        delta = buy_vol - sell_vol
        delta_pct = (delta / tot_vol * 100.0) if tot_vol > 0 else 0.0

        cleaned.append({
            "ts": ts,
            "date": datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "total_volume": tot_vol,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "delta": delta,
            "delta_pct": delta_pct,
        })

    cleaned.sort(key=lambda x: x["ts"])

    # Calculate rolling Cumulative Volume Delta (CVD)
    running_cvd = 0.0
    for c in cleaned:
        running_cvd += c["delta"]
        c["cvd"] = running_cvd

    print(f"✅ Loaded {len(cleaned)} candles: {cleaned[0]['date']} to {cleaned[-1]['date']}\n")
    return cleaned


def run_backtest_simulation(
    candles: list[dict[str, Any]],
    symbol: str = "BTCUSDT",
    strategy_mode: str = "BASELINE",  # "BASELINE", "OF_DELTA", "OF_FULL"
    fixed_risk_usd: float = 75.44,
    starting_balance: float = 1000.0,
    spread_usd: float = 20.0,
    min_sl_usd: float = 500.0,
) -> dict[str, Any]:
    """Execute historical simulation matching Aegis Alpha production execution."""
    n = len(candles)
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    opens = [c["open"] for c in candles]
    deltas = [c["delta"] for c in candles]
    delta_pcts = [c["delta_pct"] for c in candles]
    cvds = [c["cvd"] for c in candles]

    # Technical Indicators
    dc = calc_donchian(highs, lows, 10)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    ema50 = calc_ema(closes, 50)
    atr14 = calc_atr(highs, lows, closes, 14)
    rsi14 = calc_rsi(closes, 14)
    adx14 = calc_adx(highs, lows, closes, 14)

    balance = starting_balance
    peak_balance = balance
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    trades: list[dict[str, Any]] = []
    wins = 0
    losses = 0
    scratches = 0
    false_breakouts_avoided = 0
    active_until = -1

    for i in range(50, n - 48):
        if i <= active_until:
            continue

        d_up = dc["upper"][i - 1]
        d_dn = dc["lower"][i - 1]
        st_dir = st["direction"][i]
        e_trend = ema50[i]
        atr_val = atr14[i]
        r = rsi14[i]
        adx_val = adx14["adx"][i] if adx14 and "adx" in adx14 else None

        if None in (d_up, d_dn, st_dir, atr_val, r, e_trend):
            continue

        # Production ADX Guard for Ethereum: reject horizontal chop
        if "ETH" in symbol and adx_val is not None and adx_val < 22.0:
            continue

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        # 1. Base Strategy Signals (1H Donchian 10 Touch + SuperTrend + 50 EMA)
        bull_trend = (st_dir == 1) and (c > e_trend) and (r <= 75.0)
        bear_trend = (st_dir == -1) and (c < e_trend) and (r >= 25.0)

        raw_direction = None
        entry_price = c

        if (h >= d_up or c >= d_up) and bull_trend:
            raw_direction = "BUY"
            entry_price = max(o, d_up) if o <= d_up else o
        elif (l <= d_dn or c <= d_dn) and bear_trend:
            raw_direction = "SELL"
            entry_price = min(o, d_dn) if o >= d_dn else o

        if not raw_direction:
            continue

        # 2. Production Rejection Wick Rule
        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        min_body_threshold = 0.15 * atr_val

        if raw_direction == "BUY" and (upper_wick > 1.8 * max(body, min_body_threshold) or c < o) or raw_direction == "SELL" and (lower_wick > 1.8 * max(body, min_body_threshold) or c > o):
            continue

        # 3. Order Flow Microstructure Filtering
        if strategy_mode in ("OF_DELTA", "OF_FULL"):
            bar_delta = deltas[i]
            bar_delta_pct = delta_pcts[i]

            # Delta Confirmation Guard:
            # BUY requires positive Delta (aggressive buyers taking control)
            # SELL requires negative Delta (aggressive sellers taking control)
            if raw_direction == "BUY" and (bar_delta <= 0 or bar_delta_pct < 2.0):
                false_breakouts_avoided += 1
                continue
            if raw_direction == "SELL" and (bar_delta >= 0 or bar_delta_pct > -2.0):
                false_breakouts_avoided += 1
                continue

            # In OF_FULL, also check 5-bar CVD momentum alignment
            if strategy_mode == "OF_FULL" and i >= 5:
                cvd_slope = cvds[i] - cvds[i - 5]
                if raw_direction == "BUY" and cvd_slope <= 0:
                    false_breakouts_avoided += 1
                    continue
                if raw_direction == "SELL" and cvd_slope >= 0:
                    false_breakouts_avoided += 1
                    continue

        direction = raw_direction

        # 4. Stop Loss & Position Sizing
        sl_pts = max(round(1.5 * atr_val, 2), min_sl_usd)
        lots = round(fixed_risk_usd / sl_pts, 2)
        lots = max(lots, 0.01)

        eff_entry = entry_price + spread_usd if direction == "BUY" else entry_price
        sl_price = eff_entry - sl_pts if direction == "BUY" else eff_entry + sl_pts + spread_usd

        # 5. Two-Stage Execution Targets
        t1_pts = round(sl_pts * 1.0, 2)
        t1_price = eff_entry + t1_pts if direction == "BUY" else eff_entry - t1_pts - spread_usd
        t2_pts = round(sl_pts * 2.50, 2)
        t2_price = eff_entry + t2_pts if direction == "BUY" else eff_entry - t2_pts - spread_usd

        stage1_hit = False
        stage1_lots = round(lots * 0.5, 2) if lots >= 0.02 else 0.0
        runner_lots = round(lots - stage1_lots, 2)
        realized_stage1 = 0.0
        curr_runner_sl = sl_price
        peak_p = eff_entry
        trade_pnl = 0.0
        outcome = "TIME_EXIT"
        exit_reason = "TIME_EXPIRY"
        be_buffer = 30.0 if "BTC" in symbol else 2.0

        for k in range(i + 1, min(i + 72, n)):
            h_k, l_k = highs[k], lows[k]
            c_k = closes[k]

            if direction == "BUY":
                peak_p = max(peak_p, h_k)
                # Check stop loss hit
                if l_k <= curr_runner_sl:
                    if not stage1_hit:
                        outcome = "LOSS"
                        exit_reason = "STOP_LOSS"
                        trade_pnl = -fixed_risk_usd
                    else:
                        runner_pnl = (curr_runner_sl - eff_entry) * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN" if trade_pnl > 1.0 else "SCRATCH"
                        exit_reason = "BE_RUNNER" if curr_runner_sl <= eff_entry + be_buffer else "TRAILED_STOP"
                    active_until = k
                    break

                # Stage 1 Target hit
                if not stage1_hit and h_k >= t1_price:
                    stage1_hit = True
                    if stage1_lots > 0:
                        realized_stage1 = t1_pts * stage1_lots
                    curr_runner_sl = max(curr_runner_sl, eff_entry + be_buffer)

                # Order Flow Early Runner Exhaustion Exit (OF_FULL only)
                if strategy_mode == "OF_FULL" and stage1_hit:
                    # If price is at >= +1.5R, but this candle had heavy negative delta (exhaustion absorption)
                    r_excursion = (c_k - eff_entry) / sl_pts
                    if r_excursion >= 1.50 and deltas[k] < 0 and delta_pcts[k] <= -15.0:
                        runner_pnl = (c_k - eff_entry) * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN"
                        exit_reason = "OF_EXHAUSTION_EXIT"
                        active_until = k
                        break

                # Trailing runner update
                if stage1_hit:
                    runner_buffer = 1.0 * atr_val
                    trailed_sl = round(peak_p - runner_buffer, 2)
                    if trailed_sl >= curr_runner_sl + be_buffer and trailed_sl >= eff_entry + be_buffer:
                        curr_runner_sl = trailed_sl

                    if h_k >= t2_price:
                        runner_pnl = t2_pts * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN"
                        exit_reason = "TARGET_2_FULL"
                        active_until = k
                        break

            else:  # SELL
                peak_p = min(peak_p, l_k)
                if h_k >= curr_runner_sl:
                    if not stage1_hit:
                        outcome = "LOSS"
                        exit_reason = "STOP_LOSS"
                        trade_pnl = -fixed_risk_usd
                    else:
                        runner_pnl = (eff_entry - curr_runner_sl) * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN" if trade_pnl > 1.0 else "SCRATCH"
                        exit_reason = "BE_RUNNER" if curr_runner_sl >= eff_entry - be_buffer else "TRAILED_STOP"
                    active_until = k
                    break

                if not stage1_hit and l_k <= t1_price:
                    stage1_hit = True
                    if stage1_lots > 0:
                        realized_stage1 = t1_pts * stage1_lots
                    curr_runner_sl = min(curr_runner_sl, eff_entry - be_buffer)

                if strategy_mode == "OF_FULL" and stage1_hit:
                    r_excursion = (eff_entry - c_k) / sl_pts
                    if r_excursion >= 1.50 and deltas[k] > 0 and delta_pcts[k] >= 15.0:
                        runner_pnl = (eff_entry - c_k) * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN"
                        exit_reason = "OF_EXHAUSTION_EXIT"
                        active_until = k
                        break

                if stage1_hit:
                    runner_buffer = 1.0 * atr_val
                    trailed_sl = round(peak_p + runner_buffer, 2)
                    if trailed_sl <= curr_runner_sl - be_buffer and trailed_sl <= eff_entry - be_buffer:
                        curr_runner_sl = trailed_sl

                    if l_k <= t2_price:
                        runner_pnl = t2_pts * runner_lots
                        trade_pnl = realized_stage1 + runner_pnl
                        outcome = "WIN"
                        exit_reason = "TARGET_2_FULL"
                        active_until = k
                        break

        # Tally statistics
        balance += trade_pnl
        peak_balance = max(peak_balance, balance)
        dd = peak_balance - balance
        dd_pct = (dd / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        else:
            scratches += 1

        trades.append({
            "entry_date": candles[i]["date"],
            "direction": direction,
            "outcome": outcome,
            "exit_reason": exit_reason,
            "pnl": round(trade_pnl, 2),
            "pnl_r": round(trade_pnl / fixed_risk_usd, 2),
            "balance": round(balance, 2),
        })

    # Summary metrics
    tot_trades = len(trades)
    win_rate = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    gross_wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 999.0
    net_pnl = balance - starting_balance
    total_r = sum(t["pnl_r"] for t in trades)

    return {
        "mode": strategy_mode,
        "symbol": symbol,
        "total_trades": tot_trades,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": round(win_rate, 2),
        "false_breakouts_avoided": false_breakouts_avoided,
        "net_pnl": round(net_pnl, 2),
        "total_r": round(total_r, 2),
        "ending_balance": round(balance, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_usd": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "avg_win": round(gross_wins / max(1, wins), 2),
        "avg_loss": round(gross_losses / max(1, losses), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Order Flow Strategy Backtester (UAT)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (BTCUSDT, ETHUSDT)")
    parser.add_argument("--candles", type=int, default=3500, help="Number of 1H candles to fetch")
    args = parser.parse_args()

    candles = fetch_binance_candles_with_order_flow(symbol=args.symbol, interval="1h", total_candles=args.candles)

    min_sl = 500.0 if "BTC" in args.symbol else 20.0
    spread = 20.0 if "BTC" in args.symbol else 1.50

    modes = [
        ("BASELINE", "1. Current Baseline (1H Donchian + SuperTrend + 50 EMA)"),
        ("OF_DELTA", "2. + Order Flow Delta Filter (Reject opposing volume fake-outs)"),
        ("OF_FULL",  "3. + Full Order Flow (Delta Filter + 5-bar CVD Slope + Exhaustion Exit)"),
    ]

    results = []
    for mode_code, label in modes:
        res = run_backtest_simulation(
            candles=candles,
            symbol=args.symbol,
            strategy_mode=mode_code,
            fixed_risk_usd=75.44,
            starting_balance=1000.0,
            spread_usd=spread,
            min_sl_usd=min_sl,
        )
        res["label"] = label
        results.append(res)

    # Print Comparative Matrix
    print("=" * 115)
    print(f"📊 ORDER FLOW BACKTEST COMPARISON: {args.symbol} (Dataset: {len(candles)} 1H Candles | {candles[0]['date']} to {candles[-1]['date']})")
    print("   Execution Model: Flat Dollar Risk ($75.44 / trade) on $1,000 Starting Seed | Two-Stage Trailing Runners")
    print("=" * 115)
    print(f"{'Strategy Configuration':<42} | {'Trades':<6} | {'Win %':<6} | {'Net P&L ($)':<12} | {'Total R':<8} | {'Profit Factor':<8} | {'Max DD (%)':<10} | {'Avoided Traps':<12}")
    print("─" * 115)

    for r in results:
        pnl_str = f"+${r['net_pnl']:,.2f}" if r['net_pnl'] >= 0 else f"-${abs(r['net_pnl']):,.2f}"
        r_str = f"+{r['total_r']:.1f}R" if r['total_r'] >= 0 else f"{r['total_r']:.1f}R"
        print(f"{r['label']:<42} | {r['total_trades']:<6} | {r['win_rate']:>5.1f}% | {pnl_str:>12} | {r_str:>8} | {r['profit_factor']:>8.2f} | {r['max_drawdown_pct']:>9.2f}% | {r['false_breakouts_avoided']:>12}")

    print("=" * 115)


if __name__ == "__main__":
    main()
