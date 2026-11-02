"""
Quantitative 6-Month Backtest of Donchian Breakout on 15-Minute Timeframe.
Evaluates 12,000 actual XM MT5 15m Gold candles (March 26, 2026 - September 28, 2026).
Compares:
1. Standard 20-bar Donchian (5-hour window)
2. Equivalent 80-bar Donchian (20-hour window, matching 1h strategy)
Under multiple execution policies (Deferred BE, Partial Scale-Out, Pure Runner).

Remediated with:
- Strict symmetric spread, slippage, and commission modeling
- Conservative adverse-first same-bar TP/SL barrier resolution
- Explicit separate TOUCH vs CLOSE trigger evaluation
- Full execution audit disclosures
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure src is in python path
sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.strategy_engine import calculate_lot_size
from gold_trader.core.unified_config import UnifiedRunConfig


def simulate_trade_execution(
    candles: List[Dict[str, Any]],
    donchian_period: int = 20,
    be_mode: str = "PARTIAL_SCALE",
    trigger_mode: str = "TOUCH",
    spread_usd: float = 0.35,
    slippage_usd: float = 0.05,
    commission_per_lot_usd: float = 7.00,
    adverse_first: bool = True,
    session_start_hour: int = 7,
    session_end_hour: int = 19,
    max_daily_trades: int = 4,
    initial_balance: float = 10000.0,
    risk_pct: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulate Donchian strategy on 15M candles with realistic broker execution mechanics:
    - Symmetric spread, slippage, and round-turn commission
    - Conservative adverse-first same-bar TP/SL resolution
    - Next-bar execution on completed candle confirmation for CLOSE mode
    - Gap-through-stop handling
    """
    n = len(candles)
    if n < 250:
        return {"error": f"Insufficient candles ({n} < 250)"}

    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # Precalculate indicators
    dc = calc_donchian(highs.tolist(), lows.tolist(), donchian_period)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    ema50 = calc_ema(closes.tolist(), 50)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi = calc_rsi(closes.tolist(), 14)

    equity = initial_balance
    peak_equity = initial_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    closed_trades = []
    daily_trades = {}
    active_until = -1
    loss_cooldown_until = -1

    for i in range(250, n - 48):
        if i <= active_until:
            continue

        dt_str = dates[i]
        day_str = dt_str.split(" ")[0]
        try:
            hour = int(dt_str.split(" ")[1].split(":")[0])
        except Exception:
            hour = 12

        # Session Filter (07:00 - 19:00 UTC)
        if not (session_start_hour <= hour <= session_end_hour):
            continue

        # Daily cap
        if daily_trades.get(day_str, 0) >= max_daily_trades:
            continue

        # Loss cooldown: 4 bars (60 minutes) on 15m
        if i <= loss_cooldown_until:
            continue

        d_up = dc["upper"][i - 1]
        d_low = dc["lower"][i - 1]
        st_dir = st["direction"][i]
        e200_val = ema200[i]
        e50_val = ema50[i]
        a = atr[i]
        r = rsi[i]

        if None in (d_up, d_low, st_dir, a, r):
            continue

        # Macro Trend confirmation
        macro_bullish = closes[i] > e200_val if e200_val is not None else (closes[i] > e50_val if e50_val is not None else True)
        macro_bearish = closes[i] < e200_val if e200_val is not None else (closes[i] < e50_val if e50_val is not None else True)

        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        triggered_side = None
        if trigger_mode.upper() == "TOUCH":
            if h >= d_up and st_dir == 1 and macro_bullish and r <= 75.0:
                triggered_side = "BUY"
            elif l <= d_low and st_dir == -1 and macro_bearish and r >= 25.0:
                triggered_side = "SELL"
        else:  # CLOSE mode: completed candle closed outside channel
            if c >= d_up and st_dir == 1 and macro_bullish and r <= 75.0:
                triggered_side = "BUY"
            elif c <= d_low and st_dir == -1 and macro_bearish and r >= 25.0:
                triggered_side = "SELL"

        if not triggered_side:
            continue

        daily_trades[day_str] = daily_trades.get(day_str, 0) + 1

        # Stop loss distance: min $6.00 on 15m
        stop_dist = max(round(1.5 * a, 2), 6.00)
        lots = calculate_lot_size(
            account_balance=equity,
            risk_pct=risk_pct,
            entry_price=c,
            stop_loss_price=c - stop_dist if triggered_side == "BUY" else c + stop_dist,
        )
        lots = max(0.01, min(lots, 5.0))

        # Execution pricing with symmetric spread and slippage
        # In CLOSE mode, order is submitted at next bar's open
        if trigger_mode.upper() == "CLOSE" and i + 1 < n:
            base_entry = opens[i + 1]
            eval_start_idx = i + 1
        else:
            base_entry = d_up if (trigger_mode.upper() == "TOUCH" and triggered_side == "BUY") else (
                d_low if (trigger_mode.upper() == "TOUCH" and triggered_side == "SELL") else c
            )
            eval_start_idx = i + 1

        if triggered_side == "BUY":
            # Buy at Ask = base + spread + slippage
            entry_price = round(base_entry + spread_usd + slippage_usd, 2)
            sl_price = round(entry_price - stop_dist, 2)
            tp1_price = round(entry_price + (1.2 * stop_dist), 2)
            tp2_price = round(entry_price + (2.0 * stop_dist), 2)
        else:  # SELL
            # Sell at Bid = base - slippage
            entry_price = round(base_entry - slippage_usd, 2)
            sl_price = round(entry_price + stop_dist, 2)
            tp1_price = round(entry_price - (1.2 * stop_dist), 2)
            tp2_price = round(entry_price - (2.0 * stop_dist), 2)

        be_active = False
        half_closed = False
        realized_pnl = 0.0
        remaining_lots = lots
        trade_closed = False
        exit_reason = "TIMEOUT"
        exit_price = entry_price
        bars_held = 0

        # BE trigger distance
        if be_mode in ("DEFERRED_BE", "PARTIAL_SCALE"):
            be_trigger = max(round(stop_dist * 1.0, 2), 6.00)
        else:
            be_trigger = 999999.0

        # Max hold: 96 bars = 24 hours on 15m
        for k in range(eval_start_idx, min(eval_start_idx + 96, n)):
            bars_held = k - eval_start_idx
            bar_o = opens[k]
            bar_h = highs[k]
            bar_l = lows[k]

            if triggered_side == "BUY":
                # Check adverse-first: Did bar hit Stop Loss or gap down below SL?
                hit_sl = False
                sl_exit_price = sl_price
                if bar_o <= sl_price:
                    # Gapped through stop
                    hit_sl = True
                    sl_exit_price = round(bar_o - slippage_usd, 2)
                    exit_reason = "GAP_STOP"
                elif bar_l <= sl_price:
                    hit_sl = True
                    sl_exit_price = round(sl_price - slippage_usd, 2)
                    exit_reason = "BE_SCRATCH" if be_active and sl_price >= entry_price else "STOP_LOSS"

                if adverse_first and hit_sl:
                    trade_closed = True
                    exit_price = sl_exit_price
                    break

                # If stop not hit, check favorable progression (BE / TP)
                if not be_active and (bar_h - entry_price) >= be_trigger:
                    be_active = True
                    sl_price = round(entry_price + spread_usd + 0.50, 2)

                if be_mode == "PARTIAL_SCALE" and not half_closed and bar_h >= tp1_price:
                    half_closed = True
                    part_lots = round(lots * 0.5, 2)
                    tp_realized = (tp1_price - slippage_usd - entry_price) * part_lots * 100.0
                    realized_pnl += tp_realized
                    remaining_lots = max(0.01, round(lots - part_lots, 2))
                    be_active = True
                    sl_price = round(entry_price + spread_usd + 0.50, 2)

                target_price = tp2_price
                if bar_h >= target_price:
                    trade_closed = True
                    exit_price = round(target_price - slippage_usd, 2)
                    exit_reason = "TAKE_PROFIT"
                    break

                # If non-adverse first, evaluate stop loss after TP checks
                if not adverse_first and hit_sl:
                    trade_closed = True
                    exit_price = sl_exit_price
                    break

            else:  # SELL
                # Effective buyback quote = price + spread
                eff_o = bar_o + spread_usd
                eff_h = bar_h + spread_usd
                eff_l = bar_l + spread_usd

                hit_sl = False
                sl_exit_price = sl_price
                if eff_o >= sl_price:
                    # Gapped through stop
                    hit_sl = True
                    sl_exit_price = round(eff_o + slippage_usd, 2)
                    exit_reason = "GAP_STOP"
                elif eff_h >= sl_price:
                    hit_sl = True
                    sl_exit_price = round(sl_price + slippage_usd, 2)
                    exit_reason = "BE_SCRATCH" if be_active and sl_price <= entry_price else "STOP_LOSS"

                if adverse_first and hit_sl:
                    trade_closed = True
                    exit_price = sl_exit_price
                    break

                if not be_active and (entry_price - bar_l) >= be_trigger:
                    be_active = True
                    sl_price = round(entry_price - spread_usd - 0.50, 2)

                if be_mode == "PARTIAL_SCALE" and not half_closed and eff_l <= tp1_price:
                    half_closed = True
                    part_lots = round(lots * 0.5, 2)
                    tp_realized = (entry_price - (tp1_price + slippage_usd)) * part_lots * 100.0
                    realized_pnl += tp_realized
                    remaining_lots = max(0.01, round(lots - part_lots, 2))
                    be_active = True
                    sl_price = round(entry_price - spread_usd - 0.50, 2)

                target_price = tp2_price
                if eff_l <= target_price:
                    trade_closed = True
                    exit_price = round(target_price + slippage_usd, 2)
                    exit_reason = "TAKE_PROFIT"
                    break

                if not adverse_first and hit_sl:
                    trade_closed = True
                    exit_price = sl_exit_price
                    break

        # Calculate remaining PnL
        if not trade_closed:
            final_bar_idx = min(eval_start_idx + 96 - 1, n - 1)
            exit_price = closes[final_bar_idx] - slippage_usd if triggered_side == "BUY" else (closes[final_bar_idx] + spread_usd + slippage_usd)
            exit_reason = "TIMEOUT"

        diff = (exit_price - entry_price) if triggered_side == "BUY" else (entry_price - exit_price)
        realized_pnl += diff * remaining_lots * 100.0

        # Deduct round-turn commission
        commission_cost = round(lots * commission_per_lot_usd, 2)
        realized_pnl = round(realized_pnl - commission_cost, 2)

        equity = max(100.0, round(equity + realized_pnl, 2))
        if equity > peak_equity:
            peak_equity = equity
        dd_usd = peak_equity - equity
        dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd
            max_dd_pct = dd_pct

        active_until = eval_start_idx + bars_held
        if realized_pnl < -0.5:
            loss_cooldown_until = active_until + 4  # 4 bars cooldown

        closed_trades.append({
            "side": triggered_side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "lots": lots,
            "commission": commission_cost,
            "pnl": round(realized_pnl, 2),
            "reason": exit_reason,
        })

    # Calculate metrics
    num_trades = len(closed_trades)
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] < 0]
    scratches = [t for t in closed_trades if abs(t["pnl"]) <= 0.5]
    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    net_pnl = equity - initial_balance
    return_pct = (net_pnl / initial_balance) * 100.0
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    run_cfg = UnifiedRunConfig(
        symbol="XAUUSD",
        timeframe="15m",
        strategy_mode="DONCHIAN_BREAKOUT",
        trigger_mode=trigger_mode,
        execution_mode="BACKTEST",
        donchian_period=donchian_period,
        rr_multiplier=1.20,
        fixed_risk_usd=75.44,
        risk_pct_per_trade=risk_pct,
        spread_usd=spread_usd,
        slippage_usd=slippage_usd,
        commission_per_lot_usd=commission_per_lot_usd,
        session_start_hour_utc=session_start_hour,
        session_end_hour_utc=session_end_hour,
        max_daily_trades=max_daily_trades,
    )

    return {
        "trades": num_trades,
        "wins": len(wins),
        "losses": len(losses),
        "scratches": len(scratches),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_pnl": round(net_pnl, 2),
        "return_pct": round(return_pct, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "final_balance": round(equity, 2),
        "closed_trades": closed_trades,
        "config_hash": run_cfg.config_hash,
        "git_sha": run_cfg.git_sha,
        "config": run_cfg.to_dict(),
    }


def run_15m_donchian_study():
    print("=" * 80)
    print("Fetching 12,000 15-minute Gold candles directly from XM MT5 Broker...")
    broker = SocketBroker()
    candles = broker.get_rates("GOLD", "15m", 12000)
    if not candles or len(candles) < 1000:
        print(f"Error: only received {len(candles) if candles else 0} candles.")
        return

    n = len(candles)
    start_dt = candles[0]["date"]
    end_dt = candles[-1]["date"]
    print(f"Loaded {n} 15m candles from {start_dt} to {end_dt} (Full 6 Months)")
    print("Execution Model: Symmetric Spread ($0.35), Slippage ($0.05), Commission ($7.00/lot), Adverse-First Resolution")
    print("=" * 80)

    test_variants = [
        ("A. 20-Bar Donchian (5h) + Deferred BE (1.0R) [TOUCH]", 20, "DEFERRED_BE", "TOUCH"),
        ("B. 20-Bar Donchian (5h) + Partial Scale-Out (50% @ 1.2R) [TOUCH]", 20, "PARTIAL_SCALE", "TOUCH"),
        ("C. 20-Bar Donchian (5h) + Pure Runner (2.0R) [TOUCH]", 20, "NO_BE", "TOUCH"),
        ("D. 80-Bar Donchian (20h) + Deferred BE (1.0R) [TOUCH]", 80, "DEFERRED_BE", "TOUCH"),
        ("E. 80-Bar Donchian (20h) + Partial Scale-Out (50% @ 1.2R) [TOUCH]", 80, "PARTIAL_SCALE", "TOUCH"),
        ("F. 80-Bar Donchian (20h) + Pure Runner (2.0R) [TOUCH]", 80, "NO_BE", "TOUCH"),
        ("G. 20-Bar Donchian (5h) + Partial Scale-Out [CLOSE]", 20, "PARTIAL_SCALE", "CLOSE"),
    ]

    summary_table = []
    for label, period, be_mode, trig_mode in test_variants:
        res = simulate_trade_execution(
            candles=candles,
            donchian_period=period,
            be_mode=be_mode,
            trigger_mode=trig_mode,
            spread_usd=0.35,
            slippage_usd=0.05,
            commission_per_lot_usd=7.00,
            adverse_first=True,
        )
        res["variant"] = label
        summary_table.append(res)

    # Print summary
    print("\n" + "=" * 90)
    print("🏆 6-MONTH DONCHIAN BREAKOUT BACKTEST RESULTS ON 15-MINUTE TIMEFRAME")
    print(f"Data: 12,000 candles from {start_dt} to {end_dt} | $10,000 Initial Capital | 1% Risk/Trade")
    print("=" * 90)
    fmt = "{:<55} | {:<6} | {:<8} | {:<5} | {:<10} | {:<8} | {:<6}"
    print(fmt.format("Configuration", "Trades", "Win Rate", "P.F.", "Net P&L", "Payoff", "Max DD"))
    print("-" * 90)
    for row in summary_table:
        sign = "+" if row["net_pnl"] >= 0 else ""
        print(fmt.format(
            row["variant"][:55],
            f"{row['trades']}",
            f"{row['win_rate']}%",
            f"{row['profit_factor']:.2f}",
            f"{sign}${row['net_pnl']:.2f}",
            f"{row['payoff_ratio']:.2f}:1",
            f"{row['max_dd_pct']:.1f}%",
        ))
    print("=" * 90)

    # Save to JSON
    out_file = os.path.expanduser("~/.gold_trading_data/donchian_15m_backtest_results.json")
    with open(out_file, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"Saved full results to {out_file}")


if __name__ == "__main__":
    run_15m_donchian_study()
