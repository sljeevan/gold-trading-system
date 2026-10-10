"""
Comprehensive 6-Month Backtest of Strategy Combinations for Gold (XAUUSD).
Evaluates 2,900+ 1-hour candles of actual historical Gold price action (March 2026 - September 2026).
Simulates realistic $0.35 spread, 1.0% ($100) risk per trade on $10,000 initial capital.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is in path
sys.path.insert(0, "/home/mcpuser/MCP/gold-trading-system/src")

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.strategy_engine import calculate_lot_size


def run_combinations_backtest():
    print("=" * 90)
    print("🚀 QUANTITATIVE 6-MONTH GOLD STRATEGY COMBINATIONS TOURNAMENT")
    print("=" * 90)

    # 1. Fetch Data
    candles = fetch_gold_candles("XAUUSD", period="6mo", interval="1h")
    if not candles or len(candles) < 500:
        print(f"Error: Only {len(candles) if candles else 0} candles retrieved.")
        return

    n = len(candles)
    start_dt = candles[0]["date"]
    end_dt = candles[-1]["date"]
    print(f"Dataset: {n} 1-Hour candles | Period: {start_dt} to {end_dt}")
    print(f"Starting Capital: $10,000.00 | Risk: 1.0% ($100) | Spread: $0.35/oz\n")

    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    opens = np.array([float(c["open"]) for c in candles], dtype=float)
    dates = [c["date"] for c in candles]

    # 2. Precalculate Indicator Sets
    print("Precalculating indicator arrays across multiple parameter grids...")
    donchian_dict = {
        10: calc_donchian(highs.tolist(), lows.tolist(), 10),
        15: calc_donchian(highs.tolist(), lows.tolist(), 15),
        20: calc_donchian(highs.tolist(), lows.tolist(), 20),
        30: calc_donchian(highs.tolist(), lows.tolist(), 30),
        40: calc_donchian(highs.tolist(), lows.tolist(), 40),
    }

    st_3 = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    st_2 = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 2.0)

    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema100 = calc_ema(closes.tolist(), 100)
    ema200 = calc_ema(closes.tolist(), 200)

    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi14 = calc_rsi(closes.tolist(), 14)

    # MACD (12, 26, 9)
    ema12 = calc_ema(closes.tolist(), 12)
    ema26 = calc_ema(closes.tolist(), 26)
    macd_line = [
        (ema12[i] - ema26[i]) if (ema12[i] is not None and ema26[i] is not None) else None
        for i in range(n)
    ]
    valid_macd = [m for m in macd_line if m is not None]
    macd_sig_valid = calc_ema(valid_macd, 9)
    prefix_len = len(macd_line) - len(valid_macd)
    macd_signal = [None] * prefix_len + macd_sig_valid

    INITIAL_CAPITAL = 10000.0
    SPREAD = 0.35

    def simulate_strategy(
        name: str,
        donch_period: int,
        entry_trigger: str,      # "TOUCH" or "CLOSE"
        trend_filter: str,       # "ST3_EMA200", "ST3_EMA50", "ST2_EMA200", "ST3_ONLY", "MACD_ST3", "EMA50_200"
        atr_sl_mult: float,      # 1.0, 1.5, 2.0
        min_sl_dollars: float,   # 6.0, 8.0, 10.0
        exit_policy: str,        # "FIXED_1.5R", "FIXED_2.0R", "FIXED_2.5R", "FIXED_3.0R", "DEFERRED_BE_2.0R",
                                 # "PARTIAL_1.2R_2.0R", "PARTIAL_1.5R_2.5R", "PARTIAL_1.5R_3.0R", "ST_TRAILING", "MID_BAND_EXIT"
        session_mode: str,       # "LONDON_NY" (07-19), "OVERLAP" (12-17), "24H" (all)
        use_pullback_limit: bool = False,
        pullback_discount: float = 0.80, # $0.80 (8 pips)
        max_daily_trades: int = 4,
        loss_cooldown_bars: int = 1,
    ) -> Dict[str, Any]:
        dc = donchian_dict[donch_period]
        equity = INITIAL_CAPITAL
        peak_equity = INITIAL_CAPITAL
        max_dd_usd = 0.0
        max_dd_pct = 0.0

        closed_trades: List[Dict[str, Any]] = []
        daily_trades: Dict[str, int] = {}
        active_until = -1
        loss_cooldown_until = -1

        for i in range(210, n - 48):
            if i <= active_until or i <= loss_cooldown_until:
                continue

            dt_str = dates[i]
            day_str = dt_str.split("T" if "T" in dt_str else " ")[0]
            try:
                hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])
            except Exception:
                hour = 12

            # Session Filter
            if session_mode == "LONDON_NY" and not (7 <= hour <= 19):
                continue
            elif session_mode == "OVERLAP" and not (12 <= hour <= 16):
                continue

            # Daily trades cap
            if daily_trades.get(day_str, 0) >= max_daily_trades:
                continue

            c, h, l, o = closes[i], highs[i], lows[i], opens[i]
            d_up = dc["upper"][i - 1]
            d_low = dc["lower"][i - 1]
            d_mid = dc["middle"][i - 1]
            a = atr14[i]
            r = rsi14[i]

            if None in (d_up, d_low, d_mid, a, r):
                continue

            # Trend Filter Evaluation
            st_obj = st_2 if "ST2" in trend_filter else st_3
            st_dir = st_obj["direction"][i]
            e200 = ema200[i]
            e50 = ema50[i]
            macd_val = macd_line[i]
            macd_sig = macd_signal[i]

            bull_trend = False
            bear_trend = False

            if trend_filter == "ST3_EMA200" or trend_filter == "ST2_EMA200":
                bull_trend = (st_dir == 1) and (e200 is not None and c > e200) and (r <= 75.0)
                bear_trend = (st_dir == -1) and (e200 is not None and c < e200) and (r >= 25.0)
            elif trend_filter == "ST3_EMA50":
                bull_trend = (st_dir == 1) and (e50 is not None and c > e50) and (r <= 75.0)
                bear_trend = (st_dir == -1) and (e50 is not None and c < e50) and (r >= 25.0)
            elif trend_filter == "ST3_ONLY":
                bull_trend = (st_dir == 1) and (r <= 75.0)
                bear_trend = (st_dir == -1) and (r >= 25.0)
            elif trend_filter == "MACD_ST3":
                macd_bull = (macd_val is not None and macd_sig is not None and macd_val > macd_sig)
                macd_bear = (macd_val is not None and macd_sig is not None and macd_val < macd_sig)
                bull_trend = (st_dir == 1) and macd_bull and (r <= 75.0)
                bear_trend = (st_dir == -1) and macd_bear and (r >= 25.0)
            elif trend_filter == "EMA50_200":
                bull_trend = (st_dir == 1) and (e50 is not None and e200 is not None and e50 > e200) and (r <= 75.0)
                bear_trend = (st_dir == -1) and (e50 is not None and e200 is not None and e50 < e200) and (r >= 25.0)

            # Trigger Evaluation
            side: Optional[str] = None
            if entry_trigger == "TOUCH":
                if (h >= d_up or c >= d_up) and bull_trend:
                    side = "BUY"
                elif (l <= d_low or c <= d_low) and bear_trend:
                    side = "SELL"
            elif entry_trigger == "CLOSE":
                if c > d_up and bull_trend and (c > o):
                    side = "BUY"
                elif c < d_low and bear_trend and (c < o):
                    side = "SELL"

            if not side:
                continue

            # Calculate SL and TP parameters
            stop_dist = max(round(atr_sl_mult * a, 2), min_sl_dollars)
            lots = calculate_lot_size(
                account_balance=equity,
                risk_pct=1.0,
                entry_price=c,
                stop_loss_price=c - stop_dist if side == "BUY" else c + stop_dist,
            )
            lots = max(0.01, min(lots, 5.0))

            # Entry & Limits
            if use_pullback_limit:
                entry_order_price = (c - pullback_discount) if side == "BUY" else (c + pullback_discount)
                # Check if next candle fills limit
                next_bar_h = highs[i + 1] if i + 1 < n else c
                next_bar_l = lows[i + 1] if i + 1 < n else c
                if side == "BUY" and next_bar_l > entry_order_price:
                    continue  # Unfilled limit
                elif side == "SELL" and next_bar_h < entry_order_price:
                    continue  # Unfilled limit
                entry_price = entry_order_price + (SPREAD if side == "BUY" else 0.0)
                sim_start_idx = i + 2
            else:
                entry_price = (c + SPREAD) if side == "BUY" else c
                sim_start_idx = i + 1

            daily_trades[day_str] = daily_trades.get(day_str, 0) + 1

            if side == "BUY":
                sl_price = entry_price - stop_dist
            else:
                sl_price = entry_price + stop_dist

            # Define Target 1 and Target 2 depending on exit policy
            tp1_mult = 1.2
            tp2_mult = 2.0
            if exit_policy == "FIXED_1.5R":
                tp1_mult = 1.5
                tp2_mult = 1.5
            elif exit_policy == "FIXED_2.0R":
                tp1_mult = 2.0
                tp2_mult = 2.0
            elif exit_policy == "FIXED_2.5R":
                tp1_mult = 2.5
                tp2_mult = 2.5
            elif exit_policy == "FIXED_3.0R":
                tp1_mult = 3.0
                tp2_mult = 3.0
            elif exit_policy == "DEFERRED_BE_2.0R":
                tp1_mult = 2.0
                tp2_mult = 2.0
            elif exit_policy == "PARTIAL_1.2R_2.0R":
                tp1_mult = 1.2
                tp2_mult = 2.0
            elif exit_policy == "PARTIAL_1.5R_2.5R":
                tp1_mult = 1.5
                tp2_mult = 2.5
            elif exit_policy == "PARTIAL_1.5R_3.0R":
                tp1_mult = 1.5
                tp2_mult = 3.0

            if side == "BUY":
                tp1_price = entry_price + (tp1_mult * stop_dist)
                tp2_price = entry_price + (tp2_mult * stop_dist)
            else:
                tp1_price = entry_price - (tp1_mult * stop_dist)
                tp2_price = entry_price - (tp2_mult * stop_dist)

            # Bar-by-bar trade simulation
            be_active = False
            half_closed = False
            realized_pnl = 0.0
            remaining_lots = lots
            trade_closed = False
            exit_reason = "TIMEOUT"
            exit_price = entry_price
            bars_held = 0

            # BE threshold trigger: 1.0R ($8.00 minimum)
            be_trigger = max(round(stop_dist * 1.0, 2), 8.00)

            for k in range(sim_start_idx, min(sim_start_idx + 72, n)):
                bars_held = k - sim_start_idx + 1
                b_h = highs[k]
                b_l = lows[k]
                b_c = closes[k]
                b_st_dir = st_obj["direction"][k]
                b_d_mid = dc["middle"][k]

                if side == "BUY":
                    # Check Deferred Break-Even trigger
                    if exit_policy.startswith("DEFERRED") or exit_policy.startswith("PARTIAL"):
                        if not be_active and (b_h - entry_price) >= be_trigger:
                            be_active = True
                            sl_price = entry_price + SPREAD  # scratch covers XM spread

                    # Partial Scale-Out at TP1
                    if exit_policy.startswith("PARTIAL") and not half_closed and b_h >= tp1_price:
                        half_closed = True
                        part_lots = round(lots * 0.5, 2)
                        realized_pnl += (tp1_price - entry_price) * part_lots * 100.0
                        remaining_lots = max(0.01, round(lots - part_lots, 2))
                        be_active = True
                        sl_price = entry_price + SPREAD

                    # Dynamic Trailing Exit Options
                    if exit_policy == "ST_TRAILING" and b_st_dir == -1:
                        trade_closed = True
                        exit_price = b_c
                        exit_reason = "ST_FLIP"
                        break
                    elif exit_policy == "MID_BAND_EXIT" and b_d_mid is not None and b_l <= b_d_mid:
                        trade_closed = True
                        exit_price = b_d_mid
                        exit_reason = "MID_BAND"
                        break

                    # Stop Loss check
                    if b_l <= sl_price:
                        trade_closed = True
                        exit_price = sl_price
                        exit_reason = "BE_SCRATCH" if be_active and sl_price >= entry_price else "STOP_LOSS"
                        break

                    # Take Profit 2 check
                    if b_h >= tp2_price:
                        trade_closed = True
                        exit_price = tp2_price
                        exit_reason = "TAKE_PROFIT"
                        break

                else:  # SELL
                    eff_h = b_h + SPREAD
                    eff_l = b_l + SPREAD

                    if exit_policy.startswith("DEFERRED") or exit_policy.startswith("PARTIAL"):
                        if not be_active and (entry_price - b_l) >= be_trigger:
                            be_active = True
                            sl_price = entry_price - SPREAD

                    if exit_policy.startswith("PARTIAL") and not half_closed and eff_l <= tp1_price:
                        half_closed = True
                        part_lots = round(lots * 0.5, 2)
                        realized_pnl += (entry_price - tp1_price) * part_lots * 100.0
                        remaining_lots = max(0.01, round(lots - part_lots, 2))
                        be_active = True
                        sl_price = entry_price - SPREAD

                    if exit_policy == "ST_TRAILING" and b_st_dir == 1:
                        trade_closed = True
                        exit_price = b_c
                        exit_reason = "ST_FLIP"
                        break
                    elif exit_policy == "MID_BAND_EXIT" and b_d_mid is not None and eff_h >= b_d_mid:
                        trade_closed = True
                        exit_price = b_d_mid
                        exit_reason = "MID_BAND"
                        break

                    if eff_h >= sl_price:
                        trade_closed = True
                        exit_price = sl_price
                        exit_reason = "BE_SCRATCH" if be_active and sl_price <= entry_price else "STOP_LOSS"
                        break

                    if eff_l <= tp2_price:
                        trade_closed = True
                        exit_price = tp2_price
                        exit_reason = "TAKE_PROFIT"
                        break

            if not trade_closed:
                exit_price = closes[min(sim_start_idx + 72, n - 1)]
                exit_reason = "TIME_EXPIRY"

            # Compute remaining pnl
            if side == "BUY":
                runner_pnl = (exit_price - entry_price) * remaining_lots * 100.0
            else:
                runner_pnl = (entry_price - exit_price) * remaining_lots * 100.0

            total_trade_pnl = round(realized_pnl + runner_pnl, 2)

            equity = round(equity + total_trade_pnl, 2)
            if equity > peak_equity:
                peak_equity = equity
            dd_usd = peak_equity - equity
            dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            is_scratch = abs(total_trade_pnl) <= 5.0
            is_win = total_trade_pnl > 5.0

            if not is_win and not is_scratch and loss_cooldown_bars > 0:
                loss_cooldown_until = sim_start_idx + bars_held + loss_cooldown_bars

            active_until = sim_start_idx + bars_held

            closed_trades.append({
                "side": side,
                "pnl": total_trade_pnl,
                "is_win": is_win,
                "is_scratch": is_scratch,
                "exit_reason": exit_reason,
            })

        # Calculate metrics
        total_trades = len(closed_trades)
        if total_trades == 0:
            return {"name": name, "trades": 0, "net_pnl": 0.0, "return_pct": 0.0}

        wins = [t for t in closed_trades if t["is_win"]]
        losses = [t for t in closed_trades if not t["is_win"] and not t["is_scratch"]]
        scratches = [t for t in closed_trades if t["is_scratch"]]

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        net_pnl = round(sum(t["pnl"] for t in closed_trades), 2)
        ret_pct = round((net_pnl / INITIAL_CAPITAL) * 100.0, 1)

        win_rate = round(len(wins) / total_trades * 100.0, 1)
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0
        avg_win = round(gross_profit / len(wins), 2) if wins else 0.0
        avg_loss = round(gross_loss / len(losses), 2) if losses else 0.0
        payoff = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0
        expectancy = round(net_pnl / total_trades, 2)
        calmar = round(ret_pct / max_dd_pct, 2) if max_dd_pct > 0 else 0.0

        return {
            "name": name,
            "donch_period": donch_period,
            "entry_trigger": entry_trigger,
            "trend_filter": trend_filter,
            "atr_sl_mult": atr_sl_mult,
            "exit_policy": exit_policy,
            "session_mode": session_mode,
            "trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "scratches": len(scratches),
            "win_rate": win_rate,
            "profit_factor": pf,
            "net_pnl": net_pnl,
            "return_pct": ret_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff,
            "max_dd_pct": round(max_dd_pct, 1),
            "max_dd_usd": round(max_dd_usd, 2),
            "expectancy": expectancy,
            "calmar_ratio": calmar,
            "final_equity": equity,
        }

    # =========================================================================
    # Strategy Combination Definitions
    # =========================================================================
    strategies = [
        # --- Group 1: Donchian 20 Standard Variations (Exit & Target Combinations) ---
        simulate_strategy(
            name="1. Donchian 20 + Fixed 2.0R (Baseline Pure Runner)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="2. Donchian 20 + Fixed 2.5R (Extended Runner)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.5R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="3. Donchian 20 + Fixed 3.0R (Macro Big Runner)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_3.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="4. Donchian 20 + Fixed 1.5R (Quick Scalp/Swing)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_1.5R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="5. Donchian 20 + Deferred BE (1.0R) -> 2.0R Target",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="DEFERRED_BE_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="6. Donchian 20 + Hybrid Partial (50% @ 1.2R + BE Runner to 2.0R)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="PARTIAL_1.2R_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="7. Donchian 20 + Hybrid Partial (50% @ 1.5R + BE Runner to 2.5R)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="PARTIAL_1.5R_2.5R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="8. Donchian 20 + Hybrid Partial (50% @ 1.5R + BE Runner to 3.0R)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="PARTIAL_1.5R_3.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="9. Donchian 20 + SuperTrend Dynamic Trailing Exit",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="ST_TRAILING", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="10. Donchian 20 + Mid-Band Exit (Classic Turtle Trend)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="MID_BAND_EXIT", session_mode="LONDON_NY",
        ),

        # --- Group 2: Lookback Period Sensitivity (10, 15, 30, 40 Bars) ---
        simulate_strategy(
            name="11. Donchian 10-Bar Quick Breakout (Fixed 2.0R)",
            donch_period=10, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="12. Donchian 15-Bar Breakout (Fixed 2.0R)",
            donch_period=15, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="13. Donchian 30-Bar Established Breakout (Fixed 2.0R)",
            donch_period=30, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="14. Donchian 40-Bar Macro Breakout (Fixed 2.0R)",
            donch_period=40, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="15. Donchian 30-Bar + Partial Scale (50% @ 1.2R + Runner 2.0R)",
            donch_period=30, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="PARTIAL_1.2R_2.0R", session_mode="LONDON_NY",
        ),

        # --- Group 3: Trigger Confirmation & Entry Filtering ---
        simulate_strategy(
            name="16. Donchian 20 + Bar Close Confirmation (No Wicks)",
            donch_period=20, entry_trigger="CLOSE", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="17. Donchian 20 + Pullback Limit (8-Pip Value Zone Discount)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
            use_pullback_limit=True, pullback_discount=0.80,
        ),
        simulate_strategy(
            name="18. Donchian 20 + Pullback Limit + Partial Scale-Out",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="PARTIAL_1.2R_2.0R", session_mode="LONDON_NY",
            use_pullback_limit=True, pullback_discount=0.80,
        ),

        # --- Group 4: Trend Filter Combinations ---
        simulate_strategy(
            name="19. Donchian 20 + SuperTrend Only (No EMA Filter)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_ONLY",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="20. Donchian 20 + SuperTrend + 50 EMA Filter",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA50",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="21. Donchian 20 + SuperTrend Fast (10, 2.0) + 200 EMA",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST2_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="22. Donchian 20 + Golden Cross (EMA 50 > 200) + SuperTrend",
            donch_period=20, entry_trigger="TOUCH", trend_filter="EMA50_200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="23. Donchian 20 + MACD Confirmation + SuperTrend",
            donch_period=20, entry_trigger="TOUCH", trend_filter="MACD_ST3",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),

        # --- Group 5: Stop-Loss Calibration (1.0x vs 1.5x vs 2.0x ATR) ---
        simulate_strategy(
            name="24. Donchian 20 + Tight Stop (1.0x ATR, min $6) + 2.0R",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.0, min_sl_dollars=6.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="25. Donchian 20 + Wide Stop (2.0x ATR, min $10) + 2.0R",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=2.0, min_sl_dollars=10.0, exit_policy="FIXED_2.0R", session_mode="LONDON_NY",
        ),
        simulate_strategy(
            name="26. Donchian 20 + Wide Stop (2.0x ATR) + Hybrid Partial",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=2.0, min_sl_dollars=10.0, exit_policy="PARTIAL_1.2R_2.0R", session_mode="LONDON_NY",
        ),

        # --- Group 6: Session Window Sensitivity ---
        simulate_strategy(
            name="27. Donchian 20 + London/NY Overlap Only (12:00-16:00 UTC)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="OVERLAP",
        ),
        simulate_strategy(
            name="28. Donchian 20 + 24-Hour (No Session Filter)",
            donch_period=20, entry_trigger="TOUCH", trend_filter="ST3_EMA200",
            atr_sl_mult=1.5, min_sl_dollars=8.0, exit_policy="FIXED_2.0R", session_mode="24H",
        ),
    ]

    # Sort strategies by Net P&L descending
    strategies.sort(key=lambda s: s["net_pnl"], reverse=True)

    # 3. Print Leaderboard
    print("=" * 115)
    print(f"{'Rank':<4} {'Strategy Name':<52} {'Trades':<7} {'Win%':<7} {'PF':<6} {'Net P&L ($)':<12} {'Return%':<9} {'Max DD%':<8} {'Payoff':<7}")
    print("-" * 115)

    for rank, s in enumerate(strategies, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:>2}."
        pnl_str = f"+${s['net_pnl']:,.2f}" if s['net_pnl'] >= 0 else f"-${abs(s['net_pnl']):,.2f}"
        ret_str = f"+{s['return_pct']:.1f}%" if s['return_pct'] >= 0 else f"{s['return_pct']:.1f}%"
        print(f"{medal:<4} {s['name']:<52} {s['trades']:<7} {s['win_rate']:<7.1f} {s['profit_factor']:<6.2f} {pnl_str:<12} {ret_str:<9} {s['max_dd_pct']:<8.1f} {s['payoff_ratio']:<7.2f}")

    print("=" * 115)

    # Save to JSON
    out_path = os.path.expanduser("~/.gold_trading_data/strategy_combinations_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(strategies, f, indent=2)
    print(f"\nDetailed backtest results saved to {out_path}")

    return strategies


if __name__ == "__main__":
    run_combinations_backtest()
