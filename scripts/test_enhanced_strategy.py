"""
Comparative Backtest: Original Static 15-pip Compounding vs Enhanced Adaptive Strategy
Evaluates:
1. Original Strategy: Static 15-pip SL ($1.54), 20-pip TP ($2.00), No daily trade cap, 15m cooldown.
2. Enhanced Strategy: Dynamic ATR SL (min $6.00), 1.5-2.0 R:R TP, Strict Session Filter (07-19 UTC),
   Max 5 trades/day cap, 60m loss cooldown.
"""
from __future__ import annotations

import sys
import numpy as np
from datetime import datetime

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_supertrend, calc_rsi, calc_atr


def run_comparative_backtest():
    print("Fetching historical 15m Gold candles...")
    candles_15m = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
    if not candles_15m or len(candles_15m) < 200:
        print(f"Insufficient candles: {len(candles_15m) if candles_15m else 0}")
        return

    n = len(candles_15m)
    print(f"Loaded {n} candles ({candles_15m[0]['date']} to {candles_15m[-1]['date']})")

    highs = np.array([float(c["high"]) for c in candles_15m], dtype=float)
    lows = np.array([float(c["low"]) for c in candles_15m], dtype=float)
    closes = np.array([float(c["close"]) for c in candles_15m], dtype=float)
    opens = np.array([float(c["open"]) for c in candles_15m], dtype=float)
    dates = [c["date"] for c in candles_15m]

    ema9 = calc_ema(closes.tolist(), 9)
    ema21 = calc_ema(closes.tolist(), 21)
    ema50 = calc_ema(closes.tolist(), 50)
    rsi = calc_rsi(closes.tolist(), 14)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    def simulate(
        use_atr_stops: bool,
        enforce_session_filter: bool,
        max_daily_trades: int,
        loss_cooldown_bars: int,
        spread: float = 0.35,
    ):
        wins = 0
        losses = 0
        scratches = 0
        gross_profit = 0.0
        gross_loss = 0.0
        total_pnl = 0.0
        trade_log = []

        current_day = ""
        daily_trades_count = 0
        blocked_until_bar = -1
        last_trade_bar = -1

        for i in range(50, n - 16):
            if i < blocked_until_bar:
                continue

            dt = dates[i]
            day = dt.split("T")[0] if "T" in dt else dt.split(" ")[0]
            if day != current_day:
                current_day = day
                daily_trades_count = 0

            if max_daily_trades > 0 and daily_trades_count >= max_daily_trades:
                continue

            if i == last_trade_bar:
                continue

            hour = 12
            try:
                sep = "T" if "T" in dt else " "
                hour = int(dt.split(sep)[1].split(":")[0])
            except Exception:
                pass

            if enforce_session_filter and not (7 <= hour <= 19):
                continue

            e9, e21, e50, r, a = ema9[i], ema21[i], ema50[i], rsi[i], atr[i]
            if None in (e9, e21, e50, r, a):
                continue

            c, h, l, o = closes[i], highs[i], lows[i], opens[i]

            # Signal logic: Value Zone Pullback
            dir_ = None
            if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
                dir_ = "BUY"
            elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
                dir_ = "SELL"

            if not dir_:
                continue

            k_fill = i + 1
            if k_fill >= n:
                break

            fill_p = opens[k_fill]
            last_trade_bar = i
            daily_trades_count += 1

            eff_entry = fill_p + spread if dir_ == "BUY" else fill_p

            if use_atr_stops:
                sl_pts = max(1.5 * a, 6.0)
                tp_pts = max(2.0 * a, sl_pts * 1.5)
                be_pts = sl_pts * 0.75
            else:
                sl_pts = 1.54
                tp_pts = 2.00
                be_pts = 1.00

            tp = eff_entry + tp_pts if dir_ == "BUY" else eff_entry - tp_pts - spread
            sl = eff_entry - sl_pts if dir_ == "BUY" else eff_entry + sl_pts + spread
            be_trigger = eff_entry + be_pts if dir_ == "BUY" else eff_entry - be_pts

            won, lost, scratched = False, False, False
            is_be = False

            # Simulate intra-bar tick outcomes over subsequent candles
            for k in range(k_fill, min(k_fill + 24, n)):
                curr_sl = eff_entry if is_be else sl
                h_k, l_k = highs[k], lows[k]

                if dir_ == "BUY":
                    if l_k <= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
                    if h_k >= be_trigger:
                        is_be = True
                    if h_k >= tp:
                        won = True
                        break
                else:
                    if h_k >= curr_sl:
                        if is_be:
                            scratched = True
                        else:
                            lost = True
                        break
                    if l_k <= be_trigger:
                        is_be = True
                    if l_k <= tp:
                        won = True
                        break

            if not (won or lost or scratched):
                scratched = True

            # Lot size fixed at 0.05 for uniform comparison
            lot = 0.05
            contract_size = 100.0

            if won:
                wins += 1
                pnl = lot * contract_size * tp_pts
                gross_profit += pnl
                total_pnl += pnl
            elif lost:
                losses += 1
                pnl = -lot * contract_size * sl_pts
                gross_loss += abs(pnl)
                total_pnl += pnl
                if loss_cooldown_bars > 0:
                    blocked_until_bar = k + loss_cooldown_bars
            elif scratched:
                scratches += 1

        total_trades = wins + losses + scratches
        decided = wins + losses
        win_rate = (wins / decided * 100) if decided > 0 else 0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "total_pnl": total_pnl,
            "profit_factor": pf,
        }

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS ACROSS 60 DAYS OF 15M GOLD CANDLES")
    print("=" * 60)

    res_orig = simulate(
        use_atr_stops=False,
        enforce_session_filter=False,
        max_daily_trades=0,
        loss_cooldown_bars=1,
    )
    print("\n1. ORIGINAL STRATEGY (Tight $1.50 SL, No Session Filter, No Cap):")
    print(f"   Total Trades  : {res_orig['total_trades']} ({res_orig['wins']} W / {res_orig['losses']} L / {res_orig['scratches']} BE)")
    print(f"   Win Rate      : {res_orig['win_rate']:.1f}%")
    print(f"   Profit Factor : {res_orig['profit_factor']:.2f}")
    print(f"   Gross Profit  : +${res_orig['gross_profit']:,.2f}")
    print(f"   Gross Loss    : -${res_orig['gross_loss']:,.2f}")
    print(f"   Net P&L       : ${res_orig['total_pnl']:,.2f}")

    res_enhanced = simulate(
        use_atr_stops=True,
        enforce_session_filter=True,
        max_daily_trades=4,
        loss_cooldown_bars=4,  # 4 * 15m = 60m cooldown
    )
    print("\n2. ENHANCED STRATEGY (Dynamic ATR Stop >=$6, Session Filter 07-19 UTC, Max 4/day, 60m Cooldown):")
    print(f"   Total Trades  : {res_enhanced['total_trades']} ({res_enhanced['wins']} W / {res_enhanced['losses']} L / {res_enhanced['scratches']} BE)")
    print(f"   Win Rate      : {res_enhanced['win_rate']:.1f}%")
    print(f"   Profit Factor : {res_enhanced['profit_factor']:.2f}")
    print(f"   Gross Profit  : +${res_enhanced['gross_profit']:,.2f}")
    print(f"   Gross Loss    : -${res_enhanced['gross_loss']:,.2f}")
    print(f"   Net P&L       : ${res_enhanced['total_pnl']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_comparative_backtest()
