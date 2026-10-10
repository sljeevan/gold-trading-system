"""
Backtesting Engine for COMPOUNDING_TOUCH Strategy with Trailing Stop Analysis.
Evaluates:
1. Baseline: Fixed ATR Stop Loss + 1.25R Target
2. Static Profit Lock: Move SL to +X points once price advances by trigger points
3. Dynamic Trailing Stop: Continuously trail stop loss behind peak excursion by Y points
4. Multi-Timeframe High-Fidelity Simulation: 1H Signal Detection + 15m Sub-Bar Intra-Trade Resolution
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
)


def run_trailing_stop_backtest(
    period: str = "60d",
    trigger_usd: Optional[float] = 1.00,  # 100 points = $1.00
    trail_dist_usd: Optional[float] = 1.00,  # Dynamic trailing distance
    static_lock_usd: Optional[float] = None,  # If set, locks fixed profit rather than dynamic trail
    spread_usd: float = 0.35,
    rr_multiplier: float = 1.25,
    min_sl_usd: float = 8.00,
    atr_multiplier: float = 1.5,
    start_balance: float = 200.0,
) -> Dict[str, Any]:
    # 1. Fetch multi-timeframe candles
    candles_1h = fetch_gold_candles("XAUUSD", period=period, interval="1h")
    candles_15m = []
    try:
        candles_15m = fetch_gold_candles("XAUUSD", period="60d" if period != "60d" else period, interval="15m")
    except Exception:
        candles_15m = []

    if len(candles_1h) < 60:
        return {"error": f"Insufficient 1H candles ({len(candles_1h)})"}

    m15_map = {c["date"]: idx for idx, c in enumerate(candles_15m)}
    h_to_m15 = [m15_map.get(h_bar["date"]) for h_bar in candles_1h]

    highs_1h = [float(c["high"]) for c in candles_1h]
    lows_1h = [float(c["low"]) for c in candles_1h]
    closes_1h = [float(c["close"]) for c in candles_1h]
    opens_1h = [float(c["open"]) for c in candles_1h]
    dates_1h = [c["date"] for c in candles_1h]

    dc_1h = calc_donchian(highs_1h, lows_1h, 10)
    st_1h = calc_supertrend(highs_1h, lows_1h, closes_1h, 10, 3.0)
    ema50_1h = calc_ema(closes_1h, 50)
    atr_1h = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    rsi_1h = calc_rsi(closes_1h, 14)
    macd_1h = calc_macd(closes_1h, 12, 26, 9)

    trades: List[Dict[str, Any]] = []
    active_pos: Optional[Dict[str, Any]] = None

    balance = start_balance
    peak_balance = start_balance
    max_dd_usd = 0.0
    ratchet_floor = 0.0

    for i in range(50, len(candles_1h)):
        d_1h = dates_1h[i]
        m15_start_idx = h_to_m15[i]

        # Manage open position via 15m sub-bars
        if active_pos is not None:
            sub_bars = (
                candles_15m[m15_start_idx : m15_start_idx + 4]
                if (m15_start_idx is not None and len(candles_15m) > 0)
                else [candles_1h[i]]
            )

            for mb in sub_bars:
                if active_pos is None:
                    break

                side = active_pos["side"]
                entry_p = active_pos["entry_price"]
                curr_sl = active_pos["current_sl"]
                tp = active_pos["tp"]
                lots = active_pos["lots"]
                m_h, m_l = float(mb["high"]), float(mb["low"])
                m_d = mb["date"]

                fav = (m_h - entry_p) if side == "BUY" else (entry_p - m_l)
                if fav > active_pos["peak_fav"]:
                    active_pos["peak_fav"] = fav

                # Check trailing trigger
                if trigger_usd is not None and active_pos["peak_fav"] >= trigger_usd:
                    if not active_pos["is_trailed"]:
                        active_pos["is_trailed"] = True
                        if static_lock_usd is not None:
                            new_sl = (entry_p + static_lock_usd) if side == "BUY" else (entry_p - static_lock_usd)
                            active_pos["current_sl"] = max(curr_sl, new_sl) if side == "BUY" else min(curr_sl, new_sl)

                    if trail_dist_usd is not None:
                        if side == "BUY":
                            dyn_sl = entry_p + (active_pos["peak_fav"] - trail_dist_usd)
                            active_pos["current_sl"] = max(active_pos["current_sl"], dyn_sl)
                        else:
                            dyn_sl = entry_p - (active_pos["peak_fav"] - trail_dist_usd)
                            active_pos["current_sl"] = min(active_pos["current_sl"], dyn_sl)

                curr_sl = active_pos["current_sl"]
                exited, exit_price, exit_reason = False, None, None

                if side == "BUY":
                    if m_l <= curr_sl:
                        exit_price = curr_sl
                        exit_reason = "Trailed_SL" if active_pos["is_trailed"] else "SL"
                        exited = True
                    elif m_h >= tp:
                        exit_price = tp
                        exit_reason = "TP"
                        exited = True
                else:  # SELL
                    if m_h >= curr_sl:
                        exit_price = curr_sl
                        exit_reason = "Trailed_SL" if active_pos["is_trailed"] else "SL"
                        exited = True
                    elif m_l <= tp:
                        exit_price = tp
                        exit_reason = "TP"
                        exited = True

                if exited:
                    pts = (exit_price - entry_p) if side == "BUY" else (entry_p - exit_price)
                    net_pts = pts - spread_usd
                    pnl_usd = round(net_pts * lots * 100.0, 2)
                    balance = max(0.0, round(balance + pnl_usd, 2))
                    if balance > peak_balance:
                        peak_balance = balance
                        if peak_balance >= 250.0:
                            ratchet_floor = 150.0
                        if peak_balance >= 500.0:
                            ratchet_floor = 350.0
                    dd = peak_balance - balance
                    if dd > max_dd_usd:
                        max_dd_usd = dd

                    trades.append({
                        "entry_date": active_pos["entry_date"],
                        "exit_date": m_d,
                        "side": side,
                        "entry_price": entry_p,
                        "exit_price": exit_price,
                        "lots": lots,
                        "net_points": round(net_pts, 2),
                        "pnl_usd": pnl_usd,
                        "exit_reason": exit_reason,
                        "peak_fav": round(active_pos["peak_fav"], 2),
                        "balance_after": balance,
                    })
                    active_pos = None
                    break

        # Check for new entry setup on 1H bar
        if active_pos is None and balance > 5.0:
            try:
                hour = int(d_1h.split("T" if "T" in d_1h else " ")[1].split(":")[0])
            except Exception:
                hour = 12

            if 7 <= hour <= 19:
                c, h, l, o = closes_1h[i], highs_1h[i], lows_1h[i], opens_1h[i]
                prev_d_up, prev_d_low = dc_1h["upper"][i - 1], dc_1h["lower"][i - 1]
                st_dir = st_1h["direction"][i]
                e50, r = ema50_1h[i], rsi_1h[i]
                atr_val = atr_1h[i] or 10.0

                if prev_d_up is not None and prev_d_low is not None and e50 is not None and r is not None:
                    is_buy = (h >= prev_d_up or c >= prev_d_up) and st_dir == 1 and c > e50 and r <= 75.0
                    is_sell = (l <= prev_d_low or c <= prev_d_low) and st_dir == -1 and c < e50 and r >= 25.0

                    m_line = macd_1h["macd"]
                    m_sig = macd_1h["signal"]
                    if (
                        not is_buy
                        and not is_sell
                        and m_line[i] is not None
                        and m_sig[i] is not None
                        and m_line[i - 1] is not None
                        and m_sig[i - 1] is not None
                    ):
                        if st_dir == 1 and c > e50 and m_line[i - 1] <= m_sig[i - 1] and m_line[i] > m_sig[i] and 45 <= r <= 70 and c > o:
                            is_buy = True
                        elif st_dir == -1 and c < e50 and m_line[i - 1] >= m_sig[i - 1] and m_line[i] < m_sig[i] and 30 <= r <= 55 and c < o:
                            is_sell = True

                    if is_buy or is_sell:
                        riskable_equity = max(balance - ratchet_floor, balance * 0.5)
                        planned_risk_usd = round(riskable_equity * 0.10, 2)
                        sl_dist = max(atr_multiplier * atr_val, min_sl_usd)
                        lots = round(max(planned_risk_usd / (sl_dist * 100.0), 0.01), 2)

                        if is_buy:
                            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
                            entry_p = eff_entry + spread_usd
                            init_sl = round(entry_p - sl_dist, 2)
                            tp_p = round(entry_p + (rr_multiplier * sl_dist), 2)
                            active_pos = {
                                "side": "BUY",
                                "entry_date": d_1h,
                                "entry_price": round(entry_p, 2),
                                "initial_sl": init_sl,
                                "current_sl": init_sl,
                                "tp": tp_p,
                                "lots": lots,
                                "is_trailed": False,
                                "peak_fav": 0.0,
                            }
                        else:
                            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c
                            entry_p = eff_entry - spread_usd
                            init_sl = round(entry_p + sl_dist, 2)
                            tp_p = round(entry_p - (rr_multiplier * sl_dist), 2)
                            active_pos = {
                                "side": "SELL",
                                "entry_date": d_1h,
                                "entry_price": round(entry_p, 2),
                                "initial_sl": init_sl,
                                "current_sl": init_sl,
                                "tp": tp_p,
                                "lots": lots,
                                "is_trailed": False,
                                "peak_fav": 0.0,
                            }

    # Aggregate performance
    wins = [t for t in trades if t["net_points"] > 0]
    losses = [t for t in trades if t["net_points"] < 0]
    scratches = [t for t in trades if t["net_points"] == 0]

    gp_pts = sum(t["net_points"] for t in wins)
    gl_pts = abs(sum(t["net_points"] for t in losses))
    net_pts = sum(t["net_points"] for t in trades)
    wr = (len(wins) / len(trades) * 100) if trades else 0.0
    pf = (gp_pts / gl_pts) if gl_pts > 0 else 999.0

    avg_win = (gp_pts / len(wins)) if wins else 0.0
    avg_loss = (gl_pts / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    tp_hits = sum(1 for t in trades if t["exit_reason"] == "TP")
    trailed_hits = sum(1 for t in trades if t["exit_reason"] == "Trailed_SL")
    sl_hits = sum(1 for t in trades if t["exit_reason"] == "SL")

    cum, peak, max_dd_pts = 0.0, 0.0, 0.0
    for t in trades:
        cum += t["net_points"]
        if cum > peak:
            peak = cum
        if peak - cum > max_dd_pts:
            max_dd_pts = peak - cum

    return {
        "period": period,
        "strategy": "COMPOUNDING_TOUCH",
        "trigger_usd": trigger_usd,
        "trail_dist_usd": trail_dist_usd,
        "static_lock_usd": static_lock_usd,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "scratches": len(scratches),
        "win_rate_pct": round(wr, 1),
        "net_points": round(net_pts, 2),
        "profit_factor": round(pf, 2),
        "payoff_ratio": round(payoff, 2),
        "avg_win_pts": round(avg_win, 2),
        "avg_loss_pts": round(avg_loss, 2),
        "max_drawdown_pts": round(max_dd_pts, 2),
        "initial_balance": start_balance,
        "final_balance": round(balance, 2),
        "net_pnl_usd": round(balance - start_balance, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "exit_breakdown": {
            "take_profit_hits": tp_hits,
            "trailed_sl_hits": trailed_hits,
            "full_sl_hits": sl_hits,
        },
        "sample_trades": trades[-5:],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Trailing Stop on COMPOUNDING_TOUCH Strategy")
    parser.add_argument("--period", default="60d", help="Candle period (60d, 6mo)")
    parser.add_argument("--trigger", type=float, default=1.00, help="Trail activation trigger in USD ($1.00 = 100 points)")
    parser.add_argument("--trail-dist", type=float, default=1.00, help="Trailing distance in USD behind peak")
    parser.add_argument("--static-lock", type=float, default=None, help="Static profit lock in USD")
    parser.add_argument("--start-balance", type=float, default=200.0, help="Starting balance in USD")
    args = parser.parse_args()

    res = run_trailing_stop_backtest(
        period=args.period,
        trigger_usd=args.trigger,
        trail_dist_usd=args.trail_dist,
        static_lock_usd=args.static_lock,
        start_balance=args.start_balance,
    )
    print(json.dumps(res, indent=2))
