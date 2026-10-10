"""
Backtesting Engine for Institutional Two-Stage Execution on Gold (XAUUSD):
1. Stage 1: Close 50% position volume at Target 1 (+1.0R) -> Locks cash profit.
2. Stage 2: Move Stop Loss of remaining 50% to Break-Even + $0.50 -> Risk-free runner.
3. Stage 3: Dynamic Runner Trail with wide ATR-cleared buffer ($6.00 - $8.00) up to Target 2 (2.5R - 3.0R).
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


def run_institutional_backtest(
    period: str = "60d",
    scale_out_r: float = 1.0,
    scale_out_pct: float = 0.5,
    runner_trail_dist: float = 6.0,
    runner_max_r: Optional[float] = 3.0,
    be_offset: float = 0.50,
    spread_usd: float = 0.35,
    start_balance: float = 200.0,
) -> Dict[str, Any]:
    candles_1h = fetch_gold_candles("XAUUSD", period=period, interval="1h")
    candles_15m = []
    try:
        candles_15m = fetch_gold_candles("XAUUSD", period="60d" if period != "60d" else period, interval="15m")
    except Exception:
        candles_15m = []

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

    balance = start_balance
    peak_balance = start_balance
    max_dd_usd = 0.0
    ratchet_floor = 0.0
    active_pos: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = []

    for i in range(50, len(candles_1h)):
        d_1h = dates_1h[i]
        m15_start_idx = h_to_m15[i]

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
                tp1 = active_pos["tp1"]
                tp2 = active_pos["tp2"]
                lots = active_pos["lots"]
                m_h, m_l = float(mb["high"]), float(mb["low"])
                m_d = mb["date"]

                fav = (m_h - entry_p) if side == "BUY" else (entry_p - m_l)
                if fav > active_pos["peak_fav"]:
                    active_pos["peak_fav"] = fav

                # Stage 1: Check TP1 Scale-Out (50%)
                if not active_pos["tp1_taken"]:
                    hit_tp1 = (m_h >= tp1) if side == "BUY" else (m_l <= tp1)
                    if hit_tp1:
                        active_pos["tp1_taken"] = True
                        tp1_pts = abs(tp1 - entry_p) - spread_usd
                        active_pos["realized_tp1_usd"] = round(tp1_pts * (lots * scale_out_pct) * 100.0, 2)
                        new_sl = (entry_p + be_offset) if side == "BUY" else (entry_p - be_offset)
                        active_pos["current_sl"] = new_sl

                # Stage 3: Runner Trailing
                if active_pos["tp1_taken"]:
                    if side == "BUY":
                        dyn_sl = entry_p + (active_pos["peak_fav"] - runner_trail_dist)
                        active_pos["current_sl"] = max(active_pos["current_sl"], dyn_sl)
                    else:
                        dyn_sl = entry_p - (active_pos["peak_fav"] - runner_trail_dist)
                        active_pos["current_sl"] = min(active_pos["current_sl"], dyn_sl)

                curr_sl = active_pos["current_sl"]
                exited, exit_price, exit_reason = False, None, None

                if side == "BUY":
                    if active_pos["tp1_taken"] and runner_max_r and m_h >= tp2:
                        exit_price = tp2
                        exit_reason = "TP2_Extension_Hit"
                        exited = True
                    elif m_l <= curr_sl:
                        exit_price = curr_sl
                        exit_reason = "Runner_Trailed_Exit" if active_pos["tp1_taken"] else "Full_SL"
                        exited = True
                else:
                    if active_pos["tp1_taken"] and runner_max_r and m_l <= tp2:
                        exit_price = tp2
                        exit_reason = "TP2_Extension_Hit"
                        exited = True
                    elif m_h >= curr_sl:
                        exit_price = curr_sl
                        exit_reason = "Runner_Trailed_Exit" if active_pos["tp1_taken"] else "Full_SL"
                        exited = True

                if exited:
                    if active_pos["tp1_taken"]:
                        rem_pts = ((exit_price - entry_p) if side == "BUY" else (entry_p - exit_price)) - spread_usd
                        rem_usd = round(rem_pts * (lots * (1.0 - scale_out_pct)) * 100.0, 2)
                        trade_pnl_usd = round(active_pos["realized_tp1_usd"] + rem_usd, 2)
                        net_points = round(
                            (scale_out_pct * (abs(tp1 - entry_p) - spread_usd)) + ((1.0 - scale_out_pct) * rem_pts), 2
                        )
                    else:
                        pts = (exit_price - entry_p) if side == "BUY" else (entry_p - exit_price)
                        net_points = round(pts - spread_usd, 2)
                        trade_pnl_usd = round(net_points * lots * 100.0, 2)

                    balance = max(0.0, round(balance + trade_pnl_usd, 2))
                    if balance > peak_balance:
                        peak_balance = balance
                        if peak_balance >= 250.0:
                            ratchet_floor = 150.0
                        if peak_balance >= 500.0:
                            ratchet_floor = 350.0
                        if peak_balance >= 1000.0:
                            ratchet_floor = 700.0
                        if peak_balance >= 2500.0:
                            ratchet_floor = 1800.0
                    dd = peak_balance - balance
                    if dd > max_dd_usd:
                        max_dd_usd = dd

                    trades.append({
                        "entry_date": active_pos["entry_date"],
                        "exit_date": m_d,
                        "side": side,
                        "lots": lots,
                        "net_points": net_points,
                        "pnl_usd": trade_pnl_usd,
                        "exit_reason": exit_reason,
                        "peak_fav": round(active_pos["peak_fav"], 2),
                        "tp1_taken": active_pos["tp1_taken"],
                        "balance_after": balance,
                    })
                    active_pos = None
                    if balance <= 5.0:
                        break
                    break

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
                        sl_dist = max(1.5 * atr_val, 8.00)
                        lots = round(max(planned_risk_usd / (sl_dist * 100.0), 0.01), 2)

                        if is_buy:
                            eff_entry = max(o, prev_d_up) if o <= prev_d_up else c
                            entry_p = eff_entry + spread_usd
                            init_sl = round(entry_p - sl_dist, 2)
                            tp1 = round(entry_p + (scale_out_r * sl_dist), 2)
                            tp2 = round(entry_p + (runner_max_r * sl_dist), 2) if runner_max_r else None
                            active_pos = {
                                "side": "BUY",
                                "entry_date": d_1h,
                                "entry_price": round(entry_p, 2),
                                "initial_sl": init_sl,
                                "current_sl": init_sl,
                                "tp1": tp1,
                                "tp2": tp2,
                                "lots": lots,
                                "tp1_taken": False,
                                "realized_tp1_usd": 0.0,
                                "peak_fav": 0.0,
                            }
                        else:
                            eff_entry = min(o, prev_d_low) if o >= prev_d_low else c
                            entry_p = eff_entry - spread_usd
                            init_sl = round(entry_p + sl_dist, 2)
                            tp1 = round(entry_p - (scale_out_r * sl_dist), 2)
                            tp2 = round(entry_p - (runner_max_r * sl_dist), 2) if runner_max_r else None
                            active_pos = {
                                "side": "SELL",
                                "entry_date": d_1h,
                                "entry_price": round(entry_p, 2),
                                "initial_sl": init_sl,
                                "current_sl": init_sl,
                                "tp1": tp1,
                                "tp2": tp2,
                                "lots": lots,
                                "tp1_taken": False,
                                "realized_tp1_usd": 0.0,
                                "peak_fav": 0.0,
                            }

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] < 0]
    scratches = [t for t in trades if t["pnl_usd"] == 0]

    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else 999.0
    wr = (len(wins) / len(trades) * 100) if trades else 0.0

    avg_win = gp / len(wins) if wins else 0.0
    avg_loss = gl / len(losses) if losses else 0.0
    payoff = avg_win / avg_loss if (losses and avg_loss > 0) else 0.0

    return {
        "period": period,
        "strategy": "INSTITUTIONAL_TWO_STAGE",
        "scale_out_r": scale_out_r,
        "runner_trail_dist": runner_trail_dist,
        "runner_max_r": runner_max_r,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "scratches": len(scratches),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "payoff_ratio": round(payoff, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "initial_balance": start_balance,
        "final_balance": round(balance, 2),
        "net_pnl_usd": round(balance - start_balance, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "tp1_scale_out_hits": sum(1 for t in trades if t["tp1_taken"]),
        "full_sl_hits": sum(1 for t in trades if not t["tp1_taken"]),
        "sample_trades": trades[-5:],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Institutional Two-Stage Execution on Gold")
    parser.add_argument("--period", default="60d", help="Candle period (60d, 6mo)")
    parser.add_argument("--scale-out-r", type=float, default=1.0, help="Target 1 R-multiple for 50% partial (default: 1.0)")
    parser.add_argument("--trail-dist", type=float, default=6.0, help="Runner trailing buffer in USD (default: 6.0)")
    parser.add_argument("--max-r", type=float, default=3.0, help="Target 2 extension R-multiple (default: 3.0)")
    parser.add_argument("--start-balance", type=float, default=200.0, help="Starting balance in USD (default: 200.0)")
    args = parser.parse_args()

    res = run_institutional_backtest(
        period=args.period,
        scale_out_r=args.scale_out_r,
        runner_trail_dist=args.trail_dist,
        runner_max_r=args.max_r,
        start_balance=args.start_balance,
    )
    print(json.dumps(res, indent=2))
