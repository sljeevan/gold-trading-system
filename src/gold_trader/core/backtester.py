"""
Backtesting and Walk-Forward Overfitting Validation Engine for Gold Strategies.
Pure Python — zero external dependencies.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_bollinger,
    calc_donchian,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
)


def _strat_supertrend(candles: List[Dict]) -> List[Dict]:
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    st = calc_supertrend(highs, lows, closes, 10, 3.0)

    trades, pos = [], None
    for i in range(1, len(candles)):
        d, dp = st["direction"][i], st["direction"][i - 1]
        if d is None or dp is None:
            continue
        p, dt = closes[i], candles[i]["date"]
        if pos is None and dp == -1 and d == 1:
            pos = {"entry_date": dt, "entry_price": p, "strategy": "supertrend"}
        elif pos is not None and dp == 1 and d == -1:
            trades.append({**pos, "exit_date": dt, "exit_price": p})
            pos = None
    return trades


def _strat_donchian(candles: List[Dict]) -> List[Dict]:
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    dc = calc_donchian(highs, lows, 20)

    trades, pos = [], None
    for i in range(1, len(candles)):
        if dc["upper"][i] is None or dc["lower"][i] is None:
            continue
        p, dt = closes[i], candles[i]["date"]
        if pos is None and dc["upper"][i - 1] is not None and highs[i - 1] > dc["upper"][i - 1]:
            pos = {"entry_date": dt, "entry_price": p, "strategy": "donchian"}
        elif pos is not None and p < dc["lower"][i]:
            trades.append({**pos, "exit_date": dt, "exit_price": p})
            pos = None
    return trades


def _strat_ema_cross(candles: List[Dict]) -> List[Dict]:
    closes = [c["close"] for c in candles]
    fast = calc_ema(closes, 20)
    slow = calc_ema(closes, 50)

    trades, pos = [], None
    for i in range(1, len(candles)):
        f, s, fp, sp = fast[i], slow[i], fast[i - 1], slow[i - 1]
        if None in (f, s, fp, sp):
            continue
        p, dt = closes[i], candles[i]["date"]
        if pos is None and fp < sp and f >= s:
            pos = {"entry_date": dt, "entry_price": p, "strategy": "ema_cross"}
        elif pos is not None and fp > sp and f <= s:
            trades.append({**pos, "exit_date": dt, "exit_price": p})
            pos = None
    return trades


def _strat_ema_5_20(
    candles: List[Dict],
    min_target_points: Optional[float] = None,
    stop_loss_points: Optional[float] = None,
    fast_period: int = 5,
    slow_period: int = 20,
    be_trigger_points: Optional[float] = None,
    trailing_stop_points: Optional[float] = None,
) -> List[Dict]:
    """
    EMA Crossover strategy for both LONG and SHORT trades (configurable fast & slow EMA).
    Long entry: EMA fast crosses above EMA slow.
    Short entry: EMA fast crosses below EMA slow.
    Exit: Reverse crossover, or optionally minimum target points (Take Profit), stop loss,
          breakeven trigger (locks profit at entry + spread), and trailing stop.
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    dates = [c["date"] for c in candles]

    fast = calc_ema(closes, fast_period)
    slow = calc_ema(closes, slow_period)

    trades: List[Dict[str, Any]] = []
    pos: Optional[Dict[str, Any]] = None

    for i in range(1, len(candles)):
        f, s = fast[i], slow[i]
        fp, sp = fast[i - 1], slow[i - 1]
        if None in (f, s, fp, sp):
            continue

        c_close = closes[i]
        c_high = highs[i]
        c_low = lows[i]
        c_date = dates[i]

        # Check in-trade excursion, breakeven, trailing stop, target, and SL
        if pos is not None:
            direction = pos["direction"]
            entry_p = pos["entry_price"]

            fav = (c_high - entry_p) if direction == "LONG" else (entry_p - c_low)
            adv = (entry_p - c_low) if direction == "LONG" else (c_high - entry_p)
            pos["max_fav_pts"] = max(pos.get("max_fav_pts", 0.0), fav)
            pos["max_adv_pts"] = max(pos.get("max_adv_pts", 0.0), adv)

            # Dynamic Breakeven & Trailing Stop update
            if be_trigger_points is not None and fav >= be_trigger_points:
                if direction == "LONG":
                    be_level = entry_p + 0.35  # Lock spread
                    current_sl = pos.get("current_sl")
                    pos["current_sl"] = max(current_sl if current_sl is not None else be_level, be_level)
                    if trailing_stop_points is not None:
                        pos["current_sl"] = max(pos["current_sl"], c_high - trailing_stop_points)
                else:
                    be_level = entry_p - 0.35
                    current_sl = pos.get("current_sl")
                    pos["current_sl"] = min(current_sl if current_sl is not None else be_level, be_level)
                    if trailing_stop_points is not None:
                        pos["current_sl"] = min(pos["current_sl"], c_low + trailing_stop_points)

            exited = False
            exit_price = None
            exit_reason = None

            if direction == "LONG":
                if min_target_points is not None and c_high >= entry_p + min_target_points:
                    exited = True
                    exit_price = entry_p + min_target_points
                    exit_reason = f"target_{min_target_points}pts"
                elif pos.get("current_sl") is not None and c_low <= pos["current_sl"]:
                    exited = True
                    exit_price = pos["current_sl"]
                    is_trailed = (trailing_stop_points and be_trigger_points and fav >= be_trigger_points)
                    exit_reason = "trailing_sl" if is_trailed else f"sl_{stop_loss_points}pts"
            else:  # SHORT
                if min_target_points is not None and c_low <= entry_p - min_target_points:
                    exited = True
                    exit_price = entry_p - min_target_points
                    exit_reason = f"target_{min_target_points}pts"
                elif pos.get("current_sl") is not None and c_high >= pos["current_sl"]:
                    exited = True
                    exit_price = pos["current_sl"]
                    is_trailed = (trailing_stop_points and be_trigger_points and fav >= be_trigger_points)
                    exit_reason = "trailing_sl" if is_trailed else f"sl_{stop_loss_points}pts"

            if exited and exit_price is not None:
                pnl = (exit_price - entry_p) if direction == "LONG" else (entry_p - exit_price)
                trades.append({
                    **pos,
                    "exit_date": c_date,
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "pnl_points": round(pnl, 2),
                    "bars_held": i - pos["entry_idx"],
                })
                pos = None

        # Crossover checks
        bullish_cross = (fp <= sp and f > s)
        bearish_cross = (fp >= sp and f < s)

        if bullish_cross:
            # If currently in SHORT, close on reverse cross
            if pos is not None and pos["direction"] == "SHORT":
                pnl = pos["entry_price"] - c_close
                trades.append({
                    **pos,
                    "exit_date": c_date,
                    "exit_price": c_close,
                    "exit_reason": "reverse_cross",
                    "pnl_points": round(pnl, 2),
                    "bars_held": i - pos["entry_idx"],
                })
                pos = None
            # Open LONG
            if pos is None:
                initial_sl = (c_close - stop_loss_points) if stop_loss_points is not None else None
                pos = {
                    "strategy": f"ema_{fast_period}_{slow_period}",
                    "direction": "LONG",
                    "entry_date": c_date,
                    "entry_price": c_close,
                    "entry_idx": i,
                    "current_sl": initial_sl,
                    "max_fav_pts": 0.0,
                    "max_adv_pts": 0.0,
                }
        elif bearish_cross:
            # If currently in LONG, close on reverse cross
            if pos is not None and pos["direction"] == "LONG":
                pnl = c_close - pos["entry_price"]
                trades.append({
                    **pos,
                    "exit_date": c_date,
                    "exit_price": c_close,
                    "exit_reason": "reverse_cross",
                    "pnl_points": round(pnl, 2),
                    "bars_held": i - pos["entry_idx"],
                })
                pos = None
            # Open SHORT
            if pos is None:
                initial_sl = (c_close + stop_loss_points) if stop_loss_points is not None else None
                pos = {
                    "strategy": f"ema_{fast_period}_{slow_period}",
                    "direction": "SHORT",
                    "entry_date": c_date,
                    "entry_price": c_close,
                    "entry_idx": i,
                    "current_sl": initial_sl,
                    "max_fav_pts": 0.0,
                    "max_adv_pts": 0.0,
                }

    return trades


def _strat_bollinger(candles: List[Dict]) -> List[Dict]:
    closes = [c["close"] for c in candles]
    bb = calc_bollinger(closes, 20, 2.0)

    trades, pos = [], None
    for i in range(1, len(candles)):
        if bb["lower"][i] is None:
            continue
        p, dt = closes[i], candles[i]["date"]
        if pos is None and p < bb["lower"][i]:
            pos = {"entry_date": dt, "entry_price": p, "strategy": "bollinger", "direction": "LONG"}
        elif pos is not None and p > bb["middle"][i]:
            trades.append({**pos, "exit_date": dt, "exit_price": p})
            pos = None
    return trades


GOLD_STRATEGIES: Dict[str, Callable[[List[Dict]], List[Dict]]] = {
    "supertrend": _strat_supertrend,
    "donchian": _strat_donchian,
    "ema_cross": _strat_ema_cross,
    "ema_5_20": lambda c: _strat_ema_5_20(c, fast_period=5, slow_period=20),
    "bollinger": _strat_bollinger,
}


def _calculate_metrics(trades: List[Dict], initial_capital: float = 10000.0) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_points": 0.0,
            "avg_points_per_trade": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    capital = initial_capital
    peak = capital
    max_dd = 0.0
    returns = []
    winners = []
    losers = []
    point_deltas = []

    long_wins, long_losses = 0, 0
    short_wins, short_losses = 0, 0
    long_points, short_points = 0.0, 0.0

    spread_pts = 0.35  # Standard XM MT5 Gold spread (3.5 pips = $0.35 / oz)
    for t in trades:
        direction = t.get("direction", t.get("type", "LONG")).upper()
        if direction == "SHORT":
            pts = t["entry_price"] - t["exit_price"]
        else:
            pts = t["exit_price"] - t["entry_price"]

        net_pts = pts - spread_pts
        point_deltas.append(net_pts)

        if net_pts > 0:
            if direction == "SHORT":
                short_wins += 1
            else:
                long_wins += 1
        else:
            if direction == "SHORT":
                short_losses += 1
            else:
                long_losses += 1

        if direction == "SHORT":
            short_points += net_pts
        else:
            long_points += net_pts

        net_ret = net_pts / t["entry_price"]
        ret_pct = net_ret * 100.0

        capital *= (1.0 + net_ret)
        peak = max(peak, capital)
        max_dd = max(max_dd, ((peak - capital) / peak) * 100.0)
        returns.append(net_ret)

        if net_pts > 0:
            winners.append(ret_pct)
        else:
            losers.append(ret_pct)

    total_return = ((capital - initial_capital) / initial_capital) * 100.0
    wr = (len(winners) / len(trades)) * 100.0 if trades else 0.0
    gross_p = sum(winners)
    gross_l = abs(sum(losers))
    profit_factor = round(gross_p / gross_l, 2) if gross_l > 0 else float("inf")

    sharpe = 0.0
    if len(returns) > 1:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        if std_r > 0:
            sharpe = round((mean_r / std_r) * math.sqrt(252), 2)

    calmar = round(total_return / max_dd, 2) if max_dd > 0 else 0.0

    total_points = sum(point_deltas)
    long_total = long_wins + long_losses
    short_total = short_wins + short_losses

    return {
        "total_trades": len(trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": round(wr, 1),
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown_pct": round(-max_dd, 2),
        "total_points": round(total_points, 2),
        "avg_points_per_trade": round(total_points / len(trades), 2) if trades else 0.0,
        "max_win_points": round(max(point_deltas), 2) if point_deltas else 0.0,
        "max_loss_points": round(min(point_deltas), 2) if point_deltas else 0.0,
        "long_trades": long_total,
        "long_win_rate_pct": round((long_wins / long_total) * 100.0, 1) if long_total else 0.0,
        "long_points": round(long_points, 2),
        "short_trades": short_total,
        "short_win_rate_pct": round((short_wins / short_total) * 100.0, 1) if short_total else 0.0,
        "short_points": round(short_points, 2),
    }


def backtest_ema_crossover_strategy(
    symbol: str = "XAUUSD",
    fast_period: int = 5,
    slow_period: int = 20,
    min_target_points: Optional[float] = None,
    stop_loss_points: Optional[float] = None,
    be_trigger_points: Optional[float] = None,
    trailing_stop_points: Optional[float] = None,
    period: str = "1y",
    interval: str = "1h",
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """
    Exhaustive backtest for EMA Crossover (Fast & Slow) on Gold for both LONG and SHORT.
    Evaluates pure reversal trading, minimum target points capture, stop loss, breakeven, and trailing stops.
    """
    candles = fetch_gold_candles(symbol, period=period, interval=interval)
    if len(candles) < slow_period + 10:
        return {"error": f"Insufficient candles ({len(candles)}) for EMA {fast_period}/{slow_period}"}

    trades = _strat_ema_5_20(
        candles,
        min_target_points=min_target_points,
        stop_loss_points=stop_loss_points,
        fast_period=fast_period,
        slow_period=slow_period,
        be_trigger_points=be_trigger_points,
        trailing_stop_points=trailing_stop_points,
    )
    metrics = _calculate_metrics(trades, initial_capital=initial_capital)
    bnh_return = round(((candles[-1]["close"] - candles[0]["close"]) / candles[0]["close"]) * 100.0, 2)

    # Calculate MFE distribution (favorable excursion before reversal)
    pure_trades = trades if (min_target_points is None and be_trigger_points is None) else _strat_ema_5_20(
        candles, min_target_points=None, stop_loss_points=None, fast_period=fast_period, slow_period=slow_period
    )
    
    thresholds = [2, 3, 5, 8, 10, 15, 20, 30, 50]
    mfe_distribution = {}
    if pure_trades:
        for th in thresholds:
            hit = sum(1 for t in pure_trades if t.get("max_fav_pts", 0.0) >= th)
            mfe_distribution[f"{th}_pts"] = {
                "trades_reached": hit,
                "pct_reached": round((hit / len(pure_trades)) * 100.0, 1),
            }

    # Target points sensitivity grid
    sensitivity_grid = []
    if min_target_points is None and be_trigger_points is None:
        for target in [2, 3, 5, 8, 10, 15, 20]:
            tgt_trades = _strat_ema_5_20(
                candles, min_target_points=target, stop_loss_points=None, fast_period=fast_period, slow_period=slow_period
            )
            tgt_m = _calculate_metrics(tgt_trades, initial_capital=initial_capital)
            targets_hit = sum(1 for t in tgt_trades if t.get("exit_reason", "").startswith("target"))
            sensitivity_grid.append({
                "target_points": target,
                "win_rate_pct": tgt_m["win_rate_pct"],
                "targets_hit": targets_hit,
                "total_trades": tgt_m["total_trades"],
                "total_points": tgt_m["total_points"],
                "profit_factor": tgt_m["profit_factor"],
                "total_return_pct": tgt_m["total_return_pct"],
            })

    return {
        "symbol": symbol.upper(),
        "strategy": f"ema_{fast_period}_{slow_period}",
        "fast_ema": fast_period,
        "slow_ema": slow_period,
        "min_target_points": min_target_points,
        "stop_loss_points": stop_loss_points,
        "be_trigger_points": be_trigger_points,
        "trailing_stop_points": trailing_stop_points,
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "buy_and_hold_return_pct": bnh_return,
        "metrics": metrics,
        "mfe_distribution": mfe_distribution,
        "target_sensitivity_analysis": sensitivity_grid,
        "recent_trades": trades[-10:] if trades else [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def backtest_gold_strategy(
    symbol: str = "XAUUSD",
    strategy: str = "supertrend",
    period: str = "1y",
    interval: str = "1h",
    min_target_points: Optional[float] = None,
    stop_loss_points: Optional[float] = None,
) -> Dict[str, Any]:
    """Backtest a specific Gold strategy."""
    strat_key = strategy.lower().strip()
    if strat_key in ("ema_5_20", "ema_cross_5_20") or (min_target_points is not None and "ema" in strat_key):
        return backtest_ema_crossover_strategy(
            symbol=symbol,
            fast_period=5,
            slow_period=20,
            min_target_points=min_target_points,
            stop_loss_points=stop_loss_points,
            period=period,
            interval=interval,
        )

    if strat_key not in GOLD_STRATEGIES:
        return {"error": f"Unknown strategy '{strategy}'. Choose: {list(GOLD_STRATEGIES.keys())}"}

    candles = fetch_gold_candles(symbol, period=period, interval=interval)
    if len(candles) < 40:
        return {"error": f"Insufficient candles ({len(candles)})"}

    trades = GOLD_STRATEGIES[strat_key](candles)
    metrics = _calculate_metrics(trades)

    bnh_return = round(((candles[-1]["close"] - candles[0]["close"]) / candles[0]["close"]) * 100.0, 2)

    return {
        "symbol": symbol.upper(),
        "strategy": strat_key,
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "buy_and_hold_return_pct": bnh_return,
        "metrics": metrics,
        "recent_trades": trades[-5:],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def compare_gold_strategies(
    symbol: str = "XAUUSD",
    period: str = "1y",
    interval: str = "1h",
) -> Dict[str, Any]:
    """Run all Gold strategies and rank them on return, Sharpe, and drawdown."""
    candles = fetch_gold_candles(symbol, period=period, interval=interval)
    if len(candles) < 40:
        return {"error": f"Insufficient candles ({len(candles)})"}

    leaderboard = []
    for name, fn in GOLD_STRATEGIES.items():
        trades = fn(candles)
        m = _calculate_metrics(trades)
        leaderboard.append({
            "strategy": name,
            "total_return_pct": m["total_return_pct"],
            "win_rate_pct": m["win_rate_pct"],
            "total_trades": m["total_trades"],
            "profit_factor": m["profit_factor"],
            "sharpe_ratio": m["sharpe_ratio"],
            "max_drawdown_pct": m["max_drawdown_pct"],
        })

    leaderboard.sort(key=lambda x: x["total_return_pct"], reverse=True)
    for rank, item in enumerate(leaderboard, 1):
        item["rank"] = rank

    bnh = round(((candles[-1]["close"] - candles[0]["close"]) / candles[0]["close"]) * 100.0, 2)

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "winner": leaderboard[0]["strategy"] if leaderboard else None,
        "buy_and_hold_return_pct": bnh,
        "leaderboard": leaderboard,
    }
