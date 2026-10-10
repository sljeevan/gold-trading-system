#!/usr/bin/env python3
"""
Comprehensive Permutations and Combinations Optimization Engine for the Gold Compounding Ladder.
Evaluates:
1. Progression Archetypes: Discrete Matrix, Tiered Sprint, Dynamic Fractional, Streak-Adaptive, Surplus Ratchet
2. Reward-to-Risk Targets: 1.2R, 1.5R, 1.8R, 2.2R, 2.5R
3. Volatility Stop Multipliers: 1.0x ATR, 1.5x ATR, 2.0x ATR
4. Starting Seeds: $20, $50, $100, $250, $500
5. Loss Policies & Protections: Unprotected, Milestone Floors, Progressive Ratchets, Trailing DD Floor, Circuit Breakers
6. Signal Engines: 1H Donchian 15 Breakout + SuperTrend, 1H SuperTrend 1.2R, Dual-Engine Consensus
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Adjust path to find gold_trader package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.compounding_ladder import LADDER_MATRIX, get_ladder_level_info, find_closest_level_for_balance
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_atr, calc_donchian, calc_ema, calc_supertrend

SPREAD_USD = 0.35  # XM broker spread: 3.5 pips ($0.35/oz)
MIN_LOT = 0.01
MAX_LOT = 50.0
LOT_STEP = 0.01
CONTRACT_SIZE = 100.0


@dataclass
class TradeSignal:
    candle_idx: int
    date: str
    direction: str  # BUY / SELL
    entry_price: float
    atr: float
    engine: str


@dataclass
class TradeOutcome:
    signal: TradeSignal
    outcome: str  # WIN, LOSS, SCRATCH
    r_multiple: float  # e.g. +1.8, -1.0, 0.0
    sl_points: float
    tp_points: float
    bars_held: int


def generate_all_trade_outcomes(
    candles_1h: List[Dict[str, Any]],
    engine_name: str = "DONCHIAN_HYBRID",
    atr_mult: float = 1.5,
    min_sl_pts: float = 8.0,
    rr_ratio: float = 1.8,
) -> List[TradeOutcome]:
    """
    Simulates trades with strict conservative intrabar execution (worst-case adverse check first).
    Returns list of evaluated TradeOutcome events.
    """
    n = len(candles_1h)
    highs = np.array([float(c["high"]) for c in candles_1h])
    lows = np.array([float(c["low"]) for c in candles_1h])
    closes = np.array([float(c["close"]) for c in candles_1h])
    opens = np.array([float(c["open"]) for c in candles_1h])
    dates = [c["date"] for c in candles_1h]

    # Precalculate indicators
    donch15 = calc_donchian(highs.tolist(), lows.tolist(), 15)
    st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
    ema200 = calc_ema(closes.tolist(), 200)
    atr14 = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)

    outcomes: List[TradeOutcome] = []
    active_until_idx = -1

    for i in range(200, n - 48):
        if i <= active_until_idx:
            continue

        dt_str = dates[i]
        hour = int(dt_str.split("T" if "T" in dt_str else " ")[1].split(":")[0])
        # Liquid London / NY session filter (07:00 - 19:00 UTC)
        if not (7 <= hour <= 19):
            continue

        d_up = donch15["upper"][i - 1]
        d_dn = donch15["lower"][i - 1]
        st_dir = st["direction"][i]
        e200 = ema200[i]
        atr_val = atr14[i]

        if None in (d_up, d_dn, st_dir, e200, atr_val):
            continue

        c = closes[i]
        dir_ = None

        if engine_name == "DONCHIAN_HYBRID":
            if c > d_up and st_dir == 1 and c > e200:
                dir_ = "BUY"
            elif c < d_dn and st_dir == -1 and c < e200:
                dir_ = "SELL"
        elif engine_name == "SUPERTREND_ONLY":
            prev_st = st["direction"][i - 1]
            if st_dir == 1 and prev_st == -1 and c > e200:
                dir_ = "BUY"
            elif st_dir == -1 and prev_st == 1 and c < e200:
                dir_ = "SELL"

        if not dir_:
            continue

        # Stop loss & take profit distances
        sl_points = max(atr_mult * atr_val, min_sl_pts)
        tp_points = rr_ratio * sl_points

        eff_entry = c + SPREAD_USD if dir_ == "BUY" else c
        tp_price = eff_entry + tp_points if dir_ == "BUY" else eff_entry - tp_points - SPREAD_USD
        sl_price = eff_entry - sl_points if dir_ == "BUY" else eff_entry + sl_points + SPREAD_USD
        be_trigger = eff_entry + sl_points if dir_ == "BUY" else eff_entry - sl_points

        outcome = "LOSS"
        r_mult = -1.0
        is_be = False
        bars_held = 0

        # Conservative Intrabar pathing: Check adverse move first
        for k in range(i + 1, min(i + 49, n)):
            bars_held = k - i
            o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
            curr_sl = eff_entry if is_be else sl_price

            if dir_ == "BUY":
                if l_k <= curr_sl:
                    outcome = "SCRATCH" if is_be else "LOSS"
                    r_mult = 0.0 if is_be else -1.0
                    break
                if h_k >= be_trigger:
                    is_be = True
                if h_k >= tp_price:
                    outcome = "WIN"
                    r_mult = rr_ratio
                    break
            else:
                if h_k >= curr_sl:
                    outcome = "SCRATCH" if is_be else "LOSS"
                    r_mult = 0.0 if is_be else -1.0
                    break
                if l_k <= be_trigger:
                    is_be = True
                if l_k <= tp_price:
                    outcome = "WIN"
                    r_mult = rr_ratio
                    break

        active_until_idx = i + bars_held

        signal = TradeSignal(
            candle_idx=i,
            date=dt_str,
            direction=dir_,
            entry_price=eff_entry,
            atr=atr_val,
            engine=engine_name,
        )
        outcomes.append(
            TradeOutcome(
                signal=signal,
                outcome=outcome,
                r_multiple=r_mult,
                sl_points=sl_points,
                tp_points=tp_points,
                bars_held=bars_held,
            )
        )

    return outcomes


def clamp_lots(lots: float) -> float:
    rounded = math.floor(lots * 100.0 + 0.0001) / 100.0
    return max(MIN_LOT, min(MAX_LOT, rounded))


# ==============================================================================
# COMPOUNDING SIMULATION CORE
# ==============================================================================
def simulate_compounding_ladder(
    trade_outcomes: List[TradeOutcome],
    starting_balance: float = 100.0,
    target_balance: float = 5000.0,
    archetype: str = "TIERED_PHASED_SPRINT",
    base_risk_pct: float = 8.0,
    loss_policy: str = "PROGRESSIVE_RATCHET",
    enable_circuit_breaker: bool = True,
    trailing_dd_floor_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes a complete compounding sequence for a specified permutation.
    """
    balance = starting_balance
    peak_balance = balance
    ratchet_floor = 0.0
    level = find_closest_level_for_balance(balance)
    peak_level = level

    wins, losses, scratches = 0, 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    reloads = 0
    consecutive_losses = 0
    consecutive_wins = 0
    max_consecutive_losses = 0
    max_consecutive_wins = 0

    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0
    pnl_history: List[float] = []
    equity_curve: List[float] = [balance]

    milestones_reached: Dict[str, Any] = {}
    target_hit = False
    target_trade_no = None
    target_date = None

    for idx, trade in enumerate(trade_outcomes):
        trade_no = idx + 1

        # Check circuit breaker: after 2 consecutive losses, reduce risk by 50%
        circuit_breaker_active = enable_circuit_breaker and (consecutive_losses >= 2)

        # 1. Update Floor protections
        if loss_policy == "PROGRESSIVE_RATCHET":
            if balance >= 4000.0:
                ratchet_floor = max(ratchet_floor, 3000.0)
            elif balance >= 2500.0:
                ratchet_floor = max(ratchet_floor, 1800.0)
            elif balance >= 1000.0:
                ratchet_floor = max(ratchet_floor, 700.0)
            elif balance >= 500.0:
                ratchet_floor = max(ratchet_floor, 350.0)
            elif balance >= 250.0:
                ratchet_floor = max(ratchet_floor, 150.0)
        elif loss_policy == "PRESERVE_MILESTONES":
            if balance >= 2900.0:
                ratchet_floor = max(ratchet_floor, 2900.0)
            elif balance >= 200.0:
                ratchet_floor = max(ratchet_floor, 200.0)
        elif loss_policy == "TRAILING_DD_FLOOR" and trailing_dd_floor_pct is not None:
            ratchet_floor = max(ratchet_floor, peak_balance * (1.0 - (trailing_dd_floor_pct / 100.0)))
        else:
            ratchet_floor = 0.0

        # Available risk capital
        if ratchet_floor > 0.0:
            risk_capital = max(balance - ratchet_floor, balance * 0.35)
        else:
            risk_capital = balance

        # 2. Determine Position Sizing by Archetype
        sl_points = trade.sl_points

        if archetype == "DISCRETE_30_LEVEL":
            lvl_info = get_ladder_level_info(level)
            planned_risk_usd = lvl_info["risk_dollars"]
            lots = clamp_lots(planned_risk_usd / (sl_points * CONTRACT_SIZE))

        elif archetype == "TIERED_PHASED_SPRINT":
            # Phase 1: Sprint ($100-$500) -> 10%
            # Phase 2: Expansion ($500-$2000) -> 8%
            # Phase 3: Preservation ($2000-$5000+) -> 5%
            if balance < 500.0:
                active_pct = 0.10
            elif balance < 2000.0:
                active_pct = 0.08
            else:
                active_pct = 0.05

            if circuit_breaker_active:
                active_pct *= 0.50

            planned_risk_usd = risk_capital * active_pct
            lots = clamp_lots(planned_risk_usd / (sl_points * CONTRACT_SIZE))

        elif archetype == "DYNAMIC_EQUITY_PERCENT":
            active_pct = (base_risk_pct / 100.0)
            if circuit_breaker_active:
                active_pct *= 0.50
            planned_risk_usd = risk_capital * active_pct
            lots = clamp_lots(planned_risk_usd / (sl_points * CONTRACT_SIZE))

        elif archetype == "STREAK_ADAPTIVE":
            # Base risk +1.5% per consecutive win (capped at 12%), drop to 3.5% after loss
            if consecutive_losses > 0:
                active_pct = 0.035
            else:
                active_pct = min(0.12, 0.06 + (consecutive_wins * 0.015))
            planned_risk_usd = risk_capital * active_pct
            lots = clamp_lots(planned_risk_usd / (sl_points * CONTRACT_SIZE))

        elif archetype == "HIGH_WATERMARK_RATCHET":
            # Risk 15% of surplus above floor, minimum 4% of balance
            surplus = max(0.0, balance - ratchet_floor)
            planned_risk_usd = max(surplus * 0.15, balance * 0.04)
            if circuit_breaker_active:
                planned_risk_usd *= 0.50
            lots = clamp_lots(planned_risk_usd / (sl_points * CONTRACT_SIZE))

        else:
            lots = 0.01

        dollar_per_point = lots * CONTRACT_SIZE

        # 3. Calculate Realized P&L
        if trade.outcome == "WIN":
            realized_pnl = trade.tp_points * dollar_per_point
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            gross_profit += realized_pnl
            balance += realized_pnl

            if archetype == "DISCRETE_30_LEVEL":
                level = min(30, level + 1)
                peak_level = max(peak_level, level)

        elif trade.outcome == "SCRATCH":
            realized_pnl = 0.0
            scratches += 1

        else:  # LOSS
            realized_pnl = -(sl_points * dollar_per_point)
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            gross_loss += abs(realized_pnl)

            # Apply floor bound
            balance = max(ratchet_floor, balance + realized_pnl)

            # Handle step down
            if archetype == "DISCRETE_30_LEVEL":
                if loss_policy == "ONE_STEP_LADDER":
                    level = max(1, level - 1)
                elif loss_policy == "PRESERVE_MILESTONES":
                    if level >= 20:
                        level = 20
                    elif level >= 10:
                        level = 10
                    else:
                        level = 1
                elif loss_policy == "STEP_DOWN_BY_BALANCE":
                    level = find_closest_level_for_balance(balance)

            # Check reload condition (account wiped below margin floor)
            if balance < 10.0:
                reloads += 1
                balance = starting_balance
                ratchet_floor = 0.0
                level = 1

        pnl_history.append(realized_pnl)
        equity_curve.append(balance)
        peak_balance = max(peak_balance, balance)

        # Track Drawdown
        current_dd_usd = peak_balance - balance
        current_dd_pct = (current_dd_usd / peak_balance * 100.0) if peak_balance > 0 else 0.0
        max_drawdown_usd = max(max_drawdown_usd, current_dd_usd)
        max_drawdown_pct = max(max_drawdown_pct, current_dd_pct)

        # Check Milestones
        for m in [200, 350, 500, 700, 1000, 1800, 2500, 3500, 5000, 10000, 25000, 50000]:
            k_m = str(m)
            if balance >= m and k_m not in milestones_reached:
                milestones_reached[k_m] = {
                    "trade_no": trade_no,
                    "date": trade.signal.date,
                    "balance": round(balance, 2),
                }

        if balance >= target_balance and not target_hit:
            target_hit = True
            target_trade_no = trade_no
            target_date = trade.signal.date

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Calculate Sharpe & Sortino
    if len(pnl_history) > 1:
        returns = np.array(pnl_history)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (mean_ret / std_ret * math.sqrt(100)) if std_ret > 0 else 0.0

        downside = returns[returns < 0]
        std_downside = np.std(downside) if len(downside) > 0 else 0.0
        sortino = (mean_ret / std_downside * math.sqrt(100)) if std_downside > 0 else 0.0
    else:
        sharpe, sortino = 0.0, 0.0

    return {
        "starting_balance": starting_balance,
        "target_balance": target_balance,
        "final_balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "target_hit": target_hit,
        "target_trade_no": target_trade_no,
        "target_date": target_date,
        "total_trades": len(trade_outcomes),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 1),
        "reloads": reloads,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "milestones": milestones_reached,
    }


# ==============================================================================
# MONTE CARLO STRESS TEST
# ==============================================================================
def run_monte_carlo_resampling(
    trade_outcomes: List[TradeOutcome],
    config_params: Dict[str, Any],
    num_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Shuffles trade order across 1,000 to 5,000 iterations to measure:
    - Bust rate (% chance of ruin)
    - Target completion rate (% of sequences hitting target)
    - Median trades to target
    - 95th percentile max drawdown
    """
    n_trades = len(trade_outcomes)
    completions = 0
    busts = 0
    trades_to_target_list: List[int] = []
    max_dds: List[float] = []
    final_balances: List[float] = []

    rng = np.random.default_rng(seed=42)

    for _ in range(num_simulations):
        # Sample with replacement (bootstrap) to simulate varying market regimes
        sampled_indices = rng.choice(n_trades, size=n_trades, replace=True)
        shuffled_trades = [trade_outcomes[idx] for idx in sampled_indices]

        res = simulate_compounding_ladder(
            shuffled_trades,
            starting_balance=config_params.get("starting_balance", 100.0),
            target_balance=config_params.get("target_balance", 5000.0),
            archetype=config_params.get("archetype", "TIERED_PHASED_SPRINT"),
            base_risk_pct=config_params.get("base_risk_pct", 8.0),
            loss_policy=config_params.get("loss_policy", "PROGRESSIVE_RATCHET"),
            enable_circuit_breaker=config_params.get("enable_circuit_breaker", True),
            trailing_dd_floor_pct=config_params.get("trailing_dd_floor_pct", None),
        )

        if res["target_hit"]:
            completions += 1
            if res["target_trade_no"]:
                trades_to_target_list.append(res["target_trade_no"])

        if res["reloads"] > 0 or res["final_balance"] < 10.0:
            busts += 1

        max_dds.append(res["max_drawdown_pct"])
        final_balances.append(res["final_balance"])

    completion_rate = round((completions / num_simulations) * 100.0, 1)
    bust_rate = round((busts / num_simulations) * 100.0, 1)
    median_trades = int(np.median(trades_to_target_list)) if trades_to_target_list else None
    p95_dd = round(float(np.percentile(max_dds, 95)), 1)
    median_final_bal = round(float(np.median(final_balances)), 2)

    return {
        "simulations": num_simulations,
        "completion_rate_pct": completion_rate,
        "bust_rate_pct": bust_rate,
        "median_trades_to_target": median_trades,
        "p95_max_drawdown_pct": p95_dd,
        "median_final_balance": median_final_bal,
    }


def main():
    print("=" * 100)
    print("🔬 COMPREHENSIVE COMPOUNDING LADDER PERMUTATIONS & COMBINATIONS OPTIMIZER")
    print("=" * 100)

    # 1. Acquire Real Market Data from XM MT5 or Yahoo
    print("Fetching hourly gold candles...")
    candles_1h = None
    try:
        broker = SocketBroker(timeout=10.0)
        if broker.connect():
            candles_1h = broker.get_rates("XAUUSD", "1h", 5000)
            print(f"Loaded {len(candles_1h)} candles from XM MT5 Bridge.")
    except Exception as e:
        print(f"MT5 bridge read note: {e}")

    if not candles_1h or len(candles_1h) < 1000:
        print("Falling back to 2-year hourly Yahoo Finance feed...")
        candles_1h = fetch_gold_candles("XAUUSD", period="2y", interval="1h")
        print(f"Loaded {len(candles_1h)} candles from Yahoo Finance.")

    print(f"Candle Date Range: {candles_1h[0]['date']} to {candles_1h[-1]['date']}")
    print("=" * 100)

    # 2. Generate Base Trade Signal Sequences across R:R and ATR parameter combinations
    print("\nPre-generating base market trade outcomes across strategy & bracket parameter spaces...")
    trade_cache: Dict[Tuple[str, float, float], List[TradeOutcome]] = {}

    engines = ["DONCHIAN_HYBRID", "SUPERTREND_ONLY"]
    rr_ratios = [1.2, 1.5, 1.8, 2.2]
    atr_multipliers = [1.0, 1.5, 2.0]

    for eng in engines:
        for rr in rr_ratios:
            for atr_m in atr_multipliers:
                key = (eng, rr, atr_m)
                outcomes = generate_all_trade_outcomes(
                    candles_1h,
                    engine_name=eng,
                    atr_mult=atr_m,
                    min_sl_pts=8.0,
                    rr_ratio=rr,
                )
                trade_cache[key] = outcomes

    print(f"Cached {len(trade_cache)} distinct market trade execution universes.")

    # 3. Define the Permutation Matrix
    seeds = [20.0, 50.0, 100.0, 250.0, 500.0]
    archetypes = [
        "DISCRETE_30_LEVEL",
        "TIERED_PHASED_SPRINT",
        "DYNAMIC_EQUITY_PERCENT",
        "STREAK_ADAPTIVE",
        "HIGH_WATERMARK_RATCHET",
    ]
    loss_policies = [
        "ONE_STEP_LADDER",
        "PRESERVE_MILESTONES",
        "PROGRESSIVE_RATCHET",
        "TRAILING_DD_FLOOR",
    ]

    permutations_to_test = [
        # --- Group 1: Micro-Seed ($20) Comparisons ---
        {
            "name": "P01: Classic Spreadsheet ($20, Discrete 30, Unprotected, 1.8R)",
            "seed": 20.0,
            "target": 52384.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "DISCRETE_30_LEVEL",
            "risk_pct": 23.0,
            "loss_policy": "ONE_STEP_LADDER",
            "circuit_breaker": False,
        },
        {
            "name": "P02: $20 Seed + Milestone Preservation + 1.2R Fast Target",
            "seed": 20.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.2,
            "atr_m": 1.5,
            "archetype": "DISCRETE_30_LEVEL",
            "risk_pct": 23.0,
            "loss_policy": "PRESERVE_MILESTONES",
            "circuit_breaker": True,
        },
        {
            "name": "P03: $20 Seed + Dynamic Equity (8% Fixed) + 1.8R",
            "seed": 20.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "DYNAMIC_EQUITY_PERCENT",
            "risk_pct": 8.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },

        # --- Group 2: $100 Seed - Sizing Archetype Head-to-Head ---
        {
            "name": "P04: $100 Seed + Tiered Phased Sprint (10%->8%->5%) + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P05: $100 Seed + Tiered Phased Sprint + Fast 1.2R Target",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.2,
            "atr_m": 1.5,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P06: $100 Seed + Tiered Phased Sprint + 2.2R Scale-Out Runner",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 2.2,
            "atr_m": 1.5,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P07: $100 Seed + Dynamic Fixed Equity (5% Safe) + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "DYNAMIC_EQUITY_PERCENT",
            "risk_pct": 5.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P08: $100 Seed + Dynamic Fixed Equity (10% Aggressive) + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "DYNAMIC_EQUITY_PERCENT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P09: $100 Seed + Streak-Adaptive Anti-Martingale (6%-12%) + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "STREAK_ADAPTIVE",
            "risk_pct": 6.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P10: $100 Seed + High-Watermark Surplus Ratchet + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "HIGH_WATERMARK_RATCHET",
            "risk_pct": 15.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },

        # --- Group 3: Stop-Loss Distance Impact (ATR Multipliers) ---
        {
            "name": "P11: $100 Seed + 1.0x ATR Stop (Tight) + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.0,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P12: $100 Seed + 2.0x ATR Stop (Wide/Ultra-Safe) + 1.8R",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 2.0,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },

        # --- Group 4: SuperTrend Tournament-Winning Engine ---
        {
            "name": "P13: $100 Seed + SuperTrend Engine + Fast 1.2R Target",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "SUPERTREND_ONLY",
            "rr": 1.2,
            "atr_m": 1.5,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P14: $100 Seed + SuperTrend Engine + 1.8R Target",
            "seed": 100.0,
            "target": 5000.0,
            "engine": "SUPERTREND_ONLY",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "TIERED_PHASED_SPRINT",
            "risk_pct": 10.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },

        # --- Group 5: Larger Seed Capital Preservation ($500 Seed) ---
        {
            "name": "P15: $500 Seed + Dynamic Equity (5% Institutional) + 1.8R",
            "seed": 500.0,
            "target": 10000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "DYNAMIC_EQUITY_PERCENT",
            "risk_pct": 5.0,
            "loss_policy": "PROGRESSIVE_RATCHET",
            "circuit_breaker": True,
        },
        {
            "name": "P16: $500 Seed + Trailing 25% Drawdown Floor + 1.8R",
            "seed": 500.0,
            "target": 10000.0,
            "engine": "DONCHIAN_HYBRID",
            "rr": 1.8,
            "atr_m": 1.5,
            "archetype": "DYNAMIC_EQUITY_PERCENT",
            "risk_pct": 6.0,
            "loss_policy": "TRAILING_DD_FLOOR",
            "circuit_breaker": True,
            "trailing_dd_floor_pct": 25.0,
        },
    ]

    print(f"\nRunning backtest simulation across {len(permutations_to_test)} permutations...")

    results = []

    for perm in permutations_to_test:
        key = (perm["engine"], perm["rr"], perm["atr_m"])
        trades = trade_cache[key]

        res = simulate_compounding_ladder(
            trades,
            starting_balance=perm["seed"],
            target_balance=perm["target"],
            archetype=perm["archetype"],
            base_risk_pct=perm["risk_pct"],
            loss_policy=perm["loss_policy"],
            enable_circuit_breaker=perm["circuit_breaker"],
            trailing_dd_floor_pct=perm.get("trailing_dd_floor_pct", None),
        )

        # Run Monte Carlo Stress Test (1,000 resampled iterations)
        mc_res = run_monte_carlo_resampling(
            trades,
            config_params={
                "starting_balance": perm["seed"],
                "target_balance": perm["target"],
                "archetype": perm["archetype"],
                "base_risk_pct": perm["risk_pct"],
                "loss_policy": perm["loss_policy"],
                "enable_circuit_breaker": perm["circuit_breaker"],
                "trailing_dd_floor_pct": perm.get("trailing_dd_floor_pct", None),
            },
            num_simulations=1000,
        )

        # Composite Optimization Fitness Score:
        # Balances Profit Factor, Maximum Drawdown resistance, Survival (1 - Bust), and Growth
        growth_multiplier = max(0.01, res["final_balance"] / perm["seed"])
        dd_factor = max(0.05, 1.0 - (res["max_drawdown_pct"] / 100.0))
        survival_factor = max(0.01, 1.0 - (mc_res["bust_rate_pct"] / 100.0))
        fitness_score = round(
            res["profit_factor"] * dd_factor * survival_factor * math.log10(growth_multiplier + 1.0) * 10.0,
            2,
        )

        perm_record = {
            "name": perm["name"],
            "seed": perm["seed"],
            "target": perm["target"],
            "archetype": perm["archetype"],
            "engine": perm["engine"],
            "rr": perm["rr"],
            "atr_multiplier": perm["atr_m"],
            "risk_pct": perm["risk_pct"],
            "loss_policy": perm["loss_policy"],
            "circuit_breaker": perm["circuit_breaker"],
            "final_balance": res["final_balance"],
            "peak_balance": res["peak_balance"],
            "win_rate_pct": res["win_rate_pct"],
            "profit_factor": res["profit_factor"],
            "max_drawdown_pct": res["max_drawdown_pct"],
            "sharpe_ratio": res["sharpe_ratio"],
            "sortino_ratio": res["sortino_ratio"],
            "target_hit": res["target_hit"],
            "target_trade_no": res["target_trade_no"],
            "reloads": res["reloads"],
            "mc_completion_rate_pct": mc_res["completion_rate_pct"],
            "mc_bust_rate_pct": mc_res["bust_rate_pct"],
            "mc_median_trades": mc_res["median_trades_to_target"],
            "mc_p95_drawdown_pct": mc_res["p95_max_drawdown_pct"],
            "fitness_score": fitness_score,
            "milestones": res["milestones"],
        }
        results.append(perm_record)
        print(f"  ✓ Finished: {perm['name']:<65} | PF: {res['profit_factor']:>4.2f} | Final: ${res['final_balance']:>9,.2f} | Fitness: {fitness_score:>5.2f}")

    # Sort results by composite fitness score descending
    results.sort(key=lambda x: x["fitness_score"], reverse=True)

    # Save to JSON
    out_dir = os.path.expanduser("~/.gold_trading_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "compounding_permutations_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 130)
    print("🏆 FINAL RANKING: COMPOUNDING LADDER PERMUTATIONS (ORDERED BY FITNESS SCORE)")
    print("=" * 130)
    header = (
        f"{'Rank':<4} | {'Permutation Name':<48} | {'Seed':<5} | {'Final Bal ($)':<12} | "
        f"{'PF':<5} | {'WR%':<5} | {'Max DD%':<7} | {'MC Bust%':<8} | {'MC Comp%':<8} | {'Score'}"
    )
    print(header)
    print("-" * 130)

    for i, r in enumerate(results, 1):
        name = r["name"][:48]
        seed = f"${r['seed']:,.0f}"
        f_bal = f"${r['final_balance']:,.2f}"
        pf = f"{r['profit_factor']:.2f}"
        wr = f"{r['win_rate_pct']:.1f}%"
        dd = f"{r['max_drawdown_pct']:.1f}%"
        bust = f"{r['mc_bust_rate_pct']:.1f}%"
        comp = f"{r['mc_completion_rate_pct']:.1f}%"
        score = f"{r['fitness_score']:.2f}"
        print(f"{i:<4} | {name:<48} | {seed:<5} | {f_bal:<12} | {pf:<5} | {wr:<5} | {dd:<7} | {bust:<8} | {comp:<8} | {score}")

    print("=" * 130)
    print(f"\nFull permutation analytics persisted to: {out_path}")


if __name__ == "__main__":
    main()
