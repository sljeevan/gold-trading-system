"""
30-Level Exponential Compounding Ladder for Gold (XAUUSD).
Based directly on the 30-Level Gold Trading Plan ($20 -> $52,384) with comprehensive
support for advanced permutations and combinations:
- Discrete 30-Level Ladder (Classic)
- Tiered Phased Sprint (Sprint -> Expansion -> Preservation)
- Dynamic Equity Fractional (Fixed % / Kelly sizing)
- Streak-Adaptive Sizing (Anti-Martingale with cooldown circuit breaker)
- High-Watermark Surplus Ratchet (Guaranteed floor protection)

Key Parameters per Stage:
- Take Profit (TP): Fixed 20 pips ($2.00/oz) or dynamic ATR multiplier
- Stop Loss (SL): Fixed ~15.0 - 16.4 pips ($1.50 - $1.64/oz) or dynamic ATR multiplier
- Risk per Trade: Configurable per archetype (2% to 23%)
- Target Gain: Configurable per archetype (1.2R to 2.5R)
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger("CompoundingLadder")

# Canonical 30-Level Compounding Matrix from the verified trading plan
LADDER_MATRIX: List[Dict[str, Any]] = [
    {"level": 1, "starting_balance": 20.0, "risk_pct": 23.0, "risk_dollars": 5.0, "profit_pct": 30.0, "profit_dollars": 6.0, "sl_pips": 15.0, "tp_pips": 20.0, "lot_size": 0.03, "ending_balance": 26.0},
    {"level": 2, "starting_balance": 26.0, "risk_pct": 23.0, "risk_dollars": 6.0, "profit_pct": 31.0, "profit_dollars": 8.0, "sl_pips": 15.0, "tp_pips": 20.0, "lot_size": 0.04, "ending_balance": 34.0},
    {"level": 3, "starting_balance": 34.0, "risk_pct": 24.0, "risk_dollars": 8.0, "profit_pct": 29.0, "profit_dollars": 10.0, "sl_pips": 16.0, "tp_pips": 20.0, "lot_size": 0.05, "ending_balance": 44.0},
    {"level": 4, "starting_balance": 44.0, "risk_pct": 23.0, "risk_dollars": 10.0, "profit_pct": 32.0, "profit_dollars": 14.0, "sl_pips": 14.3, "tp_pips": 20.0, "lot_size": 0.07, "ending_balance": 58.0},
    {"level": 5, "starting_balance": 58.0, "risk_pct": 24.0, "risk_dollars": 14.0, "profit_pct": 31.0, "profit_dollars": 18.0, "sl_pips": 15.6, "tp_pips": 20.0, "lot_size": 0.09, "ending_balance": 76.0},
    {"level": 6, "starting_balance": 76.0, "risk_pct": 24.0, "risk_dollars": 18.0, "profit_pct": 29.0, "profit_dollars": 22.0, "sl_pips": 16.4, "tp_pips": 20.0, "lot_size": 0.11, "ending_balance": 98.0},
    {"level": 7, "starting_balance": 98.0, "risk_pct": 22.0, "risk_dollars": 22.0, "profit_pct": 29.0, "profit_dollars": 28.0, "sl_pips": 15.7, "tp_pips": 20.0, "lot_size": 0.14, "ending_balance": 126.0},
    {"level": 8, "starting_balance": 126.0, "risk_pct": 22.0, "risk_dollars": 28.0, "profit_pct": 30.0, "profit_dollars": 38.0, "sl_pips": 14.7, "tp_pips": 20.0, "lot_size": 0.19, "ending_balance": 164.0},
    {"level": 9, "starting_balance": 164.0, "risk_pct": 23.0, "risk_dollars": 38.0, "profit_pct": 29.0, "profit_dollars": 48.0, "sl_pips": 15.8, "tp_pips": 20.0, "lot_size": 0.24, "ending_balance": 212.0},
    {"level": 10, "starting_balance": 212.0, "risk_pct": 23.0, "risk_dollars": 49.0, "profit_pct": 30.0, "profit_dollars": 64.0, "sl_pips": 15.0, "tp_pips": 20.0, "lot_size": 0.32, "ending_balance": 276.0},
    {"level": 11, "starting_balance": 276.0, "risk_pct": 23.0, "risk_dollars": 63.0, "profit_pct": 30.0, "profit_dollars": 83.0, "sl_pips": 15.6, "tp_pips": 20.0, "lot_size": 0.41, "ending_balance": 358.0},
    {"level": 12, "starting_balance": 358.0, "risk_pct": 23.0, "risk_dollars": 82.0, "profit_pct": 30.0, "profit_dollars": 107.0, "sl_pips": 15.2, "tp_pips": 20.0, "lot_size": 0.54, "ending_balance": 466.0},
    {"level": 13, "starting_balance": 466.0, "risk_pct": 23.0, "risk_dollars": 107.0, "profit_pct": 30.0, "profit_dollars": 140.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 0.70, "ending_balance": 606.0},
    {"level": 14, "starting_balance": 606.0, "risk_pct": 23.0, "risk_dollars": 139.0, "profit_pct": 30.0, "profit_dollars": 182.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 0.91, "ending_balance": 787.0},
    {"level": 15, "starting_balance": 787.0, "risk_pct": 23.0, "risk_dollars": 181.0, "profit_pct": 30.0, "profit_dollars": 236.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 1.18, "ending_balance": 1023.0},
    {"level": 16, "starting_balance": 1023.0, "risk_pct": 23.0, "risk_dollars": 235.0, "profit_pct": 30.0, "profit_dollars": 307.0, "sl_pips": 15.3, "tp_pips": 20.0, "lot_size": 1.54, "ending_balance": 1330.0},
    {"level": 17, "starting_balance": 1330.0, "risk_pct": 23.0, "risk_dollars": 306.0, "profit_pct": 30.0, "profit_dollars": 399.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 2.00, "ending_balance": 1730.0},
    {"level": 18, "starting_balance": 1730.0, "risk_pct": 23.0, "risk_dollars": 398.0, "profit_pct": 30.0, "profit_dollars": 519.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 2.60, "ending_balance": 2248.0},
    {"level": 19, "starting_balance": 2248.0, "risk_pct": 23.0, "risk_dollars": 517.0, "profit_pct": 30.0, "profit_dollars": 675.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 3.37, "ending_balance": 2923.0},
    {"level": 20, "starting_balance": 2923.0, "risk_pct": 23.0, "risk_dollars": 672.0, "profit_pct": 30.0, "profit_dollars": 877.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 4.39, "ending_balance": 3800.0},
    {"level": 21, "starting_balance": 3800.0, "risk_pct": 23.0, "risk_dollars": 874.0, "profit_pct": 30.0, "profit_dollars": 1140.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 5.70, "ending_balance": 4940.0},
    {"level": 22, "starting_balance": 4940.0, "risk_pct": 23.0, "risk_dollars": 1136.0, "profit_pct": 30.0, "profit_dollars": 1482.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 7.41, "ending_balance": 6422.0},
    {"level": 23, "starting_balance": 6422.0, "risk_pct": 23.0, "risk_dollars": 1477.0, "profit_pct": 30.0, "profit_dollars": 1927.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 9.64, "ending_balance": 8348.0},
    {"level": 24, "starting_balance": 8348.0, "risk_pct": 23.0, "risk_dollars": 1920.0, "profit_pct": 30.0, "profit_dollars": 2504.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 12.53, "ending_balance": 10853.0},
    {"level": 25, "starting_balance": 10853.0, "risk_pct": 23.0, "risk_dollars": 2496.0, "profit_pct": 30.0, "profit_dollars": 3256.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 16.28, "ending_balance": 14109.0},
    {"level": 26, "starting_balance": 14109.0, "risk_pct": 23.0, "risk_dollars": 3245.0, "profit_pct": 30.0, "profit_dollars": 4233.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 21.17, "ending_balance": 18341.0},
    {"level": 27, "starting_balance": 18341.0, "risk_pct": 23.0, "risk_dollars": 4218.0, "profit_pct": 30.0, "profit_dollars": 5502.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 27.52, "ending_balance": 23844.0},
    {"level": 28, "starting_balance": 23844.0, "risk_pct": 23.0, "risk_dollars": 5484.0, "profit_pct": 30.0, "profit_dollars": 7153.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 35.78, "ending_balance": 30997.0},
    {"level": 29, "starting_balance": 30997.0, "risk_pct": 23.0, "risk_dollars": 7129.0, "profit_pct": 30.0, "profit_dollars": 9299.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 46.51, "ending_balance": 40296.0},
    {"level": 30, "starting_balance": 40296.0, "risk_pct": 23.0, "risk_dollars": 9268.0, "profit_pct": 30.0, "profit_dollars": 12089.0, "sl_pips": 15.4, "tp_pips": 20.0, "lot_size": 60.46, "ending_balance": 52384.0},
]

MIN_LOT = 0.01
MAX_LOT = 50.0
CONTRACT_SIZE = 100.0


# Noise-Adaptive Matrix for Gold (XAUUSD)
# Engineered specifically around Gold's intraday trading noise ($8.00 minimum ATR stop)
# Starting balance: $100.00 | Target balance: $5,337.29
# Risk: 10% in Sprint ($100-$500), 8% in Expansion ($500-$2,000), 5% in Preservation ($2,000+)
# Target: 1.5R (Sprint +15%, Expansion +12%, Preservation +7.5%)
NOISE_ADAPTIVE_MATRIX: List[Dict[str, Any]] = [
    {"level": 1, "starting_balance": 100.0, "risk_pct": 10.0, "risk_dollars": 10.00, "profit_pct": 15.0, "profit_dollars": 15.00, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 115.0},
    {"level": 2, "starting_balance": 115.0, "risk_pct": 10.0, "risk_dollars": 11.50, "profit_pct": 15.0, "profit_dollars": 17.25, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 132.25},
    {"level": 3, "starting_balance": 132.25, "risk_pct": 10.0, "risk_dollars": 13.23, "profit_pct": 15.0, "profit_dollars": 19.84, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 152.09},
    {"level": 4, "starting_balance": 152.09, "risk_pct": 10.0, "risk_dollars": 15.21, "profit_pct": 15.0, "profit_dollars": 22.81, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 174.90},
    {"level": 5, "starting_balance": 174.90, "risk_pct": 10.0, "risk_dollars": 17.49, "profit_pct": 15.0, "profit_dollars": 26.24, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 201.14},
    {"level": 6, "starting_balance": 201.14, "risk_pct": 10.0, "risk_dollars": 20.11, "profit_pct": 15.0, "profit_dollars": 30.17, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 231.31},
    {"level": 7, "starting_balance": 231.31, "risk_pct": 10.0, "risk_dollars": 23.13, "profit_pct": 15.0, "profit_dollars": 34.70, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 266.01},
    {"level": 8, "starting_balance": 266.01, "risk_pct": 10.0, "risk_dollars": 26.60, "profit_pct": 15.0, "profit_dollars": 39.90, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 305.91},
    {"level": 9, "starting_balance": 305.91, "risk_pct": 10.0, "risk_dollars": 30.59, "profit_pct": 15.0, "profit_dollars": 45.89, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 351.80},
    {"level": 10, "starting_balance": 351.80, "risk_pct": 10.0, "risk_dollars": 35.18, "profit_pct": 15.0, "profit_dollars": 52.77, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 404.57},
    {"level": 11, "starting_balance": 404.57, "risk_pct": 10.0, "risk_dollars": 40.46, "profit_pct": 15.0, "profit_dollars": 60.69, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 465.26},
    {"level": 12, "starting_balance": 465.26, "risk_pct": 10.0, "risk_dollars": 46.53, "profit_pct": 15.0, "profit_dollars": 69.79, "phase": "Sprint", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 535.05},
    {"level": 13, "starting_balance": 535.05, "risk_pct": 8.0, "risk_dollars": 42.80, "profit_pct": 12.0, "profit_dollars": 64.21, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 599.26},
    {"level": 14, "starting_balance": 599.26, "risk_pct": 8.0, "risk_dollars": 47.94, "profit_pct": 12.0, "profit_dollars": 71.91, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 671.17},
    {"level": 15, "starting_balance": 671.17, "risk_pct": 8.0, "risk_dollars": 53.69, "profit_pct": 12.0, "profit_dollars": 80.54, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 751.71},
    {"level": 16, "starting_balance": 751.71, "risk_pct": 8.0, "risk_dollars": 60.14, "profit_pct": 12.0, "profit_dollars": 90.21, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 841.92},
    {"level": 17, "starting_balance": 841.92, "risk_pct": 8.0, "risk_dollars": 67.35, "profit_pct": 12.0, "profit_dollars": 101.03, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 942.95},
    {"level": 18, "starting_balance": 942.95, "risk_pct": 8.0, "risk_dollars": 75.44, "profit_pct": 12.0, "profit_dollars": 113.15, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 1056.10},
    {"level": 19, "starting_balance": 1056.10, "risk_pct": 8.0, "risk_dollars": 84.49, "profit_pct": 12.0, "profit_dollars": 126.73, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 1182.83},
    {"level": 20, "starting_balance": 1182.83, "risk_pct": 8.0, "risk_dollars": 94.63, "profit_pct": 12.0, "profit_dollars": 141.94, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 1324.77},
    {"level": 21, "starting_balance": 1324.77, "risk_pct": 8.0, "risk_dollars": 105.98, "profit_pct": 12.0, "profit_dollars": 158.97, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 1483.74},
    {"level": 22, "starting_balance": 1483.74, "risk_pct": 8.0, "risk_dollars": 118.70, "profit_pct": 12.0, "profit_dollars": 178.05, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 1661.79},
    {"level": 23, "starting_balance": 1661.79, "risk_pct": 8.0, "risk_dollars": 132.94, "profit_pct": 12.0, "profit_dollars": 199.41, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 1861.20},
    {"level": 24, "starting_balance": 1861.20, "risk_pct": 8.0, "risk_dollars": 148.90, "profit_pct": 12.0, "profit_dollars": 223.34, "phase": "Expansion", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 2084.54},
    {"level": 25, "starting_balance": 2084.54, "risk_pct": 5.0, "risk_dollars": 104.23, "profit_pct": 7.5, "profit_dollars": 156.34, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 2240.88},
    {"level": 26, "starting_balance": 2240.88, "risk_pct": 5.0, "risk_dollars": 112.04, "profit_pct": 7.5, "profit_dollars": 168.07, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 2408.95},
    {"level": 27, "starting_balance": 2408.95, "risk_pct": 5.0, "risk_dollars": 120.45, "profit_pct": 7.5, "profit_dollars": 180.67, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 2589.62},
    {"level": 28, "starting_balance": 2589.62, "risk_pct": 5.0, "risk_dollars": 129.48, "profit_pct": 7.5, "profit_dollars": 194.22, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 2783.84},
    {"level": 29, "starting_balance": 2783.84, "risk_pct": 5.0, "risk_dollars": 139.19, "profit_pct": 7.5, "profit_dollars": 208.79, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 2992.63},
    {"level": 30, "starting_balance": 2992.63, "risk_pct": 5.0, "risk_dollars": 149.63, "profit_pct": 7.5, "profit_dollars": 224.45, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 3217.08},
    {"level": 31, "starting_balance": 3217.08, "risk_pct": 5.0, "risk_dollars": 160.85, "profit_pct": 7.5, "profit_dollars": 241.28, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 3458.36},
    {"level": 32, "starting_balance": 3458.36, "risk_pct": 5.0, "risk_dollars": 172.92, "profit_pct": 7.5, "profit_dollars": 259.38, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 3717.74},
    {"level": 33, "starting_balance": 3717.74, "risk_pct": 5.0, "risk_dollars": 185.89, "profit_pct": 7.5, "profit_dollars": 278.83, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 3996.57},
    {"level": 34, "starting_balance": 3996.57, "risk_pct": 5.0, "risk_dollars": 199.83, "profit_pct": 7.5, "profit_dollars": 299.74, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 4296.31},
    {"level": 35, "starting_balance": 4296.31, "risk_pct": 5.0, "risk_dollars": 214.82, "profit_pct": 7.5, "profit_dollars": 322.22, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 4618.53},
    {"level": 36, "starting_balance": 4618.53, "risk_pct": 5.0, "risk_dollars": 230.93, "profit_pct": 7.5, "profit_dollars": 346.39, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 4964.92},
    {"level": 37, "starting_balance": 4964.92, "risk_pct": 5.0, "risk_dollars": 248.25, "profit_pct": 7.5, "profit_dollars": 372.37, "phase": "Preservation", "min_sl_dollars": 8.0, "rr_ratio": 1.5, "ending_balance": 5337.29},
]


def get_ladder_level_info(level: int) -> Dict[str, Any]:
    """Retrieve specifications for a given level (1 to 30) from the classic discrete matrix."""
    idx = max(1, min(30, int(level))) - 1
    return dict(LADDER_MATRIX[idx])


def get_noise_adaptive_level_info(level: int) -> Dict[str, Any]:
    """Retrieve specifications for a given level (1 to 37) from the Noise-Adaptive Matrix."""
    idx = max(1, min(len(NOISE_ADAPTIVE_MATRIX), int(level))) - 1
    return dict(NOISE_ADAPTIVE_MATRIX[idx])


def find_closest_level_for_balance(balance: float, matrix: Optional[List[Dict[str, Any]]] = None) -> int:
    """Find the highest ladder level whose starting balance <= balance."""
    mat = matrix or LADDER_MATRIX
    if balance < mat[0]["starting_balance"]:
        return 1
    for row in reversed(mat):
        if balance >= row["starting_balance"]:
            return row["level"]
    return 1


class CompoundingLadderManager:
    """
    Manages the Account Compounding Ladder state, progression,
    bracket calculations, and risk safety mechanisms with full support for:
    - NOISE_ADAPTIVE_LADDER (Volatility-calibrated ladder with $8+ stops & 1.5R target)
    - DISCRETE_30_LEVEL (Classic step ladder)
    - TIERED_PHASED_SPRINT (Phased risk tiers with ratchet floors)
    - DYNAMIC_EQUITY_PERCENT (Fixed fractional / Kelly sizing)
    - STREAK_ADAPTIVE (Anti-martingale streak expansion & de-risking)
    - HIGH_WATERMARK_RATCHET (Surplus compounding above locked principal)
    """

    def __init__(
        self,
        state_file: Optional[str] = None,
        loss_policy: Literal[
            "ONE_STEP_LADDER",
            "RESET_TO_LEVEL_1",
            "STEP_DOWN_BY_BALANCE",
            "PRESERVE_MILESTONES",
            "PROGRESSIVE_RATCHET",
            "TRAILING_DD_FLOOR",
        ] = "ONE_STEP_LADDER",
        ladder_mode: Literal[
            "NOISE_ADAPTIVE_LADDER",
            "DISCRETE_30_LEVEL",
            "TIERED_PHASED_SPRINT",
            "DYNAMIC_EQUITY_PERCENT",
            "STREAK_ADAPTIVE",
            "HIGH_WATERMARK_RATCHET",
        ] = "DISCRETE_30_LEVEL",
        starting_balance: float = 20.0,
        target_balance: float = 52384.0,
        base_risk_pct: float = 23.0,
        enable_circuit_breaker: bool = True,
        trailing_dd_floor_pct: Optional[float] = None,
    ):
        self.state_file = os.path.expanduser(
            state_file or "~/.gold_trading_data/compounding_ladder_state.json"
        )
        self.loss_policy = loss_policy
        self.ladder_mode = ladder_mode
        self.starting_balance = starting_balance
        self.target_balance = target_balance
        self.base_risk_pct = base_risk_pct
        self.enable_circuit_breaker = enable_circuit_breaker
        self.trailing_dd_floor_pct = trailing_dd_floor_pct

        self._ensure_dir()
        self.state = self._load_state()

    def _ensure_dir(self) -> None:
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file) and os.path.getsize(self.state_file) > 0:
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    state.setdefault("ladder_mode", self.ladder_mode)
                    state.setdefault("loss_policy", self.loss_policy)
                    state.setdefault("starting_balance", self.starting_balance)
                    state.setdefault("target_balance", self.target_balance)
                    state.setdefault("base_risk_pct", self.base_risk_pct)
                    state.setdefault("enable_circuit_breaker", self.enable_circuit_breaker)
                    state.setdefault("trailing_dd_floor_pct", self.trailing_dd_floor_pct)
                    state.setdefault("scratches", 0)
                    state.setdefault("reloads", 0)
                    state.setdefault("consecutive_losses", 0)
                    state.setdefault("ratchet_floor", 0.0)
                    state.setdefault("peak_balance", state.get("current_balance", self.starting_balance))
                    # Sync instance attributes with stored configuration if present
                    self.ladder_mode = state.get("ladder_mode", self.ladder_mode)
                    self.loss_policy = state.get("loss_policy", self.loss_policy)
                    self.enable_circuit_breaker = state.get("enable_circuit_breaker", self.enable_circuit_breaker)
                    return state
            except Exception as e:
                logger.error(f"Error loading ladder state, creating fresh state: {e}")

        # Default initial state
        initial_state = {
            "current_level": 1,
            "starting_balance": self.starting_balance,
            "current_balance": self.starting_balance,
            "peak_balance": self.starting_balance,
            "ratchet_floor": 0.0,
            "peak_level": 1,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "scratches": 0,
            "reloads": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "ladder_mode": self.ladder_mode,
            "loss_policy": self.loss_policy,
            "base_risk_pct": self.base_risk_pct,
            "target_balance": self.target_balance,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "trailing_dd_floor_pct": self.trailing_dd_floor_pct,
            "status": "ACTIVE",  # ACTIVE, COMPLETED, RESET
            "history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state(initial_state)
        return initial_state

    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist ladder state: {e}")

    def update_ratchet_floor(self) -> float:
        """Computes and updates active ratchet floor based on policy and peak balance."""
        bal = self.state["current_balance"]
        peak = self.state.get("peak_balance", bal)
        floor = self.state.get("ratchet_floor", 0.0)

        if self.loss_policy == "PROGRESSIVE_RATCHET":
            if bal >= 4000.0:
                floor = max(floor, 3000.0)
            elif bal >= 2500.0:
                floor = max(floor, 1800.0)
            elif bal >= 1000.0:
                floor = max(floor, 700.0)
            elif bal >= 500.0:
                floor = max(floor, 350.0)
            elif bal >= 250.0:
                floor = max(floor, 150.0)
        elif self.loss_policy == "PRESERVE_MILESTONES":
            if bal >= 2900.0:
                floor = max(floor, 2900.0)
            elif bal >= 200.0:
                floor = max(floor, 200.0)
        elif self.loss_policy == "TRAILING_DD_FLOOR" and self.trailing_dd_floor_pct:
            floor = max(floor, peak * (1.0 - (self.trailing_dd_floor_pct / 100.0)))

        self.state["ratchet_floor"] = floor
        return floor

    def get_ratchet_floor(self) -> float:
        """Returns the currently locked capital floor."""
        return float(self.state.get("ratchet_floor", 0.0))

    def is_circuit_breaker_active(self) -> bool:
        """Returns True if the circuit breaker has engaged after 2+ consecutive losses."""
        return bool(self.enable_circuit_breaker and (self.state.get("consecutive_losses", 0) >= 2))

    def get_active_risk_pct(self) -> float:
        """Calculates effective risk percentage for the upcoming trade."""
        bal = self.state["current_balance"]

        if self.ladder_mode == "NOISE_ADAPTIVE_LADDER":
            lvl_info = get_noise_adaptive_level_info(self.state["current_level"])
            pct = lvl_info["risk_pct"]
        elif self.ladder_mode == "DISCRETE_30_LEVEL":
            lvl_info = get_ladder_level_info(self.state["current_level"])
            pct = lvl_info["risk_pct"]
        elif self.ladder_mode == "TIERED_PHASED_SPRINT":
            if bal < 500.0:
                pct = 10.0
            elif bal < 2000.0:
                pct = 8.0
            else:
                pct = 5.0
        elif self.ladder_mode == "STREAK_ADAPTIVE":
            if self.state.get("consecutive_losses", 0) > 0:
                pct = 3.5  # De-risk after loss
            else:
                consec_w = self.state.get("consecutive_wins", 0)
                pct = min(12.0, 6.0 + (consec_w * 1.5))
        elif self.ladder_mode == "HIGH_WATERMARK_RATCHET":
            pct = 15.0
        else:  # DYNAMIC_EQUITY_PERCENT
            pct = float(self.state.get("base_risk_pct", self.base_risk_pct))

        if self.is_circuit_breaker_active():
            pct *= 0.50

        return round(pct, 2)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state and parameters for the upcoming trade."""
        self.update_ratchet_floor()
        if self.ladder_mode == "NOISE_ADAPTIVE_LADDER":
            lvl_info = get_noise_adaptive_level_info(self.state["current_level"])
            next_specs = {
                "level": lvl_info["level"],
                "phase": lvl_info.get("phase", "Sprint"),
                "starting_balance": lvl_info["starting_balance"],
                "risk_pct": lvl_info["risk_pct"],
                "expected_risk_usd": lvl_info["risk_dollars"],
                "expected_gain_usd": lvl_info["profit_dollars"],
                "min_sl_dollars": lvl_info.get("min_sl_dollars", 8.0),
                "reward_risk_ratio": lvl_info.get("rr_ratio", 1.5),
                "projected_balance_on_win": lvl_info["ending_balance"],
            }
        else:
            lvl_info = get_ladder_level_info(self.state["current_level"])
            next_specs = {
                "level": lvl_info["level"],
                "target_lot_size": lvl_info["lot_size"],
                "tp_pips": lvl_info["tp_pips"],
                "sl_pips": lvl_info["sl_pips"],
                "tp_points_usd": round(lvl_info["tp_pips"] * 0.10, 2),
                "sl_points_usd": round(lvl_info["sl_pips"] * 0.10, 2),
                "breakeven_lock_pips": 10.0,
                "breakeven_lock_usd": 1.00,
                "pullback_limit_discount_pips": 8.0,
                "pullback_limit_discount_usd": 0.80,
                "expected_gain_usd": lvl_info["profit_dollars"],
                "expected_risk_usd": lvl_info["risk_dollars"],
                "projected_balance_on_win": lvl_info["ending_balance"],
            }

        decided_trades = self.state["wins"] + self.state["losses"]
        win_rate = (
            round((self.state["wins"] / decided_trades) * 100, 1)
            if decided_trades > 0
            else 0.0
        )
        return {
            "status": self.state["status"],
            "ladder_mode": self.ladder_mode,
            "loss_policy": self.loss_policy,
            "current_level": self.state["current_level"],
            "starting_balance": round(self.state.get("starting_balance", self.starting_balance), 2),
            "current_balance": round(self.state["current_balance"], 2),
            "peak_balance": round(self.state.get("peak_balance", self.state["current_balance"]), 2),
            "ratchet_floor": self.get_ratchet_floor(),
            "peak_level": self.state["peak_level"],
            "consecutive_wins": self.state["consecutive_wins"],
            "consecutive_losses": self.state.get("consecutive_losses", 0),
            "circuit_breaker_active": self.is_circuit_breaker_active(),
            "active_risk_pct": self.get_active_risk_pct(),
            "total_trades": self.state["total_trades"],
            "wins": self.state["wins"],
            "losses": self.state["losses"],
            "scratches": self.state.get("scratches", 0),
            "reloads": self.state.get("reloads", 0),
            "win_rate_pct": win_rate,
            "next_trade_specs": next_specs,
        }

    def calculate_bracket_order(
        self,
        current_price: float,
        direction: Literal["BUY", "SELL"] = "BUY",
        symbol: str = "XAUUSD",
        use_pullback_limit: bool = False,
        pullback_discount_pips: float = 8.0,
        atr: Optional[float] = None,
        use_atr_stops: bool = True,
        min_sl_pts: float = 8.0,
        use_ladder_lots: bool = True,
        rr_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate exact execution parameters for the current ladder mode and level.
        In Gold (XAUUSD): 1 pip = $0.10 (10 points).
        
        When use_atr_stops=True:
          - Dynamic SL: max(1.5 * ATR, min_sl_pts) (default min $8.00 / 80 pips).
          - Dynamic TP: rr_ratio * SL (default 1.8R for Strategy 2 Hybrid).
          - Breakeven trigger: 1.0R in profit.
          - Lot size: dynamically scaled to planned dollar risk.
        """
        self.update_ratchet_floor()
        if self.ladder_mode == "NOISE_ADAPTIVE_LADDER":
            lvl_info = get_noise_adaptive_level_info(self.state["current_level"])
        else:
            lvl_info = get_ladder_level_info(self.state["current_level"])
        bal = self.state["current_balance"]
        floor = self.get_ratchet_floor()

        # Available capital for risk calculation
        risk_capital = max(bal - floor, bal * 0.35) if floor > 0 else bal
        active_pct = self.get_active_risk_pct() / 100.0

        is_btc = "BTC" in symbol.upper()
        contract_size = 1.0 if is_btc else CONTRACT_SIZE
        default_min_sl = 500.0 if is_btc else 8.0

        if use_atr_stops and (atr is not None or min_sl_pts > 0):
            current_atr = atr if (atr and atr > 0) else (450.0 if is_btc else 15.0)
            min_pts = lvl_info.get("min_sl_dollars", default_min_sl) if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else default_min_sl
            if is_btc:
                min_pts = max(min_pts, 500.0)
            sl_pts = round(max(1.5 * current_atr, min_pts), 2)
            eff_rr = rr_ratio if rr_ratio is not None else lvl_info.get("rr_ratio", 1.25)
            tp_pts = round(sl_pts * eff_rr, 2)
            be_pts = round(sl_pts * 1.0, 2)

            if self.ladder_mode == "DISCRETE_30_LEVEL" and use_ladder_lots:
                lots = lvl_info["lot_size"]
            elif self.ladder_mode == "NOISE_ADAPTIVE_LADDER":
                planned_risk_usd = lvl_info["risk_dollars"]
                if self.is_circuit_breaker_active():
                    planned_risk_usd *= 0.50
                calculated_lots = math.floor((planned_risk_usd / (sl_pts * contract_size)) * 100.0 + 0.0001) / 100.0
                lots = max(MIN_LOT, min(MAX_LOT, calculated_lots))
            else:
                # Dynamic lot sizing based on active archetype & risk capital
                planned_risk_usd = risk_capital * active_pct
                calculated_lots = math.floor((planned_risk_usd / (sl_pts * contract_size)) * 100.0 + 0.0001) / 100.0
                lots = max(MIN_LOT, min(MAX_LOT, calculated_lots))
        else:
            lots = lvl_info.get("lot_size", 0.01)
            tp_pts = round(lvl_info.get("tp_pips", 20.0) * 0.10, 2)  # Typically $2.00
            sl_pts = round(lvl_info.get("sl_pips", 15.0) * 0.10, 2)  # Typically $1.50 - $1.64
            be_pts = round(sl_pts * 1.0, 2)

        discount_pts = round(pullback_discount_pips * 0.10, 2) if use_pullback_limit else 0.0

        if direction.upper() == "BUY":
            entry_price = round(current_price - discount_pts, 2)
            tp_price = round(entry_price + tp_pts, 2)
            sl_price = round(entry_price - sl_pts, 2)
            be_trigger_price = round(entry_price + be_pts, 2)
        else:
            entry_price = round(current_price + discount_pts, 2)
            tp_price = round(entry_price - tp_pts, 2)
            sl_price = round(entry_price + sl_pts, 2)
            be_trigger_price = round(entry_price - be_pts, 2)

        expected_profit = round(lots * contract_size * tp_pts, 2)
        expected_loss = round(lots * contract_size * sl_pts, 2)

        return {
            "symbol": symbol.upper(),
            "level": self.state["current_level"],
            "ladder_mode": self.ladder_mode,
            "direction": direction.upper(),
            "order_type": "LIMIT" if use_pullback_limit else "MARKET",
            "market_price": round(current_price, 2),
            "entry_price": entry_price,
            "pullback_discount_pips": pullback_discount_pips if use_pullback_limit else 0.0,
            "lots": lots,
            "take_profit": tp_price,
            "stop_loss": sl_price,
            "breakeven_trigger_price": be_trigger_price,
            "tp_pips": round(tp_pts * 10, 1),
            "sl_pips": round(sl_pts * 10, 1),
            "tp_points": tp_pts,
            "sl_points": sl_pts,
            "tp_pts": tp_pts,
            "sl_pts": sl_pts,
            "expected_profit_usd": expected_profit,
            "expected_loss_usd": expected_loss,
            "active_risk_pct": round(active_pct * 100.0, 2),
            "ratchet_floor": floor,
            "circuit_breaker_active": self.is_circuit_breaker_active(),
            "reward_risk_ratio": round(tp_pts / sl_pts, 2) if sl_pts > 0 else rr_ratio,
        }

    def record_trade_result(
        self,
        win: bool,
        realized_pnl: float,
        exit_price: float,
        notes: str = "",
        is_breakeven: bool = False,
    ) -> Dict[str, Any]:
        """
        Records trade outcome, updates balance, and executes level progression or step-down.
        - Win: Advances progression / increments win streak
        - Breakeven (Scratch): Preserves state and retries
        - Loss: Updates loss streak, triggers circuit breaker if applicable, applies floors
        """
        prev_level = self.state["current_level"]
        self.state["total_trades"] += 1

        # Check if scratch / breakeven exit
        if is_breakeven or (abs(realized_pnl) < 0.25 and not win):
            outcome = "BREAKEVEN"
            self.state["scratches"] = self.state.get("scratches", 0) + 1
            new_level = prev_level
            self.state["current_balance"] = max(0.0, round(self.state["current_balance"] + realized_pnl, 2))
        elif win:
            outcome = "WIN"
            self.state["wins"] += 1
            self.state["consecutive_wins"] += 1
            self.state["consecutive_losses"] = 0
            self.state["current_balance"] = round(self.state["current_balance"] + realized_pnl, 2)
            self.state["peak_balance"] = max(self.state.get("peak_balance", 0.0), self.state["current_balance"])
            self.update_ratchet_floor()

            max_lvl = len(NOISE_ADAPTIVE_MATRIX) if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else 30
            if prev_level >= max_lvl or self.state["current_balance"] >= self.target_balance:
                self.state["status"] = "COMPLETED"
                logger.info("🏆 COMPOUNDING CHALLENGE COMPLETED SUCCESSFULLY!")
                new_level = max_lvl
            else:
                new_level = prev_level + 1
                self.state["peak_level"] = max(self.state["peak_level"], new_level)
            self.state["current_level"] = new_level
        else:
            outcome = "LOSS"
            self.state["losses"] += 1
            self.state["consecutive_wins"] = 0
            self.state["consecutive_losses"] = self.state.get("consecutive_losses", 0) + 1

            # Apply ratchet floor constraint
            floor = self.get_ratchet_floor()
            raw_balance = round(self.state["current_balance"] + realized_pnl, 2)
            self.state["current_balance"] = max(floor, max(0.0, raw_balance))

            # Handle loss policy
            if self.loss_policy == "ONE_STEP_LADDER":
                new_level = max(1, prev_level - 1)
                if self.state["current_balance"] < 5.0:
                    self.state["reloads"] = self.state.get("reloads", 0) + 1
                    logger.warning("Balance fell below margin threshold ($5.00). Reloading seed capital.")
                    self.state["current_balance"] = self.starting_balance
                    new_level = 1
                self.state["current_level"] = new_level
            elif self.loss_policy == "RESET_TO_LEVEL_1":
                self.state["current_level"] = 1
                new_level = 1
            elif self.loss_policy == "STEP_DOWN_BY_BALANCE":
                mat = NOISE_ADAPTIVE_MATRIX if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else LADDER_MATRIX
                new_level = find_closest_level_for_balance(self.state["current_balance"], matrix=mat)
                self.state["current_level"] = new_level
            elif self.loss_policy == "PRESERVE_MILESTONES":
                if self.ladder_mode == "NOISE_ADAPTIVE_LADDER":
                    if prev_level >= 25:
                        new_level = 25
                    elif prev_level >= 13:
                        new_level = 13
                    else:
                        new_level = 1
                else:
                    if prev_level >= 20:
                        new_level = 20
                    elif prev_level >= 10:
                        new_level = 10
                    else:
                        new_level = 1
                self.state["current_level"] = new_level
            else:
                mat = NOISE_ADAPTIVE_MATRIX if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else LADDER_MATRIX
                new_level = find_closest_level_for_balance(self.state["current_balance"], matrix=mat)
                self.state["current_level"] = new_level

        trade_record = {
            "trade_no": self.state["total_trades"],
            "level": prev_level,
            "outcome": outcome,
            "win": win,
            "is_breakeven": (outcome == "BREAKEVEN"),
            "pnl": round(realized_pnl, 2),
            "exit_price": exit_price,
            "balance_after": self.state["current_balance"],
            "ratchet_floor": self.get_ratchet_floor(),
            "new_level": self.state["current_level"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        self.state["history"].append(trade_record)
        self._save_state()

        return {
            "trade_result": outcome,
            "pnl": realized_pnl,
            "previous_level": prev_level,
            "new_level": self.state["current_level"],
            "new_balance": self.state["current_balance"],
            "ratchet_floor": self.get_ratchet_floor(),
            "status": self.state["status"],
            "loss_policy": self.loss_policy,
            "ladder_mode": self.ladder_mode,
        }

    def reset_ladder(self, starting_balance: float = 1000.0, reset_stats: bool = False) -> Dict[str, Any]:
        """Manually reset the ladder with specified capital, automatically calibrating initial level."""
        mat = NOISE_ADAPTIVE_MATRIX if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else LADDER_MATRIX
        start_lvl = find_closest_level_for_balance(starting_balance, matrix=mat)
        self.state["current_level"] = start_lvl
        self.state["peak_level"] = max(self.state.get("peak_level", 1), start_lvl)
        self.state["starting_balance"] = starting_balance
        self.state["current_balance"] = starting_balance
        self.state["peak_balance"] = starting_balance
        self.state["ratchet_floor"] = 0.0
        self.state["consecutive_wins"] = 0
        self.state["consecutive_losses"] = 0
        if reset_stats:
            self.state["total_trades"] = 0
            self.state["wins"] = 0
            self.state["losses"] = 0
            self.state["scratches"] = 0
            self.state["reloads"] = 0
            self.state["history"] = []
        self.state["status"] = "ACTIVE"
        self.update_ratchet_floor()
        self._save_state()
        return self.get_status()

    def set_level(self, level: int) -> Dict[str, Any]:
        """Manually override current level (e.g. to match existing balance)."""
        max_lvl = len(NOISE_ADAPTIVE_MATRIX) if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else 30
        valid_level = max(1, min(max_lvl, int(level)))
        self.state["current_level"] = valid_level
        self.state["peak_level"] = max(self.state["peak_level"], valid_level)
        self.state["status"] = "ACTIVE"
        self._save_state()
        return self.get_status()

    def set_archetype(
        self,
        ladder_mode: Literal[
            "NOISE_ADAPTIVE_LADDER",
            "DISCRETE_30_LEVEL",
            "TIERED_PHASED_SPRINT",
            "DYNAMIC_EQUITY_PERCENT",
            "STREAK_ADAPTIVE",
            "HIGH_WATERMARK_RATCHET",
        ],
        base_risk_pct: Optional[float] = None,
        loss_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dynamically update the active compounding progression archetype."""
        self.ladder_mode = ladder_mode
        self.state["ladder_mode"] = ladder_mode
        if base_risk_pct is not None:
            self.base_risk_pct = base_risk_pct
            self.state["base_risk_pct"] = base_risk_pct
        if loss_policy is not None:
            self.loss_policy = loss_policy  # type: ignore
            self.state["loss_policy"] = loss_policy
        self._save_state()
        logger.info(f"Updated Compounding Ladder archetype to {ladder_mode}")
        return self.get_status()

    def sync_level_to_balance(self, balance: float) -> Dict[str, Any]:
        """Automatically synchronizes current ladder level to account balance."""
        mat = NOISE_ADAPTIVE_MATRIX if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else LADDER_MATRIX
        matched_level = find_closest_level_for_balance(balance, matrix=mat)
        self.state["current_balance"] = round(balance, 2)
        self.state["peak_balance"] = max(self.state.get("peak_balance", 0.0), balance)
        self.update_ratchet_floor()
        return self.set_level(matched_level)

    def get_all_levels(self) -> List[Dict[str, Any]]:
        """Returns the complete active compounding matrix."""
        return list(NOISE_ADAPTIVE_MATRIX if self.ladder_mode == "NOISE_ADAPTIVE_LADDER" else LADDER_MATRIX)
