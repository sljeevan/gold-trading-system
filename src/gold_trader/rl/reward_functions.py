"""
Reward functions for Gold Trading RL Environment.
Provides pluggable reward formulations tailored to leveraged CFD trading mechanics
(XAU/USD), avoiding reward hacking, excessive churn, and deep drawdowns.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRewardScheme(ABC):
    """Abstract base class for trading reward schemes."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal reward state at the start of an episode."""
        pass

    @abstractmethod
    def calculate_reward(
        self,
        prev_equity: float,
        current_equity: float,
        peak_equity: float,
        trade_event: Optional[Dict[str, Any]],
        position: int,
        bars_held: int,
    ) -> float:
        """
        Calculate scalar step reward.

        Args:
            prev_equity: Portfolio equity at t-1.
            current_equity: Portfolio equity at t.
            peak_equity: Highest historical portfolio equity in the episode.
            trade_event: Dict of closed trade details if a trade closed this step, else None.
            position: Current position (-1: SHORT, 0: FLAT, 1: LONG).
            bars_held: Number of bars the current position has been open.

        Returns:
            Scalar reward for the step.
        """
        pass


class LogReturnReward(BaseRewardScheme):
    """
    Log return reward based on net portfolio equity change.
    Includes turnover penalty to prevent over-trading and spread bleeding.
    """

    def __init__(self, turnover_penalty: float = 0.0002, holding_decay: float = 0.00002):
        self.turnover_penalty = turnover_penalty
        self.holding_decay = holding_decay

    def reset(self) -> None:
        pass

    def calculate_reward(
        self,
        prev_equity: float,
        current_equity: float,
        peak_equity: float,
        trade_event: Optional[Dict[str, Any]],
        position: int,
        bars_held: int,
    ) -> float:
        if prev_equity <= 0 or current_equity <= 0:
            return -10.0  # Ruin / margin call penalty

        # Log return of equity
        ret = math.log(current_equity / prev_equity)

        # Penalize new position entries (discourages churn)
        penalty = 0.0
        if trade_event and trade_event.get("type") == "OPEN":
            penalty += self.turnover_penalty

        # Minor holding decay to avoid passive stagnation
        if position != 0:
            penalty += self.holding_decay * min(bars_held, 50) / 50.0

        return float(ret - penalty)


class SortinoSharpeReward(BaseRewardScheme):
    """
    Differential Sortino/Sharpe reward.
    Penalizes downside variance (adverse excursion) significantly more than upside gains.
    """

    def __init__(self, downside_scale: float = 2.5, turnover_penalty: float = 0.0002):
        self.downside_scale = downside_scale
        self.turnover_penalty = turnover_penalty

    def reset(self) -> None:
        pass

    def calculate_reward(
        self,
        prev_equity: float,
        current_equity: float,
        peak_equity: float,
        trade_event: Optional[Dict[str, Any]],
        position: int,
        bars_held: int,
    ) -> float:
        if prev_equity <= 0 or current_equity <= 0:
            return -10.0

        pct_ret = (current_equity - prev_equity) / prev_equity

        if pct_ret < 0:
            # Asymmetric downside penalty
            reward = pct_ret * self.downside_scale
        else:
            reward = pct_ret

        if trade_event and trade_event.get("type") == "OPEN":
            reward -= self.turnover_penalty

        return float(reward)


class DrawdownPenalizedReward(BaseRewardScheme):
    """
    Reward function that adds a quadratic penalty proportional to drawdown from peak equity.
    Teaches the RL agent to protect profits and enforce tight capital preservation.
    """

    def __init__(
        self,
        drawdown_penalty_coef: float = 2.0,
        turnover_penalty: float = 0.0002,
    ):
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.turnover_penalty = turnover_penalty

    def reset(self) -> None:
        pass

    def calculate_reward(
        self,
        prev_equity: float,
        current_equity: float,
        peak_equity: float,
        trade_event: Optional[Dict[str, Any]],
        position: int,
        bars_held: int,
    ) -> float:
        if prev_equity <= 0 or current_equity <= 0:
            return -10.0

        pct_ret = (current_equity - prev_equity) / prev_equity
        drawdown = max(0.0, (peak_equity - current_equity) / peak_equity) if peak_equity > 0 else 0.0

        dd_penalty = self.drawdown_penalty_coef * (drawdown ** 2)
        turnover = self.turnover_penalty if (trade_event and trade_event.get("type") == "OPEN") else 0.0

        return float(pct_ret - dd_penalty - turnover)


class TripleBarrierReward(BaseRewardScheme):
    """
    Reward aligned with Marcos Lopez de Prado's Triple-Barrier Method.
    Provides discrete feedback when Profit-Taking or Stop-Loss barriers are triggered.
    """

    def __init__(
        self,
        tp_reward: float = 2.0,
        sl_penalty: float = -1.5,
        time_decay_per_bar: float = -0.0001,
        unrealized_scale: float = 0.1,
    ):
        self.tp_reward = tp_reward
        self.sl_penalty = sl_penalty
        self.time_decay_per_bar = time_decay_per_bar
        self.unrealized_scale = unrealized_scale

    def reset(self) -> None:
        pass

    def calculate_reward(
        self,
        prev_equity: float,
        current_equity: float,
        peak_equity: float,
        trade_event: Optional[Dict[str, Any]],
        position: int,
        bars_held: int,
    ) -> float:
        if prev_equity <= 0 or current_equity <= 0:
            return -10.0

        step_return = (current_equity - prev_equity) / prev_equity
        reward = step_return * self.unrealized_scale

        if trade_event and trade_event.get("type") == "CLOSE":
            outcome = trade_event.get("exit_reason", "")
            if outcome == "TP":
                reward += self.tp_reward
            elif outcome == "SL":
                reward += self.sl_penalty
            else:
                pnl = trade_event.get("pnl", 0.0)
                reward += (1.0 if pnl > 0 else -1.0) * min(abs(pnl) / 100.0, 1.5)

        if position != 0:
            reward += self.time_decay_per_bar

        return float(reward)
