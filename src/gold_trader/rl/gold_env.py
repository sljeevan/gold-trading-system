"""
Custom Gymnasium Environment for Gold (XAUUSD / COMEX Futures) Trading.
Models realistic leveraged CFD mechanics:
- Bidirectional execution (LONG, SHORT, FLAT/HOLD, CLOSE)
- Dynamic ATR-based Stop Loss & Take Profit bracket checks
- Dynamic lot sizing anchored to exact % account risk (1% default)
- Institutional spread & slippage deduction (2.5 pips default)
- Stationary observation space + portfolio state tracking
- Risk-adjusted Sharpe & Drawdown-penalized reward engineering
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("GoldTradingEnv")


class GoldTradingEnv(gym.Env):
    """
    Institutional Gold Trading Environment for Reinforcement Learning.
    """

    metadata = {"render_modes": ["human"]}

    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_CLOSE = 3

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
        contract_size: float = 100.0,  # 100 oz per standard lot
        spread_pips: float = 2.5,  # 2.5 pips = $0.25 price impact on entry/exit
        atr_sl_mult: float = 1.5,  # Stop Loss = 1.5 * ATR
        rr_ratio: float = 2.0,  # Take Profit = 2.0 * SL distance (3.0 * ATR)
        max_holding_bars: int = 48,
        drawdown_penalty_coef: float = 0.25,
        min_lot_size: float = 0.01,
        max_lot_size: float = 5.0,
    ):
        super().__init__()

        self.features = np.asarray(features, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float32)  # [Open, High, Low, Close, ATR]
        self.num_bars = len(self.features)

        if len(self.prices) != self.num_bars:
            raise ValueError(
                f"Features length ({len(self.features)}) and Prices length ({len(self.prices)}) must match."
            )

        self.initial_balance = float(initial_balance)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.contract_size = float(contract_size)
        self.spread_points = float(spread_pips * 0.10)  # 1 pip on Gold = $0.10
        self.atr_sl_mult = float(atr_sl_mult)
        self.rr_ratio = float(rr_ratio)
        self.max_holding_bars = int(max_holding_bars)
        self.drawdown_penalty_coef = float(drawdown_penalty_coef)
        self.min_lot_size = float(min_lot_size)
        self.max_lot_size = float(max_lot_size)

        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL, 3 = CLOSE
        self.action_space = spaces.Discrete(4)

        # Observation space: market features + 6 portfolio state variables
        num_market_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_market_features + 6,),
            dtype=np.float32,
        )

        # State tracking
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance

        # Active trade state
        self.position: int = 0  # +1 = LONG, -1 = SHORT, 0 = FLAT
        self.entry_price: float = 0.0
        self.stop_loss: float = 0.0
        self.take_profit: float = 0.0
        self.lot_size: float = 0.0
        self.holding_bars: int = 0

        # Performance recording
        self.trades_history: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance

        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.lot_size = 0.0
        self.holding_bars = 0

        self.trades_history.clear()
        self.equity_curve = [self.initial_balance]

        obs = self._get_observation()
        info = {"balance": self.balance, "equity": self.equity}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= self.num_bars:
            m_feat = self.features[-1]
            close_price = self.prices[-1, 3]
            atr_val = max(0.10, self.prices[-1, 4])
        else:
            m_feat = self.features[self.current_step]
            close_price = self.prices[self.current_step, 3]
            atr_val = max(0.10, self.prices[self.current_step, 4])

        # Portfolio state
        pos_dir = float(self.position)
        unrealized_pnl = 0.0
        sl_dist_atr = 0.0

        if self.position == 1:
            unrealized_pnl = (close_price - self.entry_price) * self.lot_size * self.contract_size
            sl_dist_atr = (close_price - self.stop_loss) / atr_val
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - close_price) * self.lot_size * self.contract_size
            sl_dist_atr = (self.stop_loss - close_price) / atr_val

        curr_equity = self.balance + unrealized_pnl
        norm_pnl = unrealized_pnl / max(100.0, self.balance)
        holding_norm = self.holding_bars / max(1.0, float(self.max_holding_bars))

        drawdown = max(0.0, (self.peak_equity - curr_equity) / max(100.0, self.peak_equity))
        balance_ratio = curr_equity / self.initial_balance

        portfolio_vec = np.array(
            [
                pos_dir,
                np.clip(norm_pnl, -2.0, 2.0),
                np.clip(holding_norm, 0.0, 1.0),
                np.clip(drawdown, 0.0, 1.0),
                np.clip(balance_ratio, 0.0, 5.0),
                np.clip(sl_dist_atr, -3.0, 5.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate([m_feat, portfolio_vec]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.num_bars - 1:
            return self._get_observation(), 0.0, True, False, self._get_episode_metrics()

        prev_equity = self.equity
        open_price = float(self.prices[self.current_step, 0])
        high_price = float(self.prices[self.current_step, 1])
        low_price = float(self.prices[self.current_step, 2])
        close_price = float(self.prices[self.current_step, 3])
        atr_val = max(0.10, float(self.prices[self.current_step, 4]))

        realized_pnl_this_step = 0.0

        # 1. Manage existing open position first (check Stop-Loss and Take-Profit intra-bar)
        if self.position != 0:
            self.holding_bars += 1
            exit_triggered = False
            exit_price = close_price
            exit_reason = "HOLD"

            if self.position == 1:  # LONG
                if low_price <= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = "SL"
                    exit_triggered = True
                elif high_price >= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = "TP"
                    exit_triggered = True
                elif self.holding_bars >= self.max_holding_bars or action == self.ACTION_CLOSE:
                    exit_price = close_price - self.spread_points
                    exit_reason = "TIME" if self.holding_bars >= self.max_holding_bars else "MANUAL"
                    exit_triggered = True

            elif self.position == -1:  # SHORT
                if high_price >= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = "SL"
                    exit_triggered = True
                elif low_price <= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = "TP"
                    exit_triggered = True
                elif self.holding_bars >= self.max_holding_bars or action == self.ACTION_CLOSE:
                    exit_price = close_price + self.spread_points
                    exit_reason = "TIME" if self.holding_bars >= self.max_holding_bars else "MANUAL"
                    exit_triggered = True

            if exit_triggered:
                if self.position == 1:
                    raw_pnl = (exit_price - self.entry_price) * self.lot_size * self.contract_size
                else:
                    raw_pnl = (self.entry_price - exit_price) * self.lot_size * self.contract_size

                self.balance += raw_pnl
                realized_pnl_this_step = raw_pnl

                self.trades_history.append(
                    {
                        "step": self.current_step,
                        "type": "LONG" if self.position == 1 else "SHORT",
                        "entry": self.entry_price,
                        "exit": exit_price,
                        "lot_size": self.lot_size,
                        "pnl": round(raw_pnl, 2),
                        "reason": exit_reason,
                        "holding_bars": self.holding_bars,
                    }
                )

                # Reset trade variables
                self.position = 0
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.take_profit = 0.0
                self.lot_size = 0.0
                self.holding_bars = 0

        # 2. Open new position if flat and action is BUY or SELL
        if self.position == 0:
            if action in (self.ACTION_BUY, self.ACTION_SELL):
                risk_cash = self.balance * (self.risk_per_trade_pct / 100.0)
                sl_distance = self.atr_sl_mult * atr_val
                # Lot sizing: Cash risk = lot_size * contract_size * sl_distance
                calculated_lots = risk_cash / max(0.01, sl_distance * self.contract_size)
                lot_size = float(np.clip(calculated_lots, self.min_lot_size, self.max_lot_size))
                lot_size = round(lot_size, 2)

                if action == self.ACTION_BUY:
                    self.position = 1
                    self.entry_price = close_price + self.spread_points  # Enter at Ask
                    self.stop_loss = round(self.entry_price - sl_distance, 2)
                    self.take_profit = round(self.entry_price + (sl_distance * self.rr_ratio), 2)
                    self.lot_size = lot_size
                    self.holding_bars = 0

                elif action == self.ACTION_SELL:
                    self.position = -1
                    self.entry_price = close_price - self.spread_points  # Enter at Bid
                    self.stop_loss = round(self.entry_price + sl_distance, 2)
                    self.take_profit = round(self.entry_price - (sl_distance * self.rr_ratio), 2)
                    self.lot_size = lot_size
                    self.holding_bars = 0

        # 3. Compute Current Equity
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (close_price - self.entry_price) * self.lot_size * self.contract_size
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - close_price) * self.lot_size * self.contract_size

        self.equity = max(1.0, self.balance + unrealized_pnl)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.equity_curve.append(self.equity)

        # 4. Reward Calculation: Risk-Adjusted Returns with Drawdown & Churn Penalty
        delta_equity = self.equity - prev_equity
        norm_return = delta_equity / max(100.0, prev_equity)

        current_dd = (self.peak_equity - self.equity) / max(100.0, self.peak_equity)
        dd_penalty = self.drawdown_penalty_coef * (current_dd**2)

        # Small penalty for switching/executing to discourage pointless high-frequency churn
        churn_penalty = 0.0005 if action in (self.ACTION_BUY, self.ACTION_SELL, self.ACTION_CLOSE) else 0.0

        # Bonus on successful realized win
        realized_bonus = 0.002 if realized_pnl_this_step > 0 else 0.0

        reward = float(norm_return - dd_penalty - churn_penalty + realized_bonus)

        # 5. Move to next bar
        self.current_step += 1
        done = self.current_step >= (self.num_bars - 1) or self.equity <= (self.initial_balance * 0.30)
        truncated = False

        info = self._get_episode_metrics() if done else {}

        return self._get_observation(), reward, done, truncated, info

    def _get_episode_metrics(self) -> Dict[str, Any]:
        total_trades = len(self.trades_history)
        if total_trades == 0:
            win_rate = 0.0
            profit_factor = 0.0
        else:
            wins = [t for t in self.trades_history if t["pnl"] > 0]
            losses = [t for t in self.trades_history if t["pnl"] < 0]
            win_rate = len(wins) / total_trades
            gross_profit = sum(t["pnl"] for t in wins)
            gross_loss = abs(sum(t["pnl"] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        total_return_pct = ((self.equity - self.initial_balance) / self.initial_balance) * 100.0

        # Calculate max drawdown
        eq_arr = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dds = (peak - eq_arr) / np.maximum(1.0, peak)
        max_dd_pct = float(np.max(dds)) * 100.0

        # Sharpe ratio
        returns = np.diff(eq_arr) / eq_arr[:-1]
        std_ret = np.std(returns)
        sharpe = (np.mean(returns) / std_ret * np.sqrt(252 * 24)) if (std_ret > 1e-6 and len(returns) > 5) else 0.0

        return {
            "initial_balance": self.initial_balance,
            "final_equity": round(self.equity, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "total_trades": total_trades,
            "profit_factor": round(float(profit_factor), 2),
        }
