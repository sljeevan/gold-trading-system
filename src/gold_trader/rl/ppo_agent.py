"""
PPO Agent for Gold Trading System.
Provides a unified interface supporting both Stable-Baselines3 PPO and a built-in
native PyTorch PPO implementation (TorchPPO) tailored for financial time series.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("gold_trader.rl.ppo_agent")


# Check for optional Stable-Baselines3
try:
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.callbacks import BaseCallback
    _HAS_SB3 = True
except ImportError:
    SB3_PPO = None
    BaseCallback = object
    _HAS_SB3 = False

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    optim = None
    Categorical = None
    _HAS_TORCH = False


# ==============================================================================
# Native PyTorch PPO Implementation (Zero external framework dependencies)
# ==============================================================================

if _HAS_TORCH:
    class ActorCritic(nn.Module):
        """
        Actor-Critic Multi-Layer Perceptron architecture for trading MDPs.
        Uses orthogonal weight initialization and LayerNorm for stable financial RL training.
        """

        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()

            # Feature extractor trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )

            # Policy / Actor Head (Action Logits)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

            # Value / Critic Head (Expected Return)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Orthogonal initialization
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            # Smaller gain for actor head to ensure high initial entropy
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.constant_(self.actor[-1].bias, 0.0)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.trunk(x)
            logits = self.actor(features)
            value = self.critic(features)
            return logits, value.squeeze(-1)

        def get_action_and_value(
            self, x: torch.Tensor, action: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits, value = self.forward(x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value


    class TorchPPO:
        """
        Native, lightweight Proximal Policy Optimization (PPO-Clip) engine.
        Implements GAE (Generalized Advantage Estimation), clipped surrogate objective,
        and value function loss clipping.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int = 4,
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            hidden_dim: int = 128,
            device: str = "cpu",
        ):
            self.device = torch.device(device)
            self.network = ActorCritic(obs_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_eps = clip_eps
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef
            self.max_grad_norm = max_grad_norm

        def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
            self.network.eval()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, _ = self.network(obs_t)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
            return int(action.item())

        def get_action_distribution(self, obs: np.ndarray) -> np.ndarray:
            self.network.eval()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, _ = self.network(obs_t)
                probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()[0]

        def train_step(
            self,
            obs_b: torch.Tensor,
            actions_b: torch.Tensor,
            logprobs_b: torch.Tensor,
            returns_b: torch.Tensor,
            advantages_b: torch.Tensor,
            epochs: int = 4,
            batch_size: int = 64,
        ) -> Dict[str, float]:
            self.network.train()
            total_samples = obs_b.size(0)

            epoch_losses = []
            epoch_pg_losses = []
            epoch_vf_losses = []
            epoch_entropies = []

            for _ in range(epochs):
                indices = torch.randperm(total_samples)
                for start in range(0, total_samples, batch_size):
                    end = start + batch_size
                    b_idx = indices[start:end]

                    b_obs = obs_b[b_idx]
                    b_act = actions_b[b_idx]
                    b_logprob_old = logprobs_b[b_idx]
                    b_ret = returns_b[b_idx]
                    b_adv = advantages_b[b_idx]

                    # Forward pass
                    _, new_logprob, entropy, new_value = self.network.get_action_and_value(b_obs, b_act)

                    logratio = new_logprob - b_logprob_old
                    ratio = torch.exp(logratio)

                    # PPO Policy surrogate loss
                    pg_loss1 = -b_adv * ratio
                    pg_loss2 = -b_adv * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value function loss
                    vf_loss = 0.5 * ((new_value - b_ret) ** 2).mean()

                    # Total loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    epoch_losses.append(loss.item())
                    epoch_pg_losses.append(pg_loss.item())
                    epoch_vf_losses.append(vf_loss.item())
                    epoch_entropies.append(entropy_loss.item())

            return {
                "loss": float(np.mean(epoch_losses)),
                "pg_loss": float(np.mean(epoch_pg_losses)),
                "vf_loss": float(np.mean(epoch_vf_losses)),
                "entropy": float(np.mean(epoch_entropies)),
            }


# ==============================================================================
# Unified Gold PPO Agent High-Level Interface
# ==============================================================================

class GoldPPOAgent:
    """
    High-level PPO Trading Agent for Gold (XAU/USD).
    Seamlessly orchestrates environment setup, rollouts, training epochs,
    checkpointing, and performance evaluation.
    """

    def __init__(
        self,
        env: Any,
        backend: str = "auto",  # 'auto', 'sb3', or 'torch'
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.hidden_dim = hidden_dim

        # Choose backend
        if backend == "auto":
            if _HAS_SB3:
                self.backend = "sb3"
            elif _HAS_TORCH:
                self.backend = "torch"
            else:
                raise RuntimeError("Neither stable-baselines3 nor PyTorch is available.")
        else:
            self.backend = backend

        self.model: Any = None
        self._init_model()

    def _init_model(self) -> None:
        if self.backend == "sb3":
            if not _HAS_SB3:
                raise ImportError("stable-baselines3 is not installed.")
            logger.info("Initializing Stable-Baselines3 PPO Agent for Gold Trading")
            self.model = SB3_PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                n_steps=1024,
                batch_size=64,
                n_epochs=4,
                verbose=0,
                device=self.device,
            )
        elif self.backend == "torch":
            if not _HAS_TORCH:
                raise ImportError("PyTorch is not installed.")
            logger.info("Initializing Native PyTorch PPO Agent for Gold Trading")
            self.model = TorchPPO(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                lr=self.learning_rate,
                gamma=self.gamma,
                clip_eps=self.clip_range,
                ent_coef=self.ent_coef,
                hidden_dim=self.hidden_dim,
                device=self.device,
            )

    def train(
        self,
        total_timesteps: int = 50000,
        eval_env: Optional[Any] = None,
        eval_interval: int = 10000,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute PPO policy optimization across specified timesteps.
        Returns training metrics history.
        """
        history: List[Dict[str, Any]] = []

        if self.backend == "sb3":
            logger.info(f"Training SB3 PPO for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)
            if eval_env:
                summary = self.evaluate(eval_env)
                history.append(summary)
            return history

        # Native PyTorch Training Loop
        logger.info(f"Training Native TorchPPO for {total_timesteps} timesteps...")
        rollout_steps = 1024
        epochs = 4
        batch_size = 64

        obs, _ = self.env.reset()
        timesteps_done = 0

        while timesteps_done < total_timesteps:
            obs_buf = []
            act_buf = []
            logprob_buf = []
            reward_buf = []
            val_buf = []
            done_buf = []

            # 1. Rollout collection
            for _ in range(rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                with torch.no_grad():
                    action, logprob, _, value = self.model.network.get_action_and_value(obs_t)

                action_int = int(action.item())
                next_obs, reward, terminated, truncated, _ = self.env.step(action_int)
                done = terminated or truncated

                obs_buf.append(obs)
                act_buf.append(action_int)
                logprob_buf.append(float(logprob.item()))
                reward_buf.append(float(reward))
                val_buf.append(float(value.item()))
                done_buf.append(float(done))

                obs = next_obs
                timesteps_done += 1

                if done:
                    obs, _ = self.env.reset()

            # 2. Compute GAE Returns & Advantages
            with torch.no_grad():
                last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                _, _, _, last_val = self.model.network.get_action_and_value(last_obs_t)
                next_value = float(last_val.item())

            advantages = np.zeros(rollout_steps, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(rollout_steps)):
                is_terminal = done_buf[t]
                next_v = next_value if t == rollout_steps - 1 else val_buf[t + 1]
                delta = reward_buf[t] + self.model.gamma * next_v * (1.0 - is_terminal) - val_buf[t]
                last_gae = delta + self.model.gamma * self.model.gae_lambda * (1.0 - is_terminal) * last_gae
                advantages[t] = last_gae

            returns = advantages + np.array(val_buf, dtype=np.float32)

            # Normalize advantages
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages_norm = (advantages - adv_mean) / adv_std

            # Convert to tensors
            obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=self.model.device)
            act_tensor = torch.as_tensor(np.array(act_buf), dtype=torch.int64, device=self.model.device)
            logp_tensor = torch.as_tensor(np.array(logprob_buf), dtype=torch.float32, device=self.model.device)
            ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.model.device)
            adv_tensor = torch.as_tensor(advantages_norm, dtype=torch.float32, device=self.model.device)

            # 3. PPO Update Step
            train_metrics = self.model.train_step(
                obs_tensor, act_tensor, logp_tensor, ret_tensor, adv_tensor, epochs=epochs, batch_size=batch_size
            )
            train_metrics["timesteps"] = timesteps_done

            # 4. Periodic Evaluation
            if eval_env and (timesteps_done % eval_interval < rollout_steps):
                eval_metrics = self.evaluate(eval_env)
                train_metrics.update(eval_metrics)
                logger.info(
                    f"Timesteps: {timesteps_done}/{total_timesteps} | "
                    f"Return: {eval_metrics.get('total_return_pct')}%, "
                    f"Sharpe: {eval_metrics.get('sharpe_ratio')}, "
                    f"WinRate: {eval_metrics.get('win_rate_pct')}%, "
                    f"Trades: {eval_metrics.get('total_trades')}"
                )

            history.append(train_metrics)
            if progress_callback:
                progress_callback(train_metrics)

        return history

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict action for given observation vector."""
        if self.backend == "sb3":
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action)
        return self.model.predict(obs, deterministic=deterministic)

    def evaluate(self, env: Any) -> Dict[str, Any]:
        """Run complete deterministic backtest over an evaluation environment."""
        obs, _ = env.reset()
        done = False
        while not done:
            action = self.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        return env.get_performance_summary()

    def save(self, filepath: str) -> None:
        """Save trained PPO model weights and configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if self.backend == "sb3":
            self.model.save(filepath)
            logger.info(f"Saved SB3 model to {filepath}")
        elif self.backend == "torch":
            torch.save(
                {
                    "network_state_dict": self.model.network.state_dict(),
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                },
                filepath,
            )
            logger.info(f"Saved TorchPPO model to {filepath}")

    def load(self, filepath: str) -> None:
        """Load trained model weights."""
        if self.backend == "sb3":
            self.model = SB3_PPO.load(filepath, env=self.env)
            logger.info(f"Loaded SB3 model from {filepath}")
        elif self.backend == "torch":
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.network.load_state_dict(checkpoint["network_state_dict"])
            self.model.network.eval()
            logger.info(f"Loaded TorchPPO model from {filepath}")
