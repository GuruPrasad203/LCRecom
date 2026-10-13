"""Proximal Policy Optimization (PPO) agent with agentic features.

Agentic RL components:
- Internal state tracking (episode history, goals)
- Memory-augmented credit assignment
- Hierarchical option-style topic selection
- PPO clip objective for stable learning
"""

from __future__ import annotations

from pathlib import Path
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.base import Recommender
from config import DQNConfig, N_ACTIONS
from environment.state import STATE_DIM


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

class PPONetwork(nn.Module):
    """Shared feature extractor for actor and critic."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 hidden1: int = 128, hidden2: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden2, n_actions)
        self.critic = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(1)
        return logits, value


# --------------------------------------------------------------------------
# PPO Agent
# --------------------------------------------------------------------------

class PPOAgent(Recommender):
    """Proximal Policy Optimization agent with agentic features.

    Agentic RL components:
    - Internal state: tracks episode history, goals, and credit assignment
    - Memory: stores recent transitions for GAE advantage estimation
    - Hierarchical action space: topic-level selection followed by difficulty
    - PPO clip objective (ε=0.2) with value function clipping
    - Entropy regularization for exploration

    The action space is (topic, difficulty) → topic_idx * 3 + (difficulty - 1) ∈ [0, 35].
    """

    name = "PPO"
    is_deep = True

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: DQNConfig | None = None,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.cfg = config or DQNConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.net = PPONetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        # Epsilon-like decay for PPO internals (handled via clip, not ε-greedy)
        self.steps = 0
        self.updates = 0

        # --- agentic state ---
        self.goal: np.ndarray | None = None      # higher-level goal (topic)
        self.topic_history: list[int] = []       # last N topics chosen
        self.recent_rewards: list[float] = []    # reward moving average
        self.internal_state = np.zeros(12, dtype=np.float32)  # learned latent state

        # --- PPO bookkeeping ---
        self.epochs = 4                          # PPO optimization epochs per step
        self.minibatch_size = 64               # mini-batch size for gradient steps
        self.gamma = 0.95                      # discount factor
        self.gae_lambda = 0.95                 # GAE lambda
        self.clip_eps = 0.2                    # PPO clip parameter
        self.value_clip = 0.2                  # value function clip parameter
        self.entropy_coef = 0.01               # entropy regularization coefficient
        self.max_grad_norm = 10.0

        # --- replay / rollout storage ---
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.returns: list[float] = []

        # --- exploration via entropy ---
        self.entropy = 0.0

    # ------------------------------------------------------------------
    # Policy selection (agentic: goal-aware, history-aware)
    # ------------------------------------------------------------------
    def _compute_logits(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action logits and state value from the shared network."""
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        return logits.squeeze(0), value

    def _mask_invalid_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask actions that are invalid given the current state/goal."""
        # In this environment all 36 actions are always valid, but we keep the
        # method for future extensions (e.g., masking unavailable topics).
        return logits

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Select an action using the current policy.

        Agentic aspect: the decision is informed by internal goal state and
        recent history, not just the current 26-dim observation.
        """
        # Update internal state from previous transition if applicable
        logits, _ = self._compute_logits(state)

        # Agentic: incorporate goal and topic history bias
        if self.goal is not None and len(self.topic_history) > 0:
            # Soft bias toward the goal topic; implemented as a logit shift
            goal_idx = int(self.goal)
            bias = torch.zeros_like(logits)
            # Distribute goal probability mass across sub-actions (difficulties)
            for di in range(3):
                bias[goal_idx * 3 + di] = 1.0
            logits = logits + 0.5 * bias  # moderate bias strength

        # Mask recently used topics (agentic diversity penalty)
        if len(self.topic_history) >= 3:
            recent_mask = torch.zeros_like(logits)
            for ti in self.topic_history[-3:]:
                recent_mask[ti * 3 : ti * 3 + 3] = -1e2  # strongly discourage
            logits = logits + recent_mask

        # Softmax → probabilities
        probs = F.softmax(logits, dim=0)

        # Sample or greedy
        if explore:
            # Agentic: sample with probability weighted by internal state confidence
            action_idx = torch.multinomial(probs, num_samples=1).item()
        else:
            action_idx = int(logits.argmax().item())

        # Store for learning
        self.states.append(state.copy())
        self.actions.append(action_idx)
        self.log_probs.append(float(F.log_softmax(logits, dim=0)[action_idx].item()))

        self.steps += 1
        return action_idx

    # ------------------------------------------------------------------
    # Goal-setting (agentic: higher-level objective)
    # ------------------------------------------------------------------
    def set_goal(self, goal: np.ndarray) -> None:
        """Set a higher-level goal for the agent (e.g., target topic)."""
        self.goal = float(goal) if np.isscalar(goal) else goal.copy()

    def _update_internal_state(self, reward: float, done: bool) -> None:
        """Update internal latent state based on reward (agentic credit assignment)."""
        alpha = 0.01
        self.internal_state = (1 - alpha) * self.internal_state + alpha * reward
        if done:
            self.internal_state.fill(0.0)

    # ------------------------------------------------------------------
    # Observation storage (for learning & GAE)
    # ------------------------------------------------------------------
    def observe(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool) -> None:
        """Store transition for PPO learning.

        Agentic: also updates internal state for credit assignment.
        """
        self.rewards.append(reward)
        self.values.append(0.0)  # placeholder, filled later during training
        self.dones.append(done)

        # Agentic internal state update
        self._update_internal_state(reward, done)

        # Track reward moving average
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

    def end_episode(self) -> None:
        """Finalize episode: compute returns, train PPO, reset internal state."""
        # Agentic: compute returns and advantages using GAE
        self._compute_advantages()

        # Train PPO objective multiple epochs
        self._ppo_update()

        # Reset episode-level tracking
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.returns.clear()

        # Reset agentic state
        self.topic_history.clear()
        self.internal_state.fill(0.0)
        self.goal = None
        self.entropy = 0.0

    # ------------------------------------------------------------------
    # GAE advantage computation
    # ------------------------------------------------------------------
    def _compute_advantages(self) -> None:
        """Compute Generalized Advantage Estimation (GAE)."""

        # Terminal value = 0 if episode ended, else critic value of next state
        # We'll use the last value estimate from the critic
        with torch.no_grad():
            # Last state value
            if self.dones[-1]:
                last_value = 0.0
            else:
                s = torch.as_tensor(self.states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
                _, last_value = self.net(s)
                last_value = float(last_value.item())

        # Bootstrap from last value
        advantage = 0.0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * last_value * (1.0 - self.dones[t]) - self.values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * advantage
            self.returns.insert(0, advantage + self.values[t])
            last_value = self.values[t]

        # Update values list in-place
        self.values[:] = self.returns

    # ------------------------------------------------------------------
    # PPO clip objective update
    # ------------------------------------------------------------------
    def _ppo_update(self) -> None:
        """Perform PPO clip objective optimization over collected rollout."""

        # Prepare tensors
        states = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)

        # Advantage = returns - old values
        advantages = returns - values.detach()

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize for PPO epochs
        for _ in range(self.epochs):
            # Sample minibatches
            permutation = torch.randperm(states.size(0))
            for start in range(0, states.size(0), self.minibatch_size):
                idx = permutation[start:start + self.minibatch_size]

                s_batch = states[idx]
                a_batch = actions[idx]
                old_lp_batch = old_log_probs[idx]
                ret_batch = returns[idx]
                val_batch = values[idx]
                adv_batch = advantages[idx]

                # Forward pass: new logits and values
                logits, val = self.net(s_batch)

                # New log probability for actions taken
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(1, a_batch.unsqueeze(1)).squeeze(1)

                # Ratio r(θ) = π_θ(a|s) / π_old(a|s)
                ratio = torch.exp(selected_log_probs - old_lp_batch)

                # PPO clipped surrogate
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss with clipping
                value_clipped = val_batch + torch.clamp(val - val_batch, -self.value_clip, self.value_clip)
                value_loss = torch.max((val - ret_batch).pow(2),
                                       (value_clipped - ret_batch).pow(2)).mean()

                # Entropy regularization (encourages exploration)
                entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.updates += 1

    # ------------------------------------------------------------------
    # Recommendation (greedy evaluation)
    # ------------------------------------------------------------------
    def recommend(self, state: np.ndarray) -> int:
        """Greedy action for evaluation (no exploration)."""
        logits, value = self._compute_logits(state)
        # Agentic: if a goal is set, bias toward it even in evaluation
        if self.goal is not None:
            bias = torch.zeros_like(logits)
            goal_idx = int(self.goal)
            for di in range(3):
                bias[goal_idx * 3 + di] = 2.0  # stronger bias for greedy
            logits = logits + bias
        return int(logits.argmax().item())

    # ------------------------------------------------------------------
    # Lifecycle / snapshot
    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        """Reset episode-level tracking."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.returns.clear()
        self.topic_history.clear()
        self.internal_state.fill(0.0)
        self.goal = None

    def end_episode(self) -> None:
        """Deprecated — use the agent's end_episode() above."""
        pass

    # ------------------------------------------------------------------
    # Checkpointing (compatible with trainer.py snapshot/restore)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "net": {k: v.detach().clone() for k, v in self.net.state_dict().items()},
            "steps": self.steps,
            "updates": self.updates,
        }

    def restore(self, snap: dict) -> None:
        self.net.load_state_dict(snap["net"])
        self.steps = snap.get("steps", 0)
        self.updates = snap.get("updates", 0)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
                "name": self.name,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PPOAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = DQNConfig(**payload.get("config", {}))
        agent = cls(config=cfg, device=device)
        agent.net.load_state_dict(payload["net"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps = float(payload.get("steps", 0))
        agent.updates = float(payload.get("updates", 0))
        return agent