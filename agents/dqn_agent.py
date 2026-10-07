"""Deep Q-Network (DQN).

Architecture (per the project specification):

    26-dim continuous state
        -> Linear(26, 128)  -> ReLU
        -> Linear(128, 128) -> ReLU
        -> Linear(128, 36)                (one Q-value per (topic, difficulty))

Training ingredients:

* experience replay buffer (50k transitions, uniform mini-batch sampling)
* target network (hard copy every ``target_update_every`` environment steps)
* epsilon-greedy exploration with linear decay
* Huber (SmoothL1) loss + Adam, gradient clipping
* warm-up phase before learning starts
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from baselines.base import Recommender
from config import DQNConfig, N_ACTIONS
from environment.state import STATE_DIM


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 hidden1: int = 128, hidden2: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done) -> None:
        i = self.pos
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
        )

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class DQNAgent(Recommender):
    name = "DQN"
    is_deep = True

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

        self.online = QNetwork(hidden1=self.cfg.hidden1, hidden2=self.cfg.hidden2).to(self.device)
        self.target = QNetwork(hidden1=self.cfg.hidden1, hidden2=self.cfg.hidden2).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(self.cfg.buffer_size, STATE_DIM, seed=seed)
        self.rng = np.random.default_rng(seed + 1)

        self.epsilon = self.cfg.epsilon_start
        self.steps = 0
        self.updates = 0
        self.last_loss = 0.0

    # ------------------------------------------------------------ policy
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return self._greedy(state)

    def recommend(self, state: np.ndarray) -> int:
        """Greedy (evaluation) action."""
        return self._greedy(state)

    @torch.no_grad()
    def _greedy(self, state: np.ndarray) -> int:
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x).squeeze(0).cpu().numpy()

    # -------------------------------------------------------- transitions
    def begin_episode(self) -> None:
        return None

    def end_episode(self) -> None:
        return None

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1

        frac = min(1.0, self.steps / max(self.cfg.epsilon_decay_steps, 1))
        self.epsilon = self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

        if len(self.buffer) >= max(self.cfg.warmup_steps, self.cfg.batch_size) \
                and self.steps % self.cfg.train_every == 0:
            self.train_step()
        if self.steps % self.cfg.target_update_every == 0:
            self.update_target()

    # ----------------------------------------------------------- learning
    def _targets(self, rewards, next_states, dones) -> torch.Tensor:
        """DQN target:  r + gamma * max_a' Q_target(s', a')."""
        max_next = self.target(next_states).max(dim=1).values
        return rewards + self.cfg.gamma * max_next * (1.0 - dones)

    def train_step(self) -> float:
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = self._targets(r, s2, d)
        loss = self.loss_fn(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.updates += 1
        self.last_loss = float(loss.item())
        return self.last_loss

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # --------------------------------------------------- checkpointing
    def snapshot(self) -> dict:
        return {
            "online": {k: v.detach().clone() for k, v in self.online.state_dict().items()},
            "target": {k: v.detach().clone() for k, v in self.target.state_dict().items()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "updates": self.updates,
        }

    def restore(self, snap: dict) -> None:
        self.online.load_state_dict(snap["online"])
        self.target.load_state_dict(snap["target"])
        self.optimizer.load_state_dict(snap["optimizer"])
        self.epsilon = snap["epsilon"]
        self.steps = snap["steps"]
        self.updates = snap["updates"]

    # ------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "updates": self.updates,
                "name": self.name,
                "config": self.cfg.__dict__,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = DQNConfig(**payload.get("config", {}))
        agent = cls(config=cfg, device=device)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.epsilon = float(payload["epsilon"])
        agent.steps = int(payload["steps"])
        agent.updates = int(payload["updates"])
        return agent
