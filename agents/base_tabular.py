"""Shared machinery for tabular agents (SARSA / Q-Learning).

Both agents operate on the **discrete** state produced by
``environment.state.state_to_index`` (972 states) and the 36
(topic, difficulty) actions -> a Q-table of 972 x 36 = 34,992 values.

Unified agent protocol (used by the trainer, the evaluator and the demo):

    agent.begin_episode()          # epsilon bookkeeping
    agent.act(state, explore=True) # epsilon-greedy action
    agent.observe(s, a, r, s2, done)
    agent.end_episode()            # epsilon decay
    agent.recommend(state)         # greedy (evaluation) action
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.base import Recommender
from config import N_ACTIONS, TabularConfig
from environment.state import TABULAR_N_STATES, state_to_index


class TabularAgent(Recommender):
    """Epsilon-greedy Q-function tabular base class."""

    def __init__(
        self,
        config: TabularConfig | None = None,
        n_states: int = TABULAR_N_STATES,
        n_actions: int = N_ACTIONS,
        seed: int = 0,
    ) -> None:
        self.cfg = config or TabularConfig()
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.epsilon = self.cfg.epsilon_start
        self.episode = 0
        self._cached_action: int | None = None

    # -------------------------------------------------------------- policy
    def select_action(self, state_idx: int, explore: bool = True) -> int:
        """Epsilon-greedy action for a *discrete* state index."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        row = self.Q[state_idx]
        best = row.max()
        # deterministic tie-break: smallest action index
        return int(np.flatnonzero(row == best)[0])

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        s = state_to_index(state)
        if explore and self._cached_action is not None:
            # SARSA: reuse the action chosen during the previous update so the
            # executed action and the bootstrapped action are identical.
            a = self._cached_action
            self._cached_action = None
            return a
        return self.select_action(s, explore=explore)

    def recommend(self, state: np.ndarray) -> int:
        """Greedy action used at evaluation time (no exploration)."""
        return self.select_action(state_to_index(state), explore=False)

    # ----------------------------------------------------------- lifecycle
    def begin_episode(self) -> None:
        self._cached_action = None

    def end_episode(self) -> None:
        self.episode += 1
        d = self.cfg
        frac = min(1.0, self.episode / max(d.epsilon_decay_episodes, 1))
        self.epsilon = d.epsilon_start + frac * (d.epsilon_end - d.epsilon_start)

    # ---------------------------------------------------------- transitions
    def observe(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        raise NotImplementedError

    def _target(self, reward: float, next_state_idx: int, done: bool) -> float:
        raise NotImplementedError

    # --------------------------------------------------- checkpointing
    def snapshot(self) -> dict:
        return {"Q": self.Q.copy(), "epsilon": self.epsilon, "episode": self.episode}

    def restore(self, snap: dict) -> None:
        self.Q[:] = snap["Q"]
        self.epsilon = snap["epsilon"]
        self.episode = snap["episode"]

    # ------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            Q=self.Q,
            alpha=self.cfg.alpha,
            gamma=self.cfg.gamma,
            epsilon=self.epsilon,
            episode=self.episode,
            name=self.name,
        )

    @classmethod
    def load(cls, path: str | Path, config: TabularConfig | None = None) -> "TabularAgent":
        data = np.load(path, allow_pickle=True)
        agent = cls(config=config)
        agent.Q = data["Q"]
        agent.epsilon = float(data["epsilon"])
        agent.episode = int(data["episode"])
        return agent

    @property
    def table_size(self) -> tuple[int, int]:
        return self.Q.shape
