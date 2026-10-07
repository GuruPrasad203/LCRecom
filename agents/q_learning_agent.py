"""Tabular Q-Learning — off-policy temporal-difference control.

Update rule::

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

The bootstrap uses the *optimal* next action regardless of what the behaviour
policy actually does — hence **off-policy**.  Q-Learning converges (tabular,
finite MDP, sufficient exploration) to the optimal policy, but during training
it can be more aggressive/optimistic than SARSA.
"""

from __future__ import annotations

import numpy as np

from agents.base_tabular import TabularAgent
from environment.state import state_to_index


class QLearningAgent(TabularAgent):
    name = "Q-Learning"

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.update(
            state_to_index(state), action, float(reward), state_to_index(next_state), done
        )

    def update(self, s: int, action: int, reward: float, s2: int, done: bool) -> None:
        """Q-Learning update on *discrete* state indices (unit-testable core)."""
        target = float(reward) if done else float(reward) + self.cfg.gamma * float(self.Q[s2].max())
        self.Q[s, action] += self.cfg.alpha * (target - self.Q[s, action])
