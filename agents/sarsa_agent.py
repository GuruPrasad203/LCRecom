"""Tabular SARSA — on-policy temporal-difference control.

Update rule::

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

where ``a'`` is the action the *behaviour policy* (epsilon-greedy) will
actually take in ``s'`` — hence **on-policy**.  SARSA therefore learns the
value of the exploratory policy it follows, which usually makes it more
cautious in high-variance / high-penalty regions (fewer repeated failures).
"""

from __future__ import annotations

import numpy as np

from agents.base_tabular import TabularAgent
from environment.state import state_to_index


class SarsaAgent(TabularAgent):
    name = "SARSA"

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
        """SARSA update on *discrete* state indices (unit-testable core)."""
        if done:
            target = float(reward)
            self._cached_action = None
        else:
            # on-policy: next action from the same epsilon-greedy behaviour policy
            a2 = self.select_action(s2, explore=True)
            target = float(reward) + self.cfg.gamma * float(self.Q[s2, a2])
            self._cached_action = a2   # must be the action executed next
        self.Q[s, action] += self.cfg.alpha * (target - self.Q[s, action])
