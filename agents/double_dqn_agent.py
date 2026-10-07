"""Double DQN.

Standard DQN evaluates *and* selects the best next action with the same
target network, which systematically **overestimates** Q-values (the max of
noisy estimates is biased upward).  Double DQN decouples the two roles:

    action selection :  argmax_a  Q_online(s', a)      (online network)
    action evaluation:  Q_target(s', argmax ...)       (target network)

Everything else (architecture, replay, target updates, exploration) is
identical to :class:`agents.dqn_agent.DQNAgent`, so any difference in the
learning curves is attributable to the target computation alone.
"""

from __future__ import annotations

import torch

from agents.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    name = "Double DQN"
    is_deep = True

    @torch.no_grad()
    def _targets(self, rewards, next_states, dones) -> torch.Tensor:
        """Double-DQN target:  r + gamma * Q_target(s', argmax_a Q_online(s', a))."""
        next_actions = self.online(next_states).argmax(dim=1)          # selection
        q_next = self.target(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)                                                   # evaluation
        return rewards + self.cfg.gamma * q_next * (1.0 - dones)
