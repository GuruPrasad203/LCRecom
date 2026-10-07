"""Difficulty-based baseline.

Rule (documented heuristic, no learning):

    recent_accuracy < 0.45 or >= 2 consecutive failures  -> recommend EASY
    recent_accuracy < 0.70                               -> recommend MEDIUM
    recent_accuracy >= 0.70                              -> recommend HARD

Topic choice: the *least recently recommended* topic, so the recommender
cycles through topics instead of collapsing onto one.
"""

from __future__ import annotations

import numpy as np

from baselines.base import Recommender
from config import N_TOPICS
from environment.action_space import topic_difficulty_to_action


class DifficultyRecommender(Recommender):
    name = "Difficulty-Based"

    def __init__(self) -> None:
        self.counts = np.zeros(N_TOPICS, dtype=np.int64)
        self.last_topic: int | None = None

    def reset(self) -> None:
        self.counts[:] = 0
        self.last_topic = None

    def _choose_difficulty(self, state: np.ndarray) -> int:
        acc = self.recent_accuracy(state)
        failures = self.consecutive_failures(state)
        if acc < 0.45 or failures >= 2:
            return 1
        if acc < 0.70:
            return 2
        return 3

    def recommend(self, state: np.ndarray) -> int:
        difficulty = self._choose_difficulty(state)
        # least-recently-recommended topic (ties -> lowest index)
        topic_idx = int(np.argmin(self.counts))
        self.counts[topic_idx] += 1
        self.last_topic = topic_idx
        return topic_difficulty_to_action(topic_idx, difficulty)
