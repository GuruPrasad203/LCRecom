"""Random baseline: picks one of the 36 (topic, difficulty) categories
uniformly at random.  Serves as the floor of every comparison."""

from __future__ import annotations

import numpy as np

from baselines.base import Recommender
from config import N_ACTIONS


class RandomRecommender(Recommender):
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def recommend(self, state: np.ndarray) -> int:
        return int(self.rng.integers(N_ACTIONS))
