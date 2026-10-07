"""Common interface for all recommendation strategies (baselines + RL).

Every strategy sees *exactly the same information*: the 26-dim observation
vector built by the environment.  This guarantees a fair comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from environment.state import (
    IDX_CONSEC_FAIL,
    IDX_RECENT_ACCURACY,
    MASTERY_SLICE,
)


class Recommender(ABC):
    """A strategy that maps an observation to one of the 36 actions."""

    name: str = "base"

    def reset(self) -> None:  # called at the start of every episode
        return None

    @abstractmethod
    def recommend(self, state: np.ndarray) -> int: ...

    # ---- shared helpers ------------------------------------------------
    @staticmethod
    def mastery(state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64)[MASTERY_SLICE]

    @staticmethod
    def recent_accuracy(state: np.ndarray) -> float:
        return float(state[IDX_RECENT_ACCURACY])

    @staticmethod
    def consecutive_failures(state: np.ndarray) -> int:
        return int(round(float(state[IDX_CONSEC_FAIL]) * 5))
