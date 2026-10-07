"""Prerequisite-based baseline.

Follows the topic DAG defined in ``config.TOPIC_PREREQS``::

    Arrays -> {Strings, Stack, Queue, Linked Lists, Sorting, Recursion, ...}
    Recursion -> {Trees, Dynamic Programming}
    {Stack, Queue, Trees} -> Graphs
    Sorting -> {Binary Search, Greedy}

Rule
----
1. A topic is *eligible* when every prerequisite (transitively) has
   mastery >= PREREQ_MASTERY (0.35).
2. Among eligible, not-yet-mastered (< 0.90) topics recommend the
   **weakest** one -> prerequisites are learned before advanced topics.
3. Difficulty follows the student's mastery of that topic:
       m < 0.45 -> Easy,  m < 0.70 -> Medium,  else Hard,
   adjusted down after >= 2 consecutive failures and up when the student is
   on a clean, accurate run.
"""

from __future__ import annotations

import numpy as np

from baselines.base import Recommender
from config import TOPIC_PREREQS, TOPICS
from environment.action_space import topic_difficulty_to_action

PREREQ_MASTERY = 0.35
MASTERED = 0.90


def _transitive_prereqs() -> dict[str, set[str]]:
    """Ancestor set of every topic in the prerequisite DAG."""
    out: dict[str, set[str]] = {}
    for t in TOPICS:
        seen: set[str] = set()
        stack = list(TOPIC_PREREQS[t])
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(TOPIC_PREREQS.get(p, []))
        out[t] = seen
    return out


TRANSITIVE_PREREQS = _transitive_prereqs()


class PrerequisiteRecommender(Recommender):
    name = "Prerequisite-Based"

    def __init__(self) -> None:
        self.last_topic: int | None = None

    def reset(self) -> None:
        self.last_topic = None

    def _eligible(self, mastery: np.ndarray) -> list[int]:
        idxs = []
        for i, t in enumerate(TOPICS):
            prereqs = TRANSITIVE_PREREQS[t]
            if all(mastery[TOPICS.index(p)] >= PREREQ_MASTERY for p in prereqs):
                idxs.append(i)
        return idxs

    def _choose_difficulty(self, m: float, state: np.ndarray) -> int:
        difficulty = 1 if m < 0.45 else (2 if m < 0.70 else 3)
        if self.consecutive_failures(state) >= 2:
            difficulty = max(1, difficulty - 1)
        elif self.recent_accuracy(state) >= 0.85:
            difficulty = min(3, difficulty + 1)
        return difficulty

    def recommend(self, state: np.ndarray) -> int:
        mastery = self.mastery(state)
        eligible = self._eligible(mastery)
        frontier = [i for i in eligible if mastery[i] < MASTERED]
        if frontier:
            topic_idx = min(frontier, key=lambda i: (mastery[i], i))
        else:  # everything eligible is mastered -> keep practising the weakest
            topic_idx = int(np.argmin(mastery))
        difficulty = self._choose_difficulty(float(mastery[topic_idx]), state)
        self.last_topic = topic_idx
        return topic_difficulty_to_action(topic_idx, difficulty)
