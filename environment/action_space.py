"""Action space of the coding-practice environment.

The action is **not** an individual problem (that would give an action space
of 180+).  It is a (topic, difficulty) *category*:

    action = topic_index * 3 + (difficulty - 1)

    0 -> Easy Arrays        1 -> Medium Arrays        2 -> Hard Arrays
    3 -> Easy Strings       4 -> Medium Strings       5 -> Hard Strings
    ...
    35 -> Hard Sorting      (12 topics x 3 difficulties = 36 actions)

Once the agent picks a category the environment samples an unattempted
problem from that category.
"""

from __future__ import annotations

from config import (
    DIFFICULTY_LABELS,
    DIFFICULTY_LEVELS,
    N_ACTIONS,
    N_DIFFICULTIES,
    N_TOPICS,
    TOPICS,
)

ACTION_LABELS: list[str] = [
    f"{DIFFICULTY_LABELS[d]} {topic}"
    for topic in TOPICS
    for d in DIFFICULTY_LEVELS
]


def action_to_topic_difficulty(action: int) -> tuple[int, int]:
    """Return (topic_index, difficulty) for a discrete action id."""
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} outside [0, {N_ACTIONS})")
    topic_idx = action // N_DIFFICULTIES
    difficulty = action % N_DIFFICULTIES + 1
    return topic_idx, difficulty


def topic_difficulty_to_action(topic_idx: int, difficulty: int) -> int:
    if not 0 <= topic_idx < N_TOPICS:
        raise ValueError(f"topic index {topic_idx} outside [0, {N_TOPICS})")
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"difficulty {difficulty} must be one of {DIFFICULTY_LEVELS}")
    return topic_idx * N_DIFFICULTIES + (difficulty - 1)


def action_label(action: int) -> str:
    return ACTION_LABELS[action]


def action_topic(action: int) -> str:
    return TOPICS[action_to_topic_difficulty(action)[0]]


def action_difficulty(action: int) -> int:
    return action_to_topic_difficulty(action)[1]
