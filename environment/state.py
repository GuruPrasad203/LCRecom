"""State representation of the coding-practice environment.

The observation handed to the RL agent is a compact, normalised vector:

=========================  ========  ==========================================
Idx  Feature               Range     Meaning
=========================  ========  ==========================================
0-11  mastery/<topic>      [0, 1]    EWMA mastery of each of the 12 topics
12    recent_accuracy      [0, 1]    success rate over the last HISTORY_WINDOW
                                     attempts
13    avg_solve_time_norm  [0, 1]    mean(solve_time / expected_time)/2 over the
                                     same window (0.5 == on-time, 1 == 2x late)
14    recent_hint_usage    [0, 1]    mean hints per attempt / MAX_HINTS
15    recent_error_rate    [0, 1]    wrong submissions per attempt in window
16    current_difficulty   [0, 1]    (difficulty - 1) / 2 of the last problem
17    consecutive_failures [0, 1]    min(failures, 5) / 5
18    frustration          [0, 1]    student frustration level
19    same_topic_streak     [0, 1]    min(consecutive recommendations of the
                                     same topic, 5) / 5 -- the reward penalises
                                     massed practice, so this MUST be observed
                                     for the state to be Markov
20-25  error/<type>        [0, 1]    distribution of the 6 error types inside
                                     the last N_ERROR_WINDOW attempts
                                     (sums to 0 or 1)
=========================  ========  ==========================================

The *tabular* agents (SARSA, Q-Learning) cannot use 26 continuous dimensions,
so ``state_to_index`` coarsens the vector into a small discrete state that
still captures the decision-relevant factors (see ``TABULAR_SHAPE``).
The deep agents consume the full 26-dim vector.
"""

from __future__ import annotations

import numpy as np

from config import ERROR_TYPES, N_ERROR_TYPES, N_TOPICS, TOPICS

# --------------------------------------------------------------------------
# Full (continuous) state
# --------------------------------------------------------------------------
FEATURE_NAMES: list[str] = (
    [f"mastery/{t}" for t in TOPICS]
    + [
        "recent_accuracy",
        "avg_solve_time_norm",
        "recent_hint_usage",
        "recent_error_rate",
        "current_difficulty_norm",
        "consecutive_failures_norm",
        "frustration",
        "same_topic_streak_norm",
    ]
    + [f"error/{e}" for e in ERROR_TYPES]
)

STATE_DIM = len(FEATURE_NAMES)  # 12 + 8 + 6 = 26

MASTERY_SLICE = slice(0, N_TOPICS)
IDX_RECENT_ACCURACY = 12
IDX_AVG_TIME = 13
IDX_HINT_USAGE = 14
IDX_ERROR_RATE = 15
IDX_DIFFICULTY = 16
IDX_CONSEC_FAIL = 17
IDX_FRUSTRATION = 18
IDX_STREAK = 19
ERROR_SLICE = slice(20, 20 + N_ERROR_TYPES)

# --------------------------------------------------------------------------
# Discrete (tabular) state
# --------------------------------------------------------------------------
# bins: skill(3) x accuracy(3) x consecutive failures(3) x difficulty(3)
#       x weak-topic fraction(2) x hint usage(2) x same-topic streak(3)
#       = 972 states   (the streak bin keeps the tabular agents on equal
#                       footing with the deep agents, at the cost of a 3x
#                       larger table -> exactly the state-space growth that
#                       motivates DQN, see RQ3)
TABULAR_SHAPE: tuple[int, ...] = (3, 3, 3, 3, 2, 2, 3)
TABULAR_N_STATES = int(np.prod(TABULAR_SHAPE))
TABULAR_FEATURES = (
    "skill_level",
    "recent_accuracy",
    "consecutive_failures",
    "current_difficulty",
    "weak_topic_fraction",
    "recent_hint_usage",
    "same_topic_streak",
)


def _bin3(x: float, e1: float, e2: float) -> int:
    return 0 if x < e1 else (1 if x < e2 else 2)


def state_to_index(state: np.ndarray) -> int:
    """Coarsen the 26-dim observation into one of TABULAR_N_STATES bins."""
    state = np.asarray(state, dtype=np.float64)
    mastery = state[MASTERY_SLICE]
    skill = float(mastery.mean())
    weak_frac = float((mastery < 0.45).mean())

    bins = (
        _bin3(skill, 0.35, 0.60),                       # skill level
        _bin3(float(state[IDX_RECENT_ACCURACY]), 0.35, 0.65),
        min(int(round(state[IDX_CONSEC_FAIL] * 5)), 2),  # 0 / 1 / 2+
        int(round(float(state[IDX_DIFFICULTY]) * 2)),    # 0 / 1 / 2
        int(weak_frac >= 0.50),                          # many weak topics?
        int(float(state[IDX_HINT_USAGE]) >= 0.50),
        min(int(round(state[IDX_STREAK] * 5)), 2),       # 0 / 1 / 2+ repeats
    )
    idx = 0
    for b, dim in zip(bins, TABULAR_SHAPE):
        idx = idx * dim + b
    return int(idx)


def describe_state(state: np.ndarray) -> dict:
    """Human-readable view of an observation (used by the demo / logging)."""
    state = np.asarray(state, dtype=np.float64)
    mastery = {t: round(float(state[i]), 3) for i, t in enumerate(TOPICS)}
    errors = {
        e: round(float(state[ERROR_SLICE.start + i]), 3)
        for i, e in enumerate(ERROR_TYPES)
    }
    errors = {k: v for k, v in errors.items() if v > 0}
    return {
        "topic_mastery": mastery,
        "recent_accuracy": round(float(state[IDX_RECENT_ACCURACY]), 3),
        "avg_solve_time_ratio": round(float(state[IDX_AVG_TIME]) * 2.0, 3),
        "recent_hint_usage": round(float(state[IDX_HINT_USAGE]), 3),
        "recent_error_rate": round(float(state[IDX_ERROR_RATE]), 3),
        "current_difficulty": int(round(float(state[IDX_DIFFICULTY]) * 2)) + 1,
        "consecutive_failures": int(round(float(state[IDX_CONSEC_FAIL]) * 5)),
        "frustration": round(float(state[IDX_FRUSTRATION]), 3),
        "same_topic_streak": int(round(float(state[IDX_STREAK]) * 5)),
        "recent_error_types": errors,
        "tabular_state_index": state_to_index(state),
    }


def empty_state() -> np.ndarray:
    return np.zeros(STATE_DIM, dtype=np.float32)
