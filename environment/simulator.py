"""Student behaviour simulator.

The simulator turns an (topic, difficulty) recommendation into a realistic
attempt outcome for a *profiled* student:

    success / failure, solve time, hints used, error type, skill gain.

The model is deliberately *not* uniform-random: the probability of solving a
problem depends on the student's mastery of the topic, the problem difficulty,
the student's ability, frustration and failure streak (see ``success_probability``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from config import (
    DIFFICULTY_NORM,
    ERROR_TYPES,
    MAX_HINTS,
    TOPICS,
)

# Difficulty multiplier for how much skill a problem teaches when solved.
DIFF_GAIN_FACTOR: dict[int, float] = {1: 0.70, 2: 1.00, 3: 1.25}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --------------------------------------------------------------------------
# Student profiles
# --------------------------------------------------------------------------
@dataclass
class StudentProfile:
    """A simulated learner.

    ability                  : baseline problem-solving strength, [0, 1]
    learning_rate            : how fast mastery grows per attempt, [0.5, 1.5]
    frustration_sensitivity  : frustration accumulated per failure, [0.3, 1.6]
    hint_inclination         : propensity to ask for hints when stuck, [0, 1]
    time_factor              : relative slowness, 1.0 = normal, [0.7, 1.5]
    error_bias               : multiplicative bias per error type
    mastery                  : current mastery per topic, all in [0, 1]
    """

    student_id: str
    name: str
    profile: str
    split: str
    ability: float
    learning_rate: float
    frustration_sensitivity: float
    hint_inclination: float
    time_factor: float
    error_bias: dict[str, float]
    mastery: dict[str, float]

    def mastery_array(self) -> np.ndarray:
        return np.array([self.mastery[t] for t in TOPICS], dtype=np.float64)

    def with_mastery(self, mastery: dict[str, float]) -> "StudentProfile":
        return replace(self, mastery=dict(mastery))


# Global "how easy students find this topic" bias used when seeding mastery.
TOPIC_MASTERY_BIAS: dict[str, float] = {
    "Arrays": 0.10,
    "Strings": 0.05,
    "Stack": 0.00,
    "Queue": 0.00,
    "Linked Lists": -0.05,
    "Sorting": -0.02,
    "Binary Search": -0.05,
    "Recursion": -0.10,
    "Trees": -0.12,
    "Graphs": -0.18,
    "Dynamic Programming": -0.20,
    "Greedy": -0.08,
}

# name -> (mastery_level, ability, learning_rate, frustration_sensitivity,
#          hint_inclination, time_factor, error_bias)
PROFILE_TEMPLATES: dict[str, dict] = {
    "beginner": dict(
        level=0.30, ability=0.35, learning_rate=0.90, frustration=0.90,
        hints=0.80, time_factor=1.20,
        error_bias={"Syntax Error": 1.5, "Conceptual Error": 1.3, "Edge Case Error": 0.7},
    ),
    "intermediate": dict(
        level=0.55, ability=0.55, learning_rate=1.00, frustration=0.70,
        hints=0.50, time_factor=1.00,
        error_bias={"Logic Error": 1.2, "Edge Case Error": 1.1},
    ),
    "advanced": dict(
        level=0.80, ability=0.80, learning_rate=1.10, frustration=0.50,
        hints=0.20, time_factor=0.85,
        error_bias={"Edge Case Error": 1.5, "Time Complexity Error": 1.3,
                    "Syntax Error": 0.5, "Conceptual Error": 0.5},
    ),
    "fast_learner": dict(
        level=0.35, ability=0.60, learning_rate=1.50, frustration=0.60,
        hints=0.30, time_factor=0.80,
        error_bias={"Edge Case Error": 1.1},
    ),
    "slow_learner": dict(
        level=0.32, ability=0.40, learning_rate=0.55, frustration=0.95,
        hints=0.70, time_factor=1.40,
        error_bias={"Conceptual Error": 1.4, "Implementation Error": 1.2},
    ),
    "high_frustration": dict(
        level=0.45, ability=0.50, learning_rate=0.90, frustration=1.50,
        hints=0.60, time_factor=1.10,
        error_bias={"Implementation Error": 1.6, "Syntax Error": 1.4},
    ),
    "low_frustration": dict(
        level=0.48, ability=0.50, learning_rate=1.00, frustration=0.35,
        hints=0.40, time_factor=1.00,
        error_bias={"Logic Error": 1.1},
    ),
    "strong_problem_solver": dict(
        level=0.55, ability=0.85, learning_rate=1.10, frustration=0.50,
        hints=0.20, time_factor=0.90,
        error_bias={"Edge Case Error": 1.4, "Time Complexity Error": 1.2,
                    "Syntax Error": 0.5},
    ),
    "weak_problem_solver": dict(
        level=0.30, ability=0.28, learning_rate=0.80, frustration=1.10,
        hints=0.90, time_factor=1.30,
        error_bias={"Conceptual Error": 1.6, "Implementation Error": 1.3},
    ),
}


def seed_mastery(profile_name: str, rng: np.random.Generator) -> dict[str, float]:
    """Draw an initial per-topic mastery vector for a profile."""
    tpl = PROFILE_TEMPLATES[profile_name]
    mastery = {}
    for topic in TOPICS:
        m = tpl["level"] + TOPIC_MASTERY_BIAS[topic] + rng.normal(0.0, 0.08)
        mastery[topic] = float(np.clip(m, 0.02, 0.95))
    return mastery


def build_profile(
    profile_name: str,
    student_id: str,
    name: str,
    split: str,
    rng: np.random.Generator,
    jitter: float = 0.05,
) -> StudentProfile:
    tpl = PROFILE_TEMPLATES[profile_name]

    def j(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(np.clip(v + rng.normal(0.0, jitter), lo, hi))

    return StudentProfile(
        student_id=student_id,
        name=name,
        profile=profile_name,
        split=split,
        ability=j(tpl["ability"]),
        learning_rate=float(np.clip(tpl["learning_rate"] + rng.normal(0.0, jitter), 0.4, 1.8)),
        frustration_sensitivity=float(np.clip(tpl["frustration"] + rng.normal(0.0, jitter), 0.2, 1.8)),
        hint_inclination=j(tpl["hints"]),
        time_factor=float(np.clip(tpl["time_factor"] + rng.normal(0.0, jitter), 0.6, 1.8)),
        error_bias=dict(tpl["error_bias"]),
        mastery=seed_mastery(profile_name, rng),
    )


def build_profile_from_metrics(
    student_id: str,
    name: str,
    split: str,
    ability: float,
    learning_rate: float,
    frustration_sensitivity: float,
    hint_inclination: float,
    time_factor: float,
    mastery: dict[str, float],
    error_bias: dict[str, float] | None = None,
) -> StudentProfile:
    """Build a StudentProfile from explicit proficiency metrics (e.g. from LeetCode).

    This is useful for initializing a student from external data sources
    while reusing the simulator's attempt dynamics unchanged.
    """
    if error_bias is None:
        error_bias = {"Syntax Error": 1.0, "Logic Error": 1.2, "Conceptual Error": 1.0,
                      "Time Complexity Error": 0.6, "Edge Case Error": 0.8, "Implementation Error": 1.0}

    return StudentProfile(
        student_id=student_id,
        name=name,
        profile="leetcode_import",
        split=split,
        ability=ability,
        learning_rate=learning_rate,
        frustration_sensitivity=frustration_sensitivity,
        hint_inclination=hint_inclination,
        time_factor=time_factor,
        error_bias=error_bias,
        mastery=mastery,
    )


# --------------------------------------------------------------------------
# Attempt simulation
# --------------------------------------------------------------------------
@dataclass
class AttemptResult:
    success: bool
    solve_time: float          # minutes
    hints_used: int
    error_type: str | None     # dominant error type of the attempt (None = clean)
    wrong_submissions: int     # wrong submissions before passing (0 = first try)
    p_base: float              # success probability *without* hints
    p_final: float             # success probability actually used


def success_probability(
    mastery: float,
    ability: float,
    difficulty: int,
    frustration: float,
    consecutive_failures: int,
) -> float:
    """Probability that the student solves the problem *before* using hints.

    Higher mastery and ability increase it, higher difficulty, frustration and
    an ongoing failure streak decrease it.
    """
    d = DIFFICULTY_NORM[difficulty]
    z = 2.6 * (
        mastery
        + 0.8 * ability
        - d
        - 0.25 * frustration
        - 0.10 * consecutive_failures
    ) + 0.25
    return float(np.clip(_sigmoid(z), 0.02, 0.98))


def _error_weights(
    profile: StudentProfile,
    mastery: float,
    difficulty: int,
    p_base: float,
    frustration: float,
    topic: str,
) -> np.ndarray:
    """Relative likelihood of each error type for this attempt."""
    w = {
        "Syntax Error": 1.0,
        "Logic Error": 1.2,
        "Conceptual Error": 1.0,
        "Time Complexity Error": 0.6,
        "Edge Case Error": 0.8,
        "Implementation Error": 1.0,
    }
    gap = max(0.0, DIFFICULTY_NORM[difficulty] - mastery)  # problem beyond ability
    w["Conceptual Error"] += 2.0 * gap
    w["Logic Error"] += 1.0 * max(0.0, gap - 0.20)

    if topic in ("Dynamic Programming", "Graphs", "Trees", "Recursion"):
        w["Time Complexity Error"] += 0.35 + 0.70 * (difficulty - 1) / 2
    if topic in ("Arrays", "Strings", "Sorting", "Binary Search"):
        w["Edge Case Error"] += 0.50
    if topic in ("Linked Lists", "Stack", "Queue"):
        w["Implementation Error"] += 0.40

    if mastery > 0.60:                       # near-miss failures
        w["Edge Case Error"] += 0.70
        w["Syntax Error"] = max(0.15, w["Syntax Error"] - 0.50)
    if p_base > 0.55:
        w["Logic Error"] += 0.30            # failed despite knowing the topic
    if frustration > 0.50:                   # rushed / sloppy when annoyed
        w["Implementation Error"] += 0.80 * frustration
        w["Syntax Error"] += 0.50 * frustration

    for err, mult in profile.error_bias.items():
        w[err] *= mult

    return np.array([max(1e-6, w[e]) for e in ERROR_TYPES], dtype=np.float64)


def simulate_attempt(
    profile: StudentProfile,
    topic: str,
    mastery: float,
    difficulty: int,
    expected_time: float,
    frustration: float,
    consecutive_failures: int,
    rng: np.random.Generator,
) -> AttemptResult:
    """Simulate one attempt of one coding problem."""
    p_base = success_probability(
        mastery, profile.ability, difficulty, frustration, consecutive_failures
    )

    # ---- optional hints -------------------------------------------------
    hints = 0
    if p_base < 0.65:
        urgency = profile.hint_inclination * (0.45 + 0.55 * frustration)
        urgency += 0.25 * (0.65 - p_base) / 0.65
        urgency = float(np.clip(urgency, 0.0, 1.0))
        if rng.random() < urgency:
            needed = (0.65 - p_base) / 0.13
            hints = int(min(MAX_HINTS, max(1, math.ceil(needed))))

    p_final = float(min(0.98, p_base + 0.13 * hints))
    success = bool(rng.random() < p_final)

    # ---- solve time -----------------------------------------------------
    speed = (1.35 - 0.35 * mastery) / (0.55 + 0.45 * profile.ability)
    speed *= profile.time_factor
    if not success:
        speed *= 1.30                       # struggled then gave up
    if hints:
        speed *= 1.0 - 0.08 * hints         # hints speed things up
    ratio = float(np.clip(speed * rng.lognormal(0.0, 0.20), 0.35, 3.0))
    solve_time = expected_time * ratio

    # ---- wrong submissions / error type --------------------------------
    if success:
        n_wrong = 0
        if rng.random() < 0.30 + 0.35 * (1.0 - p_final):
            n_wrong = 1 + (1 if rng.random() < 0.25 else 0)
    else:
        n_wrong = 1 + (1 if rng.random() < 0.45 else 0)

    error_type = None
    if n_wrong > 0:
        weights = _error_weights(profile, mastery, difficulty, p_base, frustration, topic)
        error_type = str(rng.choice(ERROR_TYPES, p=weights / weights.sum()))

    return AttemptResult(
        success=success,
        solve_time=float(solve_time),
        hints_used=hints,
        error_type=error_type,
        wrong_submissions=int(n_wrong),
        p_base=float(p_base),
        p_final=float(p_final),
    )


# --------------------------------------------------------------------------
# Learning dynamics
# --------------------------------------------------------------------------
def compute_skill_gain(
    success: bool,
    mastery: float,
    difficulty: int,
    ability: float,
    learning_rate: float,
    hints: int,
    frustration: float,
    challenge: str,
) -> float:
    """Mastery increase produced by one attempt (always >= 0).

    Successful attempts teach more (scaled by difficulty and inversely by
    current mastery -> diminishing returns).  Failed attempts still teach a
    little, but only when the challenge was appropriate; repeatedly failing
    something far beyond the student's ability teaches almost nothing.
    """
    base = 0.045 * learning_rate * ability * DIFF_GAIN_FACTOR[difficulty]
    if success:
        gain = (
            base
            * (1.0 - mastery)
            * (1.0 - 0.15 * hints)
            * (1.0 - 0.25 * frustration)
        )
    else:
        factor = {"appropriate": 0.25, "too_easy": 0.30, "too_hard": 0.08}[challenge]
        gain = base * factor * (1.0 - mastery)
    return float(max(0.0, gain))


def update_frustration(
    frustration: float,
    success: bool,
    difficulty: int,
    mastery: float,
    sensitivity: float,
) -> float:
    """Frustration rises on (hard) failures and decays on success."""
    if success:
        return float(max(0.0, frustration - 0.10))
    d = DIFFICULTY_NORM[difficulty]
    beyond = max(0.0, d - mastery)
    inc = sensitivity * (0.10 + 0.12 * beyond + 0.05 * (difficulty - 1) / 2)
    return float(min(1.0, frustration + inc))
