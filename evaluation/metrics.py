"""Evaluation metrics for one recommendation run.

All strategies are scored with the *same* metrics over the *same* held-out
students, seeds and episode length.

Metric definitions
------------------
skill_gain        mean over the 12 topics of (final mastery - initial mastery)
skill_gain_total  sum over the 12 topics (same signal, unscaled)
completion_rate   solved problems / recommended problems
avg_hints         hints requested per recommendation (lower is better)
mean_solve_time   average minutes per attempt
median_solve_time median minutes per attempt
calibration       recommendations in the "appropriate" challenge band / total
too_easy_rate     share of recommendations the student had already mastered
too_hard_rate     share of recommendations far beyond current ability
repetition_rate   steps that repeated the previous topic / (steps - 1)
unproductive_repeat_rate
                  steps that repeated the previous topic *while the
                  recommendation was no longer well calibrated* (challenge
                  != "appropriate") -- the spec's "unnecessarily repeated
                  topic".  Repeating a still-weak topic inside the zone of
                  proximal development is counted in repetition_rate but NOT
                  here, because it is productive massed practice.
unique_topics     number of distinct topics recommended in the session
frustration_end   frustration level after the session (lower is better)
quit_not_applicable: sessions always run the full fixed horizon, so every
                     strategy is compared over the same number of problems
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RunMetrics:
    strategy: str
    student_id: str
    profile: str
    model_seed: int
    eval_seed: int
    skill_gain: float
    skill_gain_total: float
    completion_rate: float
    avg_hints: float
    mean_solve_time: float
    median_solve_time: float
    calibration: float
    too_easy_rate: float
    too_hard_rate: float
    repetition_rate: float
    unproductive_repeat_rate: float
    unique_topics: int
    frustration_end: float
    initial_mastery: float
    final_mastery: float
    total_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_run_metrics(
    *,
    strategy: str,
    student_id: str,
    profile: str,
    model_seed: int,
    eval_seed: int,
    initial_mastery: np.ndarray,
    final_mastery: np.ndarray,
    episode_log: list[dict],
    total_reward: float,
) -> RunMetrics:
    n = len(episode_log)
    successes = [bool(e["success"]) for e in episode_log]
    gains = [e["skill_gain"] for e in episode_log]
    hints = [e["hints_used"] for e in episode_log]
    times = [e["solve_time"] for e in episode_log]
    challenges = [e["challenge"] for e in episode_log]
    topics = [e["topic"] for e in episode_log]

    diff = final_mastery - initial_mastery
    repeated = sum(
        1 for i in range(1, n) if episode_log[i]["topic"] == episode_log[i - 1]["topic"]
    )
    unproductive = sum(
        1
        for i in range(1, n)
        if episode_log[i]["topic"] == episode_log[i - 1]["topic"]
        and episode_log[i]["challenge"] != "appropriate"
    )

    return RunMetrics(
        strategy=strategy,
        student_id=student_id,
        profile=profile,
        model_seed=model_seed,
        eval_seed=eval_seed,
        skill_gain=float(diff.mean()),
        skill_gain_total=float(diff.sum()),
        completion_rate=float(np.mean(successes)) if n else 0.0,
        avg_hints=float(np.mean(hints)) if n else 0.0,
        mean_solve_time=float(np.mean(times)) if n else 0.0,
        median_solve_time=float(np.median(times)) if n else 0.0,
        calibration=float(np.mean([c == "appropriate" for c in challenges])) if n else 0.0,
        too_easy_rate=float(np.mean([c == "too_easy" for c in challenges])) if n else 0.0,
        too_hard_rate=float(np.mean([c == "too_hard" for c in challenges])) if n else 0.0,
        repetition_rate=float(repeated / max(n - 1, 1)),
        unproductive_repeat_rate=float(unproductive / max(n - 1, 1)),
        unique_topics=len(set(topics)),
        frustration_end=float(episode_log[-1]["frustration"]) if n else 0.0,
        initial_mastery=float(np.mean(initial_mastery)),
        final_mastery=float(np.mean(final_mastery)),
        total_reward=float(total_reward),
    )


SUMMARY_METRICS = [
    "skill_gain",
    "completion_rate",
    "avg_hints",
    "mean_solve_time",
    "median_solve_time",
    "calibration",
    "too_easy_rate",
    "too_hard_rate",
    "repetition_rate",
    "unproductive_repeat_rate",
    "unique_topics",
    "frustration_end",
    "total_reward",
]

# metrics where higher is better / lower is better (for the report)
HIGHER_IS_BETTER = {
    "skill_gain", "completion_rate", "calibration", "unique_topics", "total_reward",
}
LOWER_IS_BETTER = {
    "avg_hints", "mean_solve_time", "median_solve_time", "too_easy_rate",
    "too_hard_rate", "repetition_rate", "frustration_end",
}
