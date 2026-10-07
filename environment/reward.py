"""Reward function of the coding-practice environment.

Reward design (all terms and weights are documented here and exercised by the
reward-sensitivity experiment ``training/reward_sensitivity.py``)::

    R =  w_gain   * skill_gain                       (learn something new)
      +  b_success * 1{completed}                    (finish the problem)
      +  c_appropriate * 1{well-calibrated challenge}
      +  c_mismatch    * 1{too easy OR too hard}
      -  w_hint    * hints_used                      (dependence on hints)
      -  w_time    * max(0, time_ratio - tol)        (excessive solve time)
      -  w_frust   * frustration_after               (student is fed up)
      -  w_repeat  * 1{same (topic, difficulty) as previous recommendation}
      -  w_problem * 1{the very same problem twice in a row}
      -  w_streak  * max(0, same_topic_streak - 2)      (massed practice is
                                                         penalised *increasingly*)
      -> clipped to [-2, +2] for training stability

Motivation
----------
* ``skill_gain``  – the primary objective of the project: *maximise skill
  improvement*.  Weight 40 makes a typical gain (~0.013) worth ~0.5 reward.
* ``b_success``   – finishing a problem is intrinsically desirable.
* challenge terms – reward "in the zone" problems (Vygotsky's zone of
  proximal development): too easy and too hard are both penalised, which is
  exactly RQ4 (maintain challenge, avoid frustration).
* hint/time terms – encourage efficient, independent solving.
* frustration    – makes repeated failure costly, steering the agent away
  from problems the student keeps failing.
* repetition     – stops the greedy policy from collapsing onto one category.

Every weight is a field of :class:`RewardConfig`, so the sensitivity study can
vary them without touching the environment code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from config import DIFFICULTY_NORM

# Mastery vs normalised difficulty gap considered "well calibrated".
CHALLENGE_TOLERANCE = 0.35


@dataclass
class RewardConfig:
    skill_gain_weight: float = 40.0
    success_bonus: float = 0.80
    challenge_appropriate: float = 0.50
    challenge_mismatch: float = -0.60
    hint_penalty: float = 0.18
    time_penalty: float = 0.80
    time_tolerance: float = 1.15      # solve_time / expected_time
    frustration_penalty: float = 0.45
    repetition_topic_penalty: float = 0.20
    repetition_problem_penalty: float = 0.35
    repetition_streak_penalty: float = 0.30   # applied per repeat beyond the 2nd
    clip_min: float = -2.0
    clip_max: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


def classify_challenge(mastery: float, difficulty: int) -> str:
    """Is this difficulty right for the student's current mastery?

    ``too_easy``    mastery far above the problem      -> boredom / no gain
    ``too_hard``    mastery far below the problem      -> frustration / no gain
    ``appropriate`` otherwise                          -> productive practice
    """
    gap = mastery - DIFFICULTY_NORM[difficulty]
    if gap > CHALLENGE_TOLERANCE:
        return "too_easy"
    if gap < -CHALLENGE_TOLERANCE:
        return "too_hard"
    return "appropriate"


def compute_reward(
    cfg: RewardConfig,
    *,
    skill_gain: float,
    success: bool,
    challenge: str,
    hints: int,
    solve_time: float,
    expected_time: float,
    frustration: float,
    repeated_topic: bool,
    repeated_problem: bool,
    same_topic_streak: int,
) -> tuple[float, dict[str, float]]:
    """Return ``(reward, components)`` for one recommendation step."""
    time_ratio = solve_time / max(expected_time, 1e-6)

    components: dict[str, float] = {
        "skill_gain": cfg.skill_gain_weight * skill_gain,
        "success": cfg.success_bonus if success else 0.0,
        "challenge": cfg.challenge_appropriate if challenge == "appropriate"
        else cfg.challenge_mismatch,
        "hint": -cfg.hint_penalty * hints,
        "time": -cfg.time_penalty * max(0.0, time_ratio - cfg.time_tolerance),
        "frustration": -cfg.frustration_penalty * frustration,
        "repetition_topic": -cfg.repetition_topic_penalty if repeated_topic else 0.0,
        "repetition_problem": -cfg.repetition_problem_penalty if repeated_problem else 0.0,
        "repetition_streak": -cfg.repetition_streak_penalty
        * max(0, same_topic_streak - 2),
    }
    reward = float(sum(components.values()))
    clipped = float(min(cfg.clip_max, max(cfg.clip_min, reward)))
    components["raw_total"] = reward
    components["clipped"] = clipped
    return clipped, components
