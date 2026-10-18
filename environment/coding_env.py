"""Gymnasium environment: Adaptive Coding-Practice Recommendation.

One *step* = recommend one coding problem -> the simulated student attempts it
-> the student's skill state is updated -> a reward is computed.

The environment is deliberately independent of any agent / UI: it can be
trained and evaluated from scripts, tests, notebooks or a web service.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import (
    DIFFICULTY_LABELS,
    EPISODE_LEN,
    ERROR_TYPES,
    HISTORY_WINDOW,
    MAX_HINTS,
    N_ACTIONS,
    N_ERROR_WINDOW,
    N_TOPICS,
    TOPICS,
)
from environment.action_space import ACTION_LABELS, action_to_topic_difficulty
from environment.reward import RewardConfig, classify_challenge, compute_reward
from environment.simulator import (
    StudentProfile,
    build_profile_from_metrics,
    compute_skill_gain,
    simulate_attempt,
    update_frustration,
)
from leetcode_integration import build_student_from_leetcode
from environment.state import (
    ERROR_SLICE,
    IDX_AVG_TIME,
    IDX_CONSEC_FAIL,
    IDX_DIFFICULTY,
    IDX_ERROR_RATE,
    IDX_FRUSTRATION,
    IDX_HINT_USAGE,
    IDX_STREAK,
    IDX_RECENT_ACCURACY,
    MASTERY_SLICE,
    STATE_DIM,
    empty_state,
)


class CodingPracticeEnv(gym.Env):
    """Adaptive coding-practice recommendation environment.

    Parameters
    ----------
    problems : DataFrame
        Columns: problem_id, title, topic, difficulty,
        prerequisite_topics, expected_time.
    students : sequence of StudentProfile
        Candidate learners.  In ``mode="train"`` a new student is sampled at
        every ``reset()``; in ``mode="eval"`` one fixed student is used.
    reward_config : RewardConfig, optional
        Reward weights (documented in ``environment/reward.py``).
    episode_len : int
        Number of recommendations per episode (truncation point).
    mode : {"train", "eval"}
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        problems: pd.DataFrame,
        students: Sequence[StudentProfile],
        reward_config: Optional[RewardConfig] = None,
        episode_len: int = EPISODE_LEN,
        mode: str = "train",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if not len(students):
            raise ValueError("at least one student profile is required")

        self.render_mode = render_mode
        self.mode = mode
        self.episode_len = int(episode_len)
        self.reward_config = reward_config or RewardConfig()
        self.students = list(students)
        self.fixed_student: Optional[StudentProfile] = self.students[0] if mode == "eval" else None

        # ---- problem pools ---------------------------------------------
        self.problems = problems.reset_index(drop=True)
        self._pool: dict[tuple[int, int], list[int]] = {}
        for idx, row in self.problems.iterrows():
            key = (TOPICS.index(row["topic"]), int(row["difficulty"]))
            self._pool.setdefault(key, []).append(idx)
        missing = [
            f"{TOPICS[t]}/{d}"
            for t in range(N_TOPICS)
            for d in (1, 2, 3)
            if not self._pool.get((t, d))
        ]
        if missing:
            raise ValueError(f"problem pool missing categories: {missing}")

        # ---- spaces ------------------------------------------------------
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # ---- dynamic student state --------------------------------------
        self.student: Optional[StudentProfile] = None
        self.mastery = np.zeros(N_TOPICS, dtype=np.float64)
        self.frustration = 0.0
        self.consec_failures = 0
        self.current_difficulty = 1
        self.recent: deque[dict[str, Any]] = deque(maxlen=HISTORY_WINDOW)
        self.recent_errors: deque[str] = deque(maxlen=N_ERROR_WINDOW)
        self.steps = 0
        self.attempted: list[int] = []
        self.last_topic_idx: Optional[int] = None
        self.last_problem_id: Optional[int] = None
        self.same_topic_streak = 0
        self.history: list[dict[str, Any]] = []
        self._episode_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _default_history(self) -> list[dict[str, Any]]:
        """Plausible neutral history when nothing is supplied."""
        return [
            {
                "success": i % 2 == 0,
                "time_ratio": 1.0,
                "hints": 1 if i % 3 else 0,
                "errors": 1,
                "error_types": [ERROR_TYPES[i % len(ERROR_TYPES)]],
            }
            for i in range(HISTORY_WINDOW)
        ]

    def _seed_from_history(self, history: Iterable[dict[str, Any]]) -> None:
        self.recent.clear()
        self.recent_errors.clear()
        for h in history:
            self.recent.append(
                {
                    "success": bool(h["success"]),
                    "time_ratio": float(h["time_ratio"]),
                    "hints": int(h["hints"]),
                    "errors": int(h.get("errors", 1 if not h["success"] else 0)),
                }
            )
            for et in h.get("error_types", []):
                if et in ERROR_TYPES:
                    self.recent_errors.append(et)

    def _select_problem(self, topic_idx: int, difficulty: int) -> dict[str, Any]:
        pool = self._pool[(topic_idx, difficulty)]
        unattempted = [i for i in pool if int(self.problems.at[i, "problem_id"]) not in self.attempted]
        idx = int(self.np_random.choice(unattempted if unattempted else pool))
        return self.problems.iloc[idx]

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}

        leet_username = options.get("leetcode_username")
        if "student" in options:
            # Pre-built StudentProfile (e.g. imported from LeetCode) — avoids
            # refetching the profile just to seed the environment.
            self.student = options["student"]
        elif leet_username:
            from leetcode_integration import build_student_from_leetcode
            prof, _mapped = build_student_from_leetcode(leet_username, split=self.mode)
            self.student = prof
        elif "student_id" in options:
            self.student = next(
                s for s in self.students if s.student_id == options["student_id"]
            )
        elif self.mode == "eval":
            self.student = self.fixed_student
        else:
            self.student = self.students[int(self.np_random.integers(len(self.students)))]

        if "mastery" in options:
            mastery_dict = dict(options["mastery"])
        else:
            mastery_dict = dict(self.student.mastery)
        self.mastery = np.array([mastery_dict.get(t, 0.0) for t in TOPICS], dtype=np.float64)

        self.frustration = float(options.get("frustration", 0.0))
        self.consec_failures = int(options.get("consecutive_failures", 0))
        self.current_difficulty = int(options.get("current_difficulty", 1))
        history = options.get("history")
        self._seed_from_history(history if history is not None else self._default_history())

        self.steps = 0
        self.attempted = []
        self.last_topic_idx = None
        self.last_problem_id = None
        self.same_topic_streak = 0
        self.history = []
        self._episode_log = []

        obs = self._build_state()
        info = {
            "student_id": self.student.student_id,
            "profile": self.student.profile,
            "student": self.student,
            "mastery": {t: float(self.mastery[i]) for i, t in enumerate(TOPICS)},
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.student is None:
            raise RuntimeError("call reset() before step()")

        action = int(action)
        topic_idx, difficulty = action_to_topic_difficulty(action)
        topic = TOPICS[topic_idx]
        problem = self._select_problem(topic_idx, difficulty)
        expected_time = float(problem["expected_time"])

        mastery_before = float(self.mastery[topic_idx])
        challenge = classify_challenge(mastery_before, difficulty)

        attempt = simulate_attempt(
            profile=self.student,
            topic=topic,
            mastery=mastery_before,
            difficulty=difficulty,
            expected_time=expected_time,
            frustration=self.frustration,
            consecutive_failures=self.consec_failures,
            rng=self.np_random,
        )

        skill_gain = compute_skill_gain(
            success=attempt.success,
            mastery=mastery_before,
            difficulty=difficulty,
            ability=self.student.ability,
            learning_rate=self.student.learning_rate,
            hints=attempt.hints_used,
            frustration=self.frustration,
            challenge=challenge,
        )
        self.mastery[topic_idx] = min(1.0, mastery_before + skill_gain)

        self.frustration = update_frustration(
            self.frustration,
            attempt.success,
            difficulty,
            mastery_before,
            self.student.frustration_sensitivity,
        )
        self.consec_failures = 0 if attempt.success else self.consec_failures + 1
        self.current_difficulty = difficulty

        # ---- sliding windows -------------------------------------------
        time_ratio = attempt.solve_time / max(expected_time, 1e-6)
        self.recent.append(
            {
                "success": attempt.success,
                "time_ratio": time_ratio,
                "hints": attempt.hints_used,
                "errors": attempt.wrong_submissions,
            }
        )
        if attempt.error_type is not None:
            for _ in range(attempt.wrong_submissions):
                self.recent_errors.append(attempt.error_type)

        # ---- repetition bookkeeping ------------------------------------
        repeated_topic = self.last_topic_idx == topic_idx
        repeated_problem = self.last_problem_id == int(problem["problem_id"])
        self.same_topic_streak = self.same_topic_streak + 1 if repeated_topic else 1
        self.last_topic_idx = topic_idx
        self.last_problem_id = int(problem["problem_id"])
        self.attempted.append(int(problem["problem_id"]))
        self.steps += 1

        reward, reward_components = compute_reward(
            self.reward_config,
            skill_gain=skill_gain,
            success=attempt.success,
            challenge=challenge,
            hints=attempt.hints_used,
            solve_time=attempt.solve_time,
            expected_time=expected_time,
            frustration=self.frustration,
            repeated_topic=repeated_topic and self.steps > 1,
            repeated_problem=repeated_problem and self.steps > 1,
            same_topic_streak=self.same_topic_streak,
        )

        info: dict[str, Any] = {
            "action_label": ACTION_LABELS[action],
            "topic": topic,
            "topic_idx": topic_idx,
            "difficulty": difficulty,
            "difficulty_label": DIFFICULTY_LABELS[difficulty],
            "problem_id": int(problem["problem_id"]),
            "problem_title": problem["title"],
            "expected_time": expected_time,
            "success": attempt.success,
            "solve_time": attempt.solve_time,
            "hints_used": attempt.hints_used,
            "error_type": attempt.error_type,
            "wrong_submissions": attempt.wrong_submissions,
            "p_base": attempt.p_base,
            "skill_gain": skill_gain,
            "challenge": challenge,
            "mastery_before": mastery_before,
            "mastery_after": float(self.mastery[topic_idx]),
            "frustration": self.frustration,
            "consecutive_failures": self.consec_failures,
            "repeated_topic": repeated_topic and self.steps > 1,
            "repeated_problem": repeated_problem and self.steps > 1,
            "reward_components": reward_components,
        }
        self.history.append(info)
        self._episode_log.append(info)

        terminated = False  # a session always lasts EPISODE_LEN problems
        truncated = self.steps >= self.episode_len
        obs = self._build_state()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def _build_state(self) -> np.ndarray:
        s = empty_state()
        s[MASTERY_SLICE] = self.mastery

        if self.recent:
            succ = [h["success"] for h in self.recent]
            s[IDX_RECENT_ACCURACY] = float(np.mean(succ))
            s[IDX_AVG_TIME] = float(np.clip(np.mean([h["time_ratio"] for h in self.recent]) / 2.0, 0.0, 1.0))
            s[IDX_HINT_USAGE] = float(np.clip(np.mean([h["hints"] for h in self.recent]) / MAX_HINTS, 0.0, 1.0))
            s[IDX_ERROR_RATE] = float(np.clip(np.mean([h["errors"] for h in self.recent]), 0.0, 1.0))
        else:
            s[IDX_RECENT_ACCURACY] = 0.5
            s[IDX_AVG_TIME] = 0.5
            s[IDX_HINT_USAGE] = 0.15
            s[IDX_ERROR_RATE] = 0.2

        s[IDX_DIFFICULTY] = (self.current_difficulty - 1) / 2.0
        s[IDX_CONSEC_FAIL] = min(self.consec_failures, 5) / 5.0
        s[IDX_FRUSTRATION] = self.frustration
        s[IDX_STREAK] = min(self.same_topic_streak, 5) / 5.0

        if self.recent_errors:
            counts = np.zeros(len(ERROR_TYPES), dtype=np.float64)
            for e in self.recent_errors:
                counts[ERROR_TYPES.index(e)] += 1
            s[ERROR_SLICE] = counts / counts.sum()
        return np.clip(s, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def render(self) -> Optional[str]:
        if self.student is None:
            return None
        lines = [
            f"step {self.steps}/{self.episode_len} | student={self.student.name} "
            f"({self.student.profile})",
            "  mastery: "
            + ", ".join(
                f"{TOPICS[i][:4]}={self.mastery[i]:.2f}" for i in range(N_TOPICS)
            ),
            f"  frustration={self.frustration:.2f} consecutive_failures={self.consec_failures} "
            f"recent_accuracy={np.mean([h['success'] for h in self.recent]) if self.recent else 0:.2f}",
        ]
        if self.history:
            last = self.history[-1]
            lines.append(
                f"  last: {last['problem_title']} [{last['action_label']}] -> "
                f"{'PASS' if last['success'] else 'FAIL'} in {last['solve_time']:.1f} min, "
                f"hints={last['hints_used']}, error={last['error_type']}"
            )
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:  # pragma: no cover - no external resources
        return None
