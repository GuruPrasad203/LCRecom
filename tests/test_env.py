"""Environment contract tests (Stage 2 gate).

Covers: observation/action spaces, determinism, challenge classification,
mastery/frustration dynamics, and the two reward-bound requirements:

* the **raw** reward sum may leave [-2, +2] (documented, allowed),
* the **clipped** reward returned to the agent is always inside [-2, +2].
"""

import numpy as np
import pytest

from config import EPISODE_LEN, N_ACTIONS, N_TOPICS, TOPICS
from data.generate_data import load_problems, load_students, state_history_for, load_interactions
from environment.action_space import (
    action_to_topic_difficulty,
    action_label,
    topic_difficulty_to_action,
)
from environment.coding_env import CodingPracticeEnv
from environment.reward import RewardConfig, classify_challenge, compute_reward
from environment.state import STATE_DIM, TABULAR_N_STATES, describe_state, state_to_index


@pytest.fixture(scope="module")
def env():
    students = load_students(split="train")
    return CodingPracticeEnv(
        problems=load_problems(),
        students=students,
        mode="train",
        render_mode="ansi",
    )


# ----------------------------------------------------------------- spaces
def test_spaces(env):
    assert env.observation_space.shape == (STATE_DIM,)
    assert env.action_space.n == N_ACTIONS
    low, high = env.observation_space.low, env.observation_space.high
    assert low.min() >= 0.0 and high.max() <= 1.0


def test_action_mapping_roundtrip():
    for a in range(N_ACTIONS):
        t, d = action_to_topic_difficulty(a)
        assert topic_difficulty_to_action(t, d) == a
        assert d in (1, 2, 3)
        assert 0 <= t < N_TOPICS
    assert action_label(0) == "Easy Arrays"
    assert action_label(1) == "Medium Arrays"
    assert action_label(2) == "Hard Arrays"
    assert action_label(N_ACTIONS - 1) == "Hard Sorting"


def test_action_out_of_range():
    with pytest.raises(ValueError):
        action_to_topic_difficulty(N_ACTIONS)


# ----------------------------------------------------------------- reset/step
def test_reset_and_step(env):
    obs, info = env.reset(seed=7)
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert info["student"].split == "train"
    assert set(info["mastery"]) == set(TOPICS)

    obs2, reward, terminated, truncated, step_info = env.step(5)
    assert obs2.shape == (STATE_DIM,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert 0 <= step_info["problem_id"]
    assert step_info["topic"] == TOPICS[5 // 3]
    assert step_info["difficulty"] == 5 % 3 + 1
    assert 0.0 <= step_info["mastery_after"] <= 1.0


def test_episode_truncates_at_episode_len(env):
    env.reset(seed=1)
    for t in range(EPISODE_LEN - 1):
        _, _, term, trunc, _ = env.step(0)
        assert not term and not trunc
    _, _, term, trunc, _ = env.step(0)
    assert trunc is True and term is False


def test_determinism_with_same_seed(env):
    def rollout(seed):
        o, _ = env.reset(seed=seed)
        traj = [(o.copy())]
        rng = np.random.default_rng(seed + 1)
        for _ in range(EPISODE_LEN):
            a = int(rng.integers(N_ACTIONS))
            o, r, _, _, info = env.step(a)
            traj.append((o.copy(), r, info["success"], round(info["skill_gain"], 8)))
        return traj

    assert len(rollout(123)) == len(rollout(123))
    a, b = rollout(123), rollout(123)
    for x, y in zip(a, b):
        if isinstance(x, tuple):
            assert np.allclose(x[0], y[0])
            assert x[1:] == y[1:]
        else:
            assert np.allclose(x, y)
    c = rollout(999)
    assert not np.allclose(a[3][0], c[3][0]), "different seeds must diverge"


def test_mastery_updates_only_recommended_topic(env):
    _, info = env.reset(seed=3)
    before = dict(info["mastery"])
    _, _, _, _, s = env.step(topic_difficulty_to_action(TOPICS.index("Trees"), 2))
    obs2 = env._build_state()
    for i, t in enumerate(TOPICS):
        now = float(obs2[i])
        if t == "Trees":
            assert now >= before[t]
        else:
            assert abs(now - before[t]) < 1e-6
    assert s["skill_gain"] >= 0.0


def test_state_index_and_description(env):
    obs, _ = env.reset(seed=4)
    idx = state_to_index(obs)
    assert 0 <= idx < TABULAR_N_STATES
    d = describe_state(obs)
    assert len(d["topic_mastery"]) == 12
    assert d["current_difficulty"] in (1, 2, 3)
    err = d["recent_error_types"]
    if err:
        assert abs(sum(err.values()) - 1.0) < 1e-5


def test_error_distribution_sums_to_at_most_one(env):
    env.reset(seed=5)
    for a in range(0, N_ACTIONS, 2):
        obs, _, _, _, _ = env.step(a)
        block = obs[20:26]
        assert block.sum() == pytest.approx(0.0) or block.sum() == pytest.approx(1.0)


def test_render_returns_text(env):
    env.reset(seed=6)
    env.step(0)
    text = env.render()
    assert isinstance(text, str) and "mastery" in text


def test_eval_mode_fixes_student():
    students = load_students(split="test")
    e = CodingPracticeEnv(problems=load_problems(), students=students, mode="eval")
    _, i1 = e.reset(seed=1)
    _, i2 = e.reset(seed=2)
    assert i1["student_id"] == i2["student_id"]
    assert i1["student"].split == "test"


def test_history_seeding_changes_state():
    students = load_students(split="test")
    problems = load_problems()
    inter = load_interactions()
    e = CodingPracticeEnv(problems=problems, students=students, mode="eval")
    sid = students[0].student_id
    hist = state_history_for(inter, sid, problems)
    o1, _ = e.reset(seed=1, options={"student_id": sid, "history": hist})
    o2, _ = e.reset(seed=1, options={"student_id": sid, "history": []})
    # sliding-window features differ when history differs
    assert not np.allclose(o1[12:16], o2[12:16])


# ----------------------------------------------------------------- reward
def test_challenge_classification_boundaries():
    # tolerance = 0.35; normalised difficulty: easy 0.10, medium 0.50, hard 0.90
    assert classify_challenge(mastery=0.90, difficulty=1) == "too_easy"
    assert classify_challenge(mastery=0.10, difficulty=3) == "too_hard"
    assert classify_challenge(mastery=0.50, difficulty=2) == "appropriate"
    assert classify_challenge(0.46, 1) == "too_easy"      # gap = 0.36 > 0.35
    assert classify_challenge(0.44, 1) == "appropriate"   # gap = 0.34 <= 0.35
    assert classify_challenge(0.54, 3) == "too_hard"      # gap = -0.36 < -0.35
    assert classify_challenge(0.56, 3) == "appropriate"   # gap = -0.34


def test_clipped_reward_always_in_bounds(env):
    """Final reward returned by step() must live in [-2, 2]..."""
    cfg = RewardConfig()
    env.reward_config = cfg
    env.reset(seed=11)
    for a in range(N_ACTIONS):
        _, r, _, _, _ = env.step(a)
        assert cfg.clip_min <= r <= cfg.clip_max


def test_raw_reward_may_leave_bounds_and_is_clipped():
    """...while the raw (unclipped) sum is allowed to be outside."""
    cfg = RewardConfig()
    raw, comps = compute_reward(
        cfg,
        skill_gain=0.20,          # 40 * 0.20 = 8.0
        success=True,
        challenge="appropriate",
        hints=0,
        solve_time=1.0,
        expected_time=10.0,
        frustration=0.0,
        repeated_topic=False,
        repeated_problem=False,
        same_topic_streak=1,
    )
    assert comps["raw_total"] > cfg.clip_max, "raw reward should exceed +2 here"
    assert raw == cfg.clip_max, "clipped reward must saturate at +2"

    raw_neg, comps_neg = compute_reward(
        cfg,
        skill_gain=0.0,
        success=False,
        challenge="too_hard",
        hints=3,
        solve_time=100.0,
        expected_time=10.0,
        frustration=1.0,
        repeated_topic=True,
        repeated_problem=True,
        same_topic_streak=5,
    )
    assert comps_neg["raw_total"] < cfg.clip_min
    assert raw_neg == cfg.clip_min


def test_reward_components_sum_to_raw():
    cfg = RewardConfig()
    _, c = compute_reward(
        cfg, skill_gain=0.02, success=True, challenge="too_easy", hints=2,
        solve_time=20.0, expected_time=10.0, frustration=0.4,
        repeated_topic=True, repeated_problem=False, same_topic_streak=3,
    )
    keys = [k for k in c if k not in ("raw_total", "clipped")]
    assert sum(c[k] for k in keys) == pytest.approx(c["raw_total"])
    assert c["clipped"] == pytest.approx(
        min(cfg.clip_max, max(cfg.clip_min, c["raw_total"]))
    )
