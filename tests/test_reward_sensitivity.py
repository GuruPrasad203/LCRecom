"""Phase 2 tests — reward-sensitivity harness.

Covers:

* the six variants are exactly the documented one-line changes and the
  ``default`` variant is *byte-for-byte* the reference ``RewardConfig()``;
* the leakage guard: any held-out (test-split) student id is rejected before
  an environment is even built, both in the harness and in the readout;
* ``training.trainer.train`` still refuses a pool without train students;
* the study writes a well-formed ``reward_sensitivity.csv`` whose cells are
  6 TRAINING students x 2 eval seeds per (variant, seed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.q_learning_agent import QLearningAgent
from config import READOUT_EVAL_SEEDS, READOUT_N_STUDENTS, SENSITIVITY_EPISODES, TabularConfig
from data.generate_data import load_students
from environment.coding_env import CodingPracticeEnv
from environment.reward import RewardConfig
from data.generate_data import load_problems
from training import reward_sensitivity as rs


# --------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------- #
def test_default_variant_is_the_untouched_reference():
    assert rs.VARIANTS["default"] == {}
    assert rs.variant_config("default") == RewardConfig()
    assert rs.variant_config("default").to_dict() == RewardConfig().to_dict()


def test_variants_are_exactly_the_documented_changes():
    expected = {
        "gain_half": {"skill_gain_weight": 20.0},
        "gain_double": {"skill_gain_weight": 80.0},
        "time_penalty_off": {"time_penalty": 0.0},
        "repetition_x2": {
            "repetition_topic_penalty": 0.40,
            "repetition_problem_penalty": 0.70,
            "repetition_streak_penalty": 0.60,
        },
        "success_bonus_low": {"success_bonus": 0.20},
    }
    assert set(rs.VARIANTS) == {"default", *expected}
    base = RewardConfig().to_dict()
    for name, overrides in expected.items():
        cfg = rs.variant_config(name).to_dict()
        changed = {k for k in base if cfg[k] != base[k]}
        assert changed == set(overrides), f"{name}: unexpected fields changed"
        for k, v in overrides.items():
            assert cfg[k] == v
    # reward magnitudes stay inside the clipping window contract
    for name in rs.VARIANTS:
        cfg = rs.variant_config(name)
        assert cfg.clip_min == -2.0 and cfg.clip_max == 2.0


def test_unknown_variant_is_rejected():
    with pytest.raises(KeyError):
        rs.variant_config("does_not_exist")


# --------------------------------------------------------------------- #
# leakage guards
# --------------------------------------------------------------------- #
def test_readout_students_are_the_first_6_train_students():
    ids = rs.sensitivity_student_ids()
    train_ids = [s.student_id for s in load_students(split="train")]
    test_ids = {s.student_id for s in load_students(split="test")}
    assert ids == train_ids[:READOUT_N_STUDENTS]
    assert not set(ids) & test_ids
    assert len(ids) == READOUT_N_STUDENTS == 6
    assert len(READOUT_EVAL_SEEDS) == 2


def test_harness_rejects_test_split_ids():
    test_ids = [s.student_id for s in load_students(split="test")]
    assert test_ids, "expected held-out students in the dataset"
    with pytest.raises(ValueError, match="TRAIN-split"):
        rs.resolve_students([test_ids[0]])
    with pytest.raises(ValueError, match="held-out"):
        rs.resolve_students(["S001", test_ids[0], test_ids[3]])


def test_harness_rejects_unknown_ids():
    with pytest.raises(ValueError, match="unknown student ids"):
        rs.resolve_students(["S999"])


def test_readout_rejects_test_split_ids():
    """The greedy readout itself must refuse held-out students."""
    agent = QLearningAgent(config=TabularConfig(), seed=0)
    test_id = load_students(split="test")[0].student_id
    with pytest.raises(ValueError, match="TRAIN-split"):
        rs.greedy_readout("default", 0, agent, student_ids=[test_id])


def test_trainer_rejects_a_pool_without_train_students():
    """The pre-existing guard in training.trainer.train stays in force."""
    env = CodingPracticeEnv(
        problems=load_problems(), students=load_students(split="test"), mode="train"
    )
    agent = QLearningAgent(config=TabularConfig(), seed=0)
    from training.trainer import train

    with pytest.raises(ValueError, match="train"):
        train(agent, env, episodes=1, seed=0, name="should_not_run",
              save=False, verbose=False)


def test_readout_geometry_matches_the_shared_config():
    assert rs.READOUT_N_STUDENTS == READOUT_N_STUDENTS
    assert rs.READOUT_EVAL_SEEDS == READOUT_EVAL_SEEDS
    assert SENSITIVITY_EPISODES == 600
    assert rs.SENSITIVITY_SEEDS == (0, 1)


# --------------------------------------------------------------------- #
# end-to-end smoke run (tiny; writes only into tmp_path)
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("sens") / "reward_sensitivity.csv"
    df = rs.run_study(
        variants=("default", "gain_half"),
        episodes=8,
        seeds=(0,),
        verbose=False,
        out_csv=out,
        save_training=False,
    )
    return df, out


def test_smoke_run_shape_and_columns(smoke):
    df, out = smoke
    assert out.exists()
    assert out.with_name("reward_sensitivity_summary.csv").exists()
    # 2 variants x 1 seed x 6 train students x 2 eval seeds
    assert len(df) == 2 * 1 * READOUT_N_STUDENTS * len(READOUT_EVAL_SEEDS)
    assert set(df["variant"]) == {"default", "gain_half"}
    for col in ("reward_default", "reward_variant", "skill_gain_total",
                "completion_rate", "unique_actions", "env_seed"):
        assert col in df.columns


def test_smoke_run_scores_the_default_variant_identically_under_both_rulers(smoke):
    df, _ = smoke
    d = df[df.variant == "default"]
    assert np.allclose(d["reward_default"], d["reward_variant"]), (
        "the default variant IS the reference reward, so both rulers must agree"
    )
    g = df[df.variant == "gain_half"]
    assert not np.allclose(g["reward_default"], g["reward_variant"]), (
        "a changed reward must score the same session differently"
    )


def test_smoke_run_never_touched_the_test_split(smoke):
    df, _ = smoke
    train_ids = {s.student_id for s in load_students(split="train")}
    test_ids = {s.student_id for s in load_students(split="test")}
    assert set(df["student_id"]) <= train_ids
    assert not set(df["student_id"]) & test_ids


def test_protocol_constants_are_the_approved_ones():
    assert SENSITIVITY_EPISODES == 600
    assert rs.SENSITIVITY_SEEDS == (0, 1)
    assert list(rs.VARIANTS) == [
        "default", "gain_half", "gain_double",
        "time_penalty_off", "repetition_x2", "success_bonus_low",
    ]
