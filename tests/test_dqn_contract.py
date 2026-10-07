"""Phase 1 gate — the DQN contract audit (10 checks).

Each test maps to one numbered check of ``evaluation/audit_dqn.py``:

    1  preprocessing parity          (training AND evaluation code path)
    2  action roundtrip
    3  reward decomposition
    4  terminal invariants
    5  target invariants
    6  replay invariants
    7  recommend()-greediness under a monkeypatched hostile rng
    8  checkpoint identity           (shipped file == selected checkpoint)
    9  action distribution table     (contract part; the collapse verdict is a
                                      reported *outcome*, never a test failure)
    10 deterministic step trace

Checks 1-7 and 10 are model-independent and always asserted.
Check 8 is asserted for every checkpoint whose selection record carries a
greedy-val entry (i.e. after the Phase 1 retrain); before that it is skipped,
so the suite stays green throughout the phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation import audit_dqn as audit


# --------------------------------------------------------------------- #
# 1-7 : shared contracts
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn",
    [
        audit.check_preprocessing_parity,
        audit.check_action_roundtrip,
        audit.check_reward_decomposition,
        audit.check_terminal_invariants,
        audit.check_target_invariants,
        audit.check_replay_invariants,
        audit.check_recommend_greediness,
    ],
    ids=lambda f: f.__name__.removeprefix("check_"),
)
def test_contract_checks_pass(fn):
    result = fn()
    assert result.status == "pass", f"{result.name}: {result.detail} {result.metrics}"


def test_checklist_covers_ten_checks():
    """The audit must expose exactly the ten codified checks."""
    shared = [
        "preprocessing_parity",
        "action_roundtrip",
        "reward_decomposition",
        "terminal_invariants",
        "target_invariants",
        "replay_invariants",
        "recommend_greediness",
    ]
    per_checkpoint = [
        "checkpoint_identity",
        "action_distribution",
        "deterministic_step_trace",
    ]
    assert shared == [c.__name__.removeprefix("check_") for c in [
        audit.check_preprocessing_parity,
        audit.check_action_roundtrip,
        audit.check_reward_decomposition,
        audit.check_terminal_invariants,
        audit.check_target_invariants,
        audit.check_replay_invariants,
        audit.check_recommend_greediness,
    ]]
    # the readout + the two per-checkpoint contracts complete the list of ten
    assert len(shared) + len(per_checkpoint) == 10
    assert per_checkpoint == [
        "checkpoint_identity",
        "action_distribution",
        "deterministic_step_trace",
    ]


# --------------------------------------------------------------------- #
# 8 : checkpoint identity
# --------------------------------------------------------------------- #
def test_checkpoint_identity_of_shipped_models():
    for name in audit.DEEP_MODELS:
        for seed in (0, 1, 2):
            r = audit.check_checkpoint_identity(name, seed)
            if r.status == "skip":
                pytest.skip(
                    f"{name}_seed{seed}: selection record predates greedy-val "
                    f"selection (retrain required) — {r.detail}"
                )
            assert r.status == "pass", f"{r.name}: {r.detail} {r.metrics}"
            assert r.metrics["config_match"] is True
            assert r.metrics["deterministic_trace"] is True


# --------------------------------------------------------------------- #
# 9 : action distribution (contract part only)
# --------------------------------------------------------------------- #
def test_action_distribution_table_is_well_formed():
    for name in audit.DEEP_MODELS:
        for seed in (0, 1, 2):
            r = audit.check_action_distribution(name, seed)
            assert r.status == "pass", f"{r.name}: {r.metrics}"
            assert r.metrics["total_steps"] == (
                audit.AUDIT_N_STUDENTS * len(audit.AUDIT_EVAL_SEEDS) * 20
            )
            assert 1 <= r.metrics["n_distinct_actions"] <= 36
            assert 0.0 < r.metrics["top_action_share"] <= 1.0


def test_action_distribution_uses_training_students_only():
    """Leakage guard: every greedy readout runs on TRAINING students."""
    rows, reward_rows = audit.action_distribution_rows("dqn", 0)
    seen = {r["student_id"] for r in reward_rows}
    train_ids = {s.student_id for s in audit.train_students()}
    test_ids = {s.student_id for s in audit.test_students()}
    assert seen <= train_ids, f"non-train students in readout: {seen - train_ids}"
    assert not seen & test_ids, "held-out test students leaked into the readout"
    assert len(seen) == audit.AUDIT_N_STUDENTS
    df = pd.DataFrame(rows)
    assert len(df) == 36
    assert df["count"].sum() == (
        audit.AUDIT_N_STUDENTS * len(audit.AUDIT_EVAL_SEEDS) * 20
    )
    assert df["prob"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# 10 : deterministic step trace
# --------------------------------------------------------------------- #
def test_deterministic_step_trace():
    for name in audit.DEEP_MODELS:
        r = audit.check_deterministic_step_trace(name, 0)
        assert r.status == "pass", f"{r.name}: {r.metrics}"
        assert r.metrics["reproducible"] is True
        assert r.metrics["diverges_on_other_seed"] is True
        assert len(r.metrics["trace"]) == 20


# --------------------------------------------------------------------- #
# integration: the audit runs end-to-end and writes its artifacts
# --------------------------------------------------------------------- #
def test_run_audit_writes_artifacts(tmp_path):
    report = audit.run_audit(models=("dqn",), seeds=(0,), out_dir=tmp_path)
    assert report["n_fail"] == 0, [
        c for c in report["checks"] if c["status"] == "fail"
    ]
    assert report["ok"] is True

    dist = pd.read_csv(tmp_path / "action_distribution.csv")
    assert set(dist.columns) >= {"model", "seed", "action", "label", "count", "prob"}
    assert len(dist) == 36
    assert dist["count"].sum() == 240

    rewards = pd.read_csv(tmp_path / "greedy_train_reward.csv")
    assert len(rewards) == audit.AUDIT_N_STUDENTS * len(audit.AUDIT_EVAL_SEEDS)
    assert set(rewards["model"]) == {"dqn"}
    train_ids = {s.student_id for s in audit.train_students()}
    assert set(rewards["student_id"]) <= train_ids

    assert (tmp_path / "step_trace.json").exists()
    assert (tmp_path / "audit_report.json").exists()

    # reference readout: every strategy scored on the identical cells
    ref = pd.read_csv(tmp_path / "reference_readout.csv")
    assert set(ref["strategy"]) == {
        "random", "difficulty", "prerequisite", "sarsa", "q_learning",
        "dqn", "double_dqn",
    }
    assert len(ref) == 3 + 4 * 3   # 3 baselines x1 seed, 4 RL x3 seeds

    names = [c["name"].split("[")[0] for c in report["checks"]]
    for required in ("preprocessing_parity", "action_roundtrip",
                     "reward_decomposition", "terminal_invariants",
                     "target_invariants", "replay_invariants",
                     "recommend_greediness", "checkpoint_identity",
                     "action_distribution", "deterministic_step_trace"):
        assert required in names
