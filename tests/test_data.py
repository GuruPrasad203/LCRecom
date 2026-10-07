"""Dataset validation tests (Stage 1 gate)."""

import numpy as np
import pandas as pd
import pytest

from config import DATA_DIR, ERROR_TYPES, N_ACTIONS, TOPICS
from data.generate_data import generate_all, load_interactions, load_problems, load_students


@pytest.fixture(scope="module")
def problems():
    return load_problems()


@pytest.fixture(scope="module")
def students():
    return load_students()


@pytest.fixture(scope="module")
def interactions():
    return load_interactions()


def test_problems_shape_and_categories(problems):
    assert len(problems) == 180
    assert problems.problem_id.nunique() == 180
    combos = problems.groupby(["topic", "difficulty"]).size()
    assert len(combos) == 12 * 3  # every action category populated
    assert combos.min() >= 5
    assert set(problems.topic) == set(TOPICS)
    assert sorted(problems.difficulty.unique()) == [1, 2, 3]
    assert problems.expected_time.between(1, 200).all()


def test_prerequisites_are_valid_topics(problems):
    for raw in problems.prerequisite_topics.fillna(""):
        for t in str(raw).split(";"):
            if t:
                assert t in TOPICS


def test_students_split_and_ranges(students):
    assert len(students) == 28
    ids = [s.student_id for s in students]
    assert len(set(ids)) == 28
    train = [s for s in students if s.split == "train"]
    test = [s for s in students if s.split == "test"]
    assert len(train) == 18, "training pool must have 18 students"
    assert len(test) == 10, "held-out test split must have exactly 10 students"
    for s in students:
        assert 0.0 <= s.ability <= 1.0
        assert 0.0 <= s.hint_inclination <= 1.0
        for t in TOPICS:
            assert 0.0 <= s.mastery[t] <= 1.0, f"{s.student_id}/{t} mastery out of range"
        assert set(s.error_bias) <= set(ERROR_TYPES)


def test_interactions_consistent(interactions, problems, students):
    assert len(interactions) == 40 * 28
    assert set(interactions.student_id) <= {s.student_id for s in students}
    assert set(interactions.problem_id) <= set(problems.problem_id)
    assert set(interactions.success) <= {0, 1}
    bad = interactions[interactions.error_type != ""]
    assert set(bad.error_type) <= set(ERROR_TYPES)
    assert (interactions.solve_time > 0).all()
    assert interactions.groupby("student_id").size().eq(40).all()


def test_generation_is_reproducible(tmp_path):
    a = generate_all(out_dir=tmp_path / "a", seed=42)
    b = generate_all(out_dir=tmp_path / "b", seed=42)
    for key in a:
        assert a[key].read_bytes() == b[key].read_bytes(), f"{key} not reproducible"


def test_action_space_alignment(problems):
    from environment.action_space import action_to_topic_difficulty
    from config import TOPICS as T

    seen = set()
    for a in range(N_ACTIONS):
        topic_idx, diff = action_to_topic_difficulty(a)
        seen.add((T[topic_idx], diff))
    assert seen == set(zip(problems.topic, problems.difficulty))
