"""Phase 3 test — the LOCK (results/LOCK.json).

The lock freezes data, models, config and the design code.  From Phase 4
onwards every other phase must only *read* those files; this test is the trip
wire that fails the suite if a locked file is edited, deleted or added to.

Before Phase 3 has run the test is skipped, so the suite stays green
throughout Phases 1-2.
"""

from __future__ import annotations

import json

import pytest

from evaluation.lock import LOCK_PATH, manifest, verify_lock


def test_lock_exists_or_phase3_not_reached():
    if not LOCK_PATH.exists():
        pytest.skip("LOCK.json not written yet (Phase 3 has not run)")
    payload = json.loads(LOCK_PATH.read_text())
    assert payload["n_files"] > 0
    assert set(payload["groups"]) == {"data", "models", "config", "design"}
    assert payload["groups"]["models"] >= 12       # 6 tabular + 6 deep checkpoints
    assert payload["groups"]["data"] == 3          # problems / students / interactions
    assert payload["groups"]["config"] == 1
    assert "no hyper-parameter" in payload["statement"]


def test_locked_files_still_match_the_manifest():
    if not LOCK_PATH.exists():
        pytest.skip("LOCK.json not written yet (Phase 3 has not run)")
    problems = verify_lock()
    assert problems == [], "locked files drifted:\n" + "\n".join(problems)


def test_manifest_is_deterministic():
    a, b = manifest(), manifest()
    assert a == b
    assert all(len(v["md5"]) == 32 for v in a.values())
