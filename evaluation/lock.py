"""Phase 3 — LOCK the experiment before any test-split contact.

From the moment ``results/LOCK.json`` exists, **no design change is allowed**:
the data set, the trained checkpoints, the configuration (reward weights,
hyper-parameters, episode geometry, seeds) and the code that *defines the
design* (environment, agents, training loop) are frozen.  Everything that
happens afterwards — final evaluation, comparison tables, figures, demo,
report — only *reads* those artifacts.

Locked groups
-------------
data        data/*.csv                        problems / students / interactions
models      models/*.{npz,pt}                 every trained checkpoint
config      config.py                         reward geometry, hyper-params, seeds
design      environment/*.py, agents/*.py,
            training/*.py                     the reward function, the algorithms
                                              and the training loop themselves

Deliberately *not* locked (they produce the read-out, they do not define the
design): evaluation/, visualization/, tests/, demo.py, README, REPORT.

Usage::

    python evaluation/lock.py            # write results/LOCK.json
    python evaluation/lock.py verify     # re-hash and compare (exit 1 on drift)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR  # noqa: E402

LOCK_PATH = RESULTS_DIR / "LOCK.json"

GROUPS: dict[str, list[Path]] = {
    "data": sorted((ROOT / "data").glob("*.csv")),
    "models": sorted(p for p in (ROOT / "models").iterdir() if p.is_file()),
    "config": [ROOT / "config.py"],
    "design": sorted(
        list((ROOT / "environment").glob("*.py"))
        + list((ROOT / "agents").glob("*.py"))
        + list((ROOT / "training").glob("*.py"))
    ),
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    files: dict[str, dict] = {}
    for group, paths in GROUPS.items():
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"cannot lock missing file: {p}")
            files[str(p.relative_to(ROOT))] = {
                "group": group,
                "md5": md5(p),
                "bytes": p.stat().st_size,
            }
    return files


def write_lock(path: Path = LOCK_PATH) -> dict:
    files = manifest()
    payload = {
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": (
            "Data, models, configuration and design code are frozen. "
            "The held-out test split may now be read once, by "
            "evaluation/evaluate.py; no hyper-parameter, reward, geometry or "
            "selection-rule change is permitted afterwards."
        ),
        "groups": {g: sum(1 for f in files.values() if f["group"] == g)
                   for g in GROUPS},
        "n_files": len(files),
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def verify_lock(path: Path = LOCK_PATH) -> list[str]:
    """Return a list of drift descriptions (empty == lock still holds)."""
    if not path.exists():
        return [f"{path} does not exist - run `python evaluation/lock.py` first"]
    locked = json.loads(path.read_text())["files"]
    problems: list[str] = []

    current = manifest()
    for rel, meta in locked.items():
        p = ROOT / rel
        if not p.exists():
            problems.append(f"deleted since lock: {rel}")
        elif md5(p) != meta["md5"]:
            problems.append(f"modified since lock: {rel}")
    for rel in current:
        if rel not in locked:
            problems.append(f"new file in a locked group: {rel}")
    return problems


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Freeze data/models/config/design")
    p.add_argument("command", nargs="?", default="write", choices=["write", "verify"])
    args = p.parse_args()

    if args.command == "write":
        payload = write_lock()
        print(f"LOCK written -> {LOCK_PATH}")
        print(f"  {payload['n_files']} files: "
              + ", ".join(f"{g}={n}" for g, n in payload["groups"].items()))
        return 0

    problems = verify_lock()
    if problems:
        print("LOCK VIOLATED:")
        for pr in problems:
            print(f"  - {pr}")
        return 1
    print(f"LOCK holds: {LOCK_PATH} matches the working tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
