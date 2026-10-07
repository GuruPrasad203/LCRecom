"""Shared helpers for the training entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TRAIN_EPISODES, TRAIN_SEEDS  # noqa: E402
from data.generate_data import load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402


def make_env() -> CodingPracticeEnv:
    """Environment with the full student pool; the trainer only uses train."""
    return CodingPracticeEnv(
        problems=load_problems(),
        students=load_students(),
        mode="train",
    )


def training_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--episodes", type=int, default=TRAIN_EPISODES,
                   help="number of training episodes")
    p.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS),
                   help="training seeds")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p
