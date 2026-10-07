"""Final evaluation of every strategy on the held-out test students.

The 10 test students are used **only here** - never during training,
hyper-parameter selection or the reward-sensitivity study.

For each strategy we run:

    3 model seeds (RL only) x 10 test students x 3 evaluation seeds
    = 90 runs for RL strategies, 30 runs for baselines

every run consisting of one fixed-horizon session of EPISODE_LEN (20)
recommendations, so all strategies face exactly the same workload.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EPISODE_LEN, EVAL_SEEDS, RESULTS_DIR, TOPICS, TRAIN_SEEDS  # noqa: E402
from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from evaluation.metrics import RunMetrics, compute_run_metrics  # noqa: E402
from evaluation.strategies import (  # noqa: E402
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    build_strategy,
    model_seeds_for,
)

EVAL_DIR = RESULTS_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_session(
    recommender,
    env: CodingPracticeEnv,
    *,
    student,
    history: list[dict],
    env_seed: int,
) -> tuple[list[dict], np.ndarray, np.ndarray, float]:
    """One full fixed-horizon practice session with a frozen strategy."""
    recommender.reset()
    state, info = env.reset(
        seed=env_seed, options={"student_id": student.student_id, "history": history}
    )
    initial = env.mastery.copy()

    log: list[dict] = []
    total_reward = 0.0
    while True:
        action = recommender.recommend(state)
        state, reward, terminated, truncated, sinfo = env.step(action)
        total_reward += reward
        log.append(sinfo)
        if terminated or truncated:
            break
    return log, initial, env.mastery.copy(), total_reward


def evaluate_all(
    strategies: list[str] | None = None,
    model_seeds: tuple[int, ...] = TRAIN_SEEDS,
    eval_seeds: tuple[int, ...] = EVAL_SEEDS,
    episode_len: int = EPISODE_LEN,
    verbose: bool = True,
    out_name: str = "per_run.csv",
) -> pd.DataFrame:
    strategies = strategies or STRATEGY_ORDER
    problems = load_problems()
    test_students = load_students(split="test")
    history = history_lookup(problems)

    env = CodingPracticeEnv(
        problems=problems, students=test_students, mode="eval", episode_len=episode_len
    )

    rows: list[dict] = []
    t0 = time.time()
    for strat in strategies:
        label = STRATEGY_LABELS[strat]
        runs = 0
        for model_seed in model_seeds_for(strat, model_seeds):
            for s_idx, student in enumerate(test_students):
                for eval_seed in eval_seeds:
                    # identical simulator seed for every strategy
                    env_seed = int(eval_seed * 100_003 + s_idx)
                    random_seed = int(eval_seed * 977 + model_seed)
                    rec = build_strategy(strat, model_seed=model_seed, random_seed=random_seed)
                    log, init, final, reward = run_session(
                        rec,
                        env,
                        student=student,
                        history=history.get(student.student_id, []),
                        env_seed=env_seed,
                    )
                    m = compute_run_metrics(
                        strategy=label,
                        student_id=student.student_id,
                        profile=student.profile,
                        model_seed=model_seed,
                        eval_seed=eval_seed,
                        initial_mastery=init,
                        final_mastery=final,
                        episode_log=log,
                        total_reward=reward,
                    )
                    row = m.to_dict()
                    for i, t in enumerate(TOPICS):
                        row[f"init_{t}"] = float(init[i])
                        row[f"final_{t}"] = float(final[i])
                    rows.append(row)
                    runs += 1
        if verbose:
            print(f"  evaluated {label:22s} ({runs} runs, {time.time() - t0:5.1f}s)")

    df = pd.DataFrame(rows)
    out_path = EVAL_DIR / out_name
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"-> {out_path}")
    return df


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate all strategies on test students")
    p.add_argument("--strategies", nargs="+", default=None)
    p.add_argument("--quick", action="store_true", help="1 model seed x 1 eval seed")
    args = p.parse_args()
    model_seeds = TRAIN_SEEDS[:1] if args.quick else TRAIN_SEEDS
    eval_seeds = EVAL_SEEDS[:1] if args.quick else EVAL_SEEDS
    evaluate_all(
        strategies=args.strategies,
        model_seeds=model_seeds,
        eval_seeds=eval_seeds,
    )


if __name__ == "__main__":
    main()
