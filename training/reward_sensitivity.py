"""Phase 2 — reward-sensitivity study (TRAIN split only).

The reward function is the one part of the design that encodes a *choice*
rather than a fact, so this study measures how much the conclusions depend on
it.  Six variants are trained and scored:

===================  =====================================================
variant              change w.r.t. ``RewardConfig()`` (the reference)
===================  =====================================================
default              none — the deployed reward, untouched
gain_half            skill_gain_weight 40 -> 20      (gain x 0.5)
gain_double          skill_gain_weight 40 -> 80      (gain x 2)
time_penalty_off     time_penalty 0.80 -> 0.0
repetition_x2        repetition penalties x2 (0.40 / 0.70 / 0.60)
success_bonus_low    success_bonus 0.80 -> 0.20
===================  =====================================================

Protocol (identical to the main training protocol, only the reward differs):

* agent        : Q-Learning (tabular), default ``TabularConfig()``
* episodes     : ``SENSITIVITY_EPISODES`` = 600 per run
* seeds        : (0, 1)
* selection    : greedy-val checkpoint selection on 4 TRAINING students
                 (``training.trainer.train``), models are not written
* readout      : greedy rollouts on the 6 fixed TRAINING students x 2 eval
                 seeds = 12 cells per (variant, seed)

Every cell is scored twice with *identical* trajectories:

    reward_default  under ``RewardConfig()``      <- the common yardstick
    reward_variant  under the variant's own reward

The reward configuration never influences the simulated student dynamics (only
the number returned by ``compute_reward``), so both numbers describe exactly
the same session.  ``reward_default`` is the primary column: variants are only
comparable when they are scored with the same ruler.

Leakage guard
-------------
:func:`resolve_students` refuses any student id outside the train split, and
``training.trainer.train`` raises if its pool contains no training student, so
the held-out test students cannot enter this study even by accident.

Output::

    results/reward_sensitivity.csv            one row per (variant, seed, cell)
    results/reward_sensitivity_summary.csv    mean +/- sd per variant
    results/training/q_learning_<variant>_seed<s>.{csv,json}   learning curves

Usage::

    python training/reward_sensitivity.py [--episodes 600] [--seeds 0 1]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.q_learning_agent import QLearningAgent  # noqa: E402
from config import (  # noqa: E402
    READOUT_EVAL_SEEDS,
    READOUT_N_STUDENTS,
    RESULTS_DIR,
    SENSITIVITY_EPISODES,
    TabularConfig,
)
from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from environment.reward import RewardConfig  # noqa: E402
from environment.simulator import StudentProfile  # noqa: E402
from evaluation.audit_dqn import env_seed_for  # noqa: E402  (identical geometry)
from training.common import make_env  # noqa: E402
from training.trainer import train  # noqa: E402

SENSITIVITY_SEEDS: tuple[int, ...] = (0, 1)
OUT_CSV = RESULTS_DIR / "reward_sensitivity.csv"

# --------------------------------------------------------------------------
# The six variants.  ``{}`` = the reference reward, byte-for-byte untouched.
# --------------------------------------------------------------------------
VARIANTS: dict[str, dict[str, float]] = {
    "default": {},
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

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "default": "reference reward (unchanged)",
    "gain_half": "skill_gain_weight 40 -> 20",
    "gain_double": "skill_gain_weight 40 -> 80",
    "time_penalty_off": "time_penalty 0.80 -> 0.0",
    "repetition_x2": "repetition penalties x2",
    "success_bonus_low": "success_bonus 0.80 -> 0.20",
}


def variant_config(name: str) -> RewardConfig:
    """RewardConfig for ``name``; 'default' is exactly ``RewardConfig()``."""
    if name not in VARIANTS:
        raise KeyError(f"unknown reward variant {name!r}; known: {sorted(VARIANTS)}")
    return RewardConfig(**VARIANTS[name])


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
def resolve_students(student_ids: Sequence[str]) -> list[StudentProfile]:
    """Resolve ids -> profiles, rejecting anything outside the TRAIN split.

    This is the hard boundary of the sensitivity study: test-split ids raise
    ``ValueError`` before any environment is built.
    """
    students = load_students()
    by_id = {s.student_id: s for s in students}
    train_ids = {s.student_id for s in students if s.split == "train"}

    unknown = [sid for sid in student_ids if sid not in by_id]
    if unknown:
        raise ValueError(f"unknown student ids: {unknown}")
    rejected = [sid for sid in student_ids if sid not in train_ids]
    if rejected:
        raise ValueError(
            f"the reward-sensitivity harness only accepts TRAIN-split students; "
            f"rejected held-out ids: {rejected}"
        )
    return [by_id[sid] for sid in student_ids]


def sensitivity_student_ids(n: int = READOUT_N_STUDENTS) -> list[str]:
    """The fixed TRAINING students used by every readout cell."""
    ids = [s.student_id for s in load_students(split="train")]
    if n > len(ids):
        raise ValueError(f"requested {n} train students, only {len(ids)} exist")
    return ids[:n]


# --------------------------------------------------------------------------
# Readout
# --------------------------------------------------------------------------
def _greedy_rollout(env, recommender, *, student, history, env_seed: int):
    recommender.reset()
    state, _ = env.reset(
        seed=env_seed, options={"student_id": student.student_id, "history": history}
    )
    actions: list[int] = []
    total = 0.0
    log: list[dict] = []
    while True:
        action = recommender.recommend(state)
        state, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        log.append(info)
        total += reward
        if terminated or truncated:
            break
    return float(total), actions, log


def greedy_readout(
    variant: str,
    seed: int,
    agent,
    student_ids: Optional[Sequence[str]] = None,
    variant_cfg: Optional[RewardConfig] = None,
) -> list[dict]:
    """Greedy rollouts on 6 fixed TRAINING students x 2 eval seeds."""
    ids = list(student_ids) if student_ids is not None else sensitivity_student_ids()
    students = resolve_students(ids)                      # <- leakage guard
    cfg = variant_cfg if variant_cfg is not None else variant_config(variant)

    problems = load_problems()
    history = history_lookup(problems)
    pool = load_students()
    env_default = CodingPracticeEnv(problems=problems, students=pool, mode="train")
    env_variant = CodingPracticeEnv(
        problems=problems, students=pool, mode="train", reward_config=cfg
    )

    rows: list[dict] = []
    for s_idx, student in enumerate(students):
        for eval_seed in READOUT_EVAL_SEEDS:
            es = env_seed_for(eval_seed, s_idx)
            h = history.get(student.student_id, [])
            r_default, actions, log = _greedy_rollout(
                env_default, agent, student=student, history=h, env_seed=es
            )
            r_variant, _, _ = _greedy_rollout(
                env_variant, agent, student=student, history=h, env_seed=es
            )
            rows.append(
                {
                    "variant": variant,
                    "variant_description": VARIANT_DESCRIPTIONS[variant],
                    "seed": seed,
                    "student_id": student.student_id,
                    "eval_seed": eval_seed,
                    "env_seed": es,
                    "reward_default": r_default,
                    "reward_variant": r_variant,
                    "skill_gain_total": float(sum(i["skill_gain"] for i in log)),
                    "completion_rate": float(np.mean([i["success"] for i in log])),
                    "unique_actions": len(set(actions)),
                    "n_steps": len(actions),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Study driver
# --------------------------------------------------------------------------
def run_study(
    variants: Optional[Sequence[str]] = None,
    episodes: int = SENSITIVITY_EPISODES,
    seeds: Sequence[int] = SENSITIVITY_SEEDS,
    verbose: bool = True,
    out_csv: Path = OUT_CSV,
    save_training: bool = True,
) -> pd.DataFrame:
    variants = list(variants) if variants else list(VARIANTS)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise KeyError(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")

    rows: list[dict] = []
    for variant in variants:
        cfg = variant_config(variant)
        for seed in seeds:
            env = make_env()
            env.reward_config = cfg
            agent = QLearningAgent(config=TabularConfig(), seed=seed)
            if verbose:
                print(f"  [sensitivity] {variant:<18} seed={seed} "
                      f"({episodes} episodes)")
            train(
                agent,
                env,
                episodes=episodes,
                seed=seed,
                name=f"q_learning_{variant}",
                verbose=False,
                save=save_training,
                save_model=False,       # throw-away models: curve only
            )
            rows.extend(greedy_readout(variant, seed, agent, variant_cfg=cfg))

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["variant", "variant_description"])
        .agg(
            n_cells=("reward_default", "size"),
            reward_default_mean=("reward_default", "mean"),
            reward_default_sd=("reward_default", "std"),
            reward_variant_mean=("reward_variant", "mean"),
            reward_variant_sd=("reward_variant", "std"),
            skill_gain_mean=("skill_gain_total", "mean"),
            completion_mean=("completion_rate", "mean"),
            unique_actions_mean=("unique_actions", "mean"),
        )
        .reset_index()
        .sort_values("reward_default_mean", ascending=False)
    )
    out_summary = out_csv.with_name("reward_sensitivity_summary.csv")
    summary.to_csv(out_summary, index=False)

    # paired comparison of every variant against the reference reward, on the
    # matched (seed, student, eval_seed) cells -- the evidence used to decide
    # whether the default reward should change.
    paired = paired_vs_default(df)
    if paired is not None:
        paired.to_csv(out_csv.with_name("reward_sensitivity_paired.csv"), index=False)

    if verbose:
        print(f"\n-> {out_csv}")
        print(f"-> {out_summary}\n")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
        if paired is not None:
            print("\n  paired vs default (same cells):")
            print(paired.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    return df


def paired_vs_default(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Paired t + Wilcoxon of each variant against 'default' (24 matched cells)."""
    if "default" not in set(df["variant"]):
        return None
    from scipy import stats

    piv = df.pivot_table(
        index=["seed", "student_id", "eval_seed"],
        columns="variant",
        values="reward_default",
    )
    ref = piv["default"]
    rows = []
    for variant in [c for c in piv.columns if c != "default"]:
        diff = piv[variant] - ref
        t, p_t = stats.ttest_rel(piv[variant], ref)
        try:
            _, p_w = stats.wilcoxon(piv[variant], ref)
        except ValueError:          # pragma: no cover - all differences zero
            p_w = float("nan")
        rows.append(
            {
                "variant": variant,
                "n_cells": int(len(diff)),
                "mean_diff_vs_default": float(diff.mean()),
                "sd_diff": float(diff.std(ddof=1)),
                "t_stat": float(t),
                "p_paired_t": float(p_t),
                "p_wilcoxon": float(p_w),
                "significant_05": bool(min(p_t, p_w if p_w == p_w else 1.0) < 0.05),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Reward sensitivity study (train split)")
    p.add_argument("--episodes", type=int, default=SENSITIVITY_EPISODES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SENSITIVITY_SEEDS))
    p.add_argument("--variants", nargs="+", default=None, choices=sorted(VARIANTS))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_study(
        variants=args.variants,
        episodes=args.episodes,
        seeds=tuple(args.seeds),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
