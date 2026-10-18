"""CP-4 — hyper-parameter sensitivity study, one-factor-at-a-time (TRAIN split).

Extends the reward-sensitivity protocol (training/reward_sensitivity.py) to
the *algorithm* hyper-parameters that are design choices rather than facts.
Two agents are covered, exactly as pre-registered:

    Q-Learning (tabular)   alpha, gamma, epsilon_decay_episodes
    DQN         (deep)     hidden size, replay buffer, target-update freq

DQN ``lr`` (1e-3) and ``batch_size`` (64) are held FIXED at the config
defaults and are explicitly NOT varied.

Every grid contains its ``config.py`` default value; the default run is
trained ONCE per (agent, seed) and re-emitted into each factor's curve so the
default point is present on every plot (``is_default=True``).

Protocol (identical to the main training protocol, only the varied
hyper-parameter differs):

* episodes : ``SENSITIVITY_EPISODES`` = 600
* seeds    : (0, 1)                          <- 2 seeds, separate from the
                                                5-seed main experiment
* selection: greedy-val checkpoint selection on 4 TRAINING students
* readout  : greedy rollouts on 6 fixed TRAINING students x 2 eval seeds
             = 12 matched cells per (variant, seed); reward is always scored
             under the default ``RewardConfig`` (common yardstick)
* leakage  : resolve_students() rejects any test-split id; trainer refuses
             pools without training students

Output::

    results/sensitivity.csv            one row per (agent, factor, value, seed, cell)
    results/sensitivity_summary.csv    mean +/- sd per (agent, factor, value)
    results/sensitivity_paired.csv     paired t + Wilcoxon vs the default value
    results/sensitivity_runtime.json   per-run and total wall-clock seconds

Usage::

    python training/hyperparam_sensitivity.py [--episodes 600] [--seeds 0 1]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dqn_agent import DQNAgent  # noqa: E402
from agents.q_learning_agent import QLearningAgent  # noqa: E402
from config import (  # noqa: E402
    DQNConfig,
    READOUT_EVAL_SEEDS,
    READOUT_N_STUDENTS,
    RESULTS_DIR,
    SENSITIVITY_EPISODES,
    TabularConfig,
)
from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from environment.reward import RewardConfig  # noqa: E402
from evaluation.audit_dqn import env_seed_for  # noqa: E402
from training.common import make_env  # noqa: E402
from training.reward_sensitivity import (  # noqa: E402
    resolve_students,
    sensitivity_student_ids,
)
from training.trainer import train  # noqa: E402

SENSITIVITY_SEEDS: tuple[int, ...] = (0, 1)
OUT_CSV = RESULTS_DIR / "sensitivity.csv"

# --------------------------------------------------------------------------
# OFAT grids.  ``is_default=True`` marks the config.py default (no override).
# --------------------------------------------------------------------------
QLEARN_DEFAULTS = TabularConfig()
DQN_DEFAULTS = DQNConfig()

VARIANTS: list[dict] = [
    # --- Q-Learning: alpha ------------------------------------------------
    {"agent": "q_learning", "factor": "alpha", "value": 0.05, "fields": {"alpha": 0.05}},
    {"agent": "q_learning", "factor": "alpha", "value": 0.10, "fields": {"alpha": 0.10}},
    {"agent": "q_learning", "factor": "alpha", "value": 0.18, "fields": {}},
    {"agent": "q_learning", "factor": "alpha", "value": 0.30, "fields": {"alpha": 0.30}},
    {"agent": "q_learning", "factor": "alpha", "value": 0.50, "fields": {"alpha": 0.50}},
    # --- Q-Learning: gamma -----------------------------------------------
    {"agent": "q_learning", "factor": "gamma", "value": 0.80, "fields": {"gamma": 0.80}},
    {"agent": "q_learning", "factor": "gamma", "value": 0.90, "fields": {"gamma": 0.90}},
    {"agent": "q_learning", "factor": "gamma", "value": 0.95, "fields": {}},
    {"agent": "q_learning", "factor": "gamma", "value": 0.99, "fields": {"gamma": 0.99}},
    # --- Q-Learning: epsilon decay ---------------------------------------
    {"agent": "q_learning", "factor": "epsilon_decay_episodes", "value": 600,
     "fields": {"epsilon_decay_episodes": 600}},
    {"agent": "q_learning", "factor": "epsilon_decay_episodes", "value": 1200,
     "fields": {}},
    {"agent": "q_learning", "factor": "epsilon_decay_episodes", "value": 2400,
     "fields": {"epsilon_decay_episodes": 2400}},
    # --- DQN: hidden width (lr=1e-3, batch=64 held fixed) -----------------
    {"agent": "dqn", "factor": "hidden", "value": 64,
     "fields": {"hidden1": 64, "hidden2": 64}},
    {"agent": "dqn", "factor": "hidden", "value": 128, "fields": {}},
    {"agent": "dqn", "factor": "hidden", "value": 256,
     "fields": {"hidden1": 256, "hidden2": 256}},
    # --- DQN: replay buffer ----------------------------------------------
    {"agent": "dqn", "factor": "buffer_size", "value": 10_000,
     "fields": {"buffer_size": 10_000}},
    {"agent": "dqn", "factor": "buffer_size", "value": 50_000, "fields": {}},
    {"agent": "dqn", "factor": "buffer_size", "value": 100_000,
     "fields": {"buffer_size": 100_000}},
    # --- DQN: target-update frequency ------------------------------------
    {"agent": "dqn", "factor": "target_update_every", "value": 100,
     "fields": {"target_update_every": 100}},
    {"agent": "dqn", "factor": "target_update_every", "value": 250, "fields": {}},
    {"agent": "dqn", "factor": "target_update_every", "value": 500,
     "fields": {"target_update_every": 500}},
]

FIXED_NOTE = {
    "q_learning": "epsilon_start=1.0, epsilon_end=0.05 fixed; only the listed factor varies",
    "dqn": "lr=1e-3, batch_size=64, gamma=0.95, warmup=1000, train_every=4 FIXED",
}


def _is_default(v: dict) -> bool:
    return len(v["fields"]) == 0


def _slug(agent: str, v: dict) -> str:
    """Training-curve file stem; the shared default config gets one name."""
    if _is_default(v):
        return f"{agent}_sens_default"
    val = str(v["value"]).replace(".", "p")
    return f"{agent}_sens_{v['factor']}{val}"


def build_config(agent: str, fields: dict):
    if agent == "q_learning":
        return TabularConfig(**fields)
    if agent == "dqn":
        return DQNConfig(**fields)
    raise KeyError(agent)


def make_agent(agent: str, cfg, seed: int):
    if agent == "q_learning":
        return QLearningAgent(config=cfg, seed=seed)
    return DQNAgent(config=cfg, seed=seed)


# --------------------------------------------------------------------------
# Readout (identical geometry to reward_sensitivity.greedy_readout)
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


def greedy_readout(agent, student_ids: Optional[Sequence[str]] = None) -> list[dict]:
    """Greedy rollouts on 6 fixed TRAINING students x 2 eval seeds."""
    ids = list(student_ids) if student_ids is not None else sensitivity_student_ids()
    students = resolve_students(ids)                      # <- leakage guard

    problems = load_problems()
    history = history_lookup(problems)
    pool = load_students()
    env = CodingPracticeEnv(problems=problems, students=pool, mode="train",
                            reward_config=RewardConfig())

    rows: list[dict] = []
    for s_idx, student in enumerate(students):
        for eval_seed in READOUT_EVAL_SEEDS:
            es = env_seed_for(eval_seed, s_idx)
            h = history.get(student.student_id, [])
            r, actions, log = _greedy_rollout(
                env, agent, student=student, history=h, env_seed=es
            )
            rows.append(
                {
                    "student_id": student.student_id,
                    "eval_seed": eval_seed,
                    "env_seed": es,
                    "reward_default": r,
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
    episodes: int = SENSITIVITY_EPISODES,
    seeds: Sequence[int] = SENSITIVITY_SEEDS,
    verbose: bool = True,
    save_training: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    runtime: dict[str, float] = {}
    t_start = time.time()

    # de-duplicate trainings: the default config is shared by all factor
    # slots of an agent (trained once, read out once, emitted per slot).
    train_keys: dict[tuple, dict] = {}
    for v in VARIANTS:
        key = (v["agent"], tuple(sorted(v["fields"].items())))
        train_keys.setdefault(key, v)

    trained: dict[tuple, tuple[object, float]] = {}
    for (agent, fields_key), v in train_keys.items():
        fields = dict(fields_key)
        cfg = build_config(agent, fields)
        for seed in seeds:
            env = make_env()
            a = make_agent(agent, cfg, seed)
            t0 = time.time()
            if verbose:
                tag = "DEFAULT" if not fields else f"{v['factor']}={v['value']}"
                print(f"  [sens] {agent:<12} {tag:<28} seed={seed} ({episodes} eps)")
            train(
                a,
                env,
                episodes=episodes,
                seed=seed,
                name=_slug(agent, v),
                verbose=False,
                save=save_training,
                save_model=False,       # throw-away models: curve only
            )
            dt = time.time() - t0
            runtime[f"{_slug(agent, v)}_seed{seed}"] = round(dt, 2)
            # readout stored per trained model; default configs are emitted
            # into their factor slots below (avoids double-counting the
            # first slot), non-default configs are emitted right here.
            readout = greedy_readout(a)
            trained[(agent, fields_key, seed)] = (readout, dt)
            if fields:
                rows.extend(
                    {
                        "agent": agent,
                        "factor": v["factor"],
                        "value": v["value"],
                        "is_default": False,
                        "seed": seed,
                        "train_seconds": dt,
                        **cell,
                    }
                    for cell in readout
                )

    # emit the shared default readout into the other factor slots
    for v in VARIANTS:
        if not _is_default(v):
            continue
        key = (v["agent"], tuple(sorted(v["fields"].items())))
        for seed in seeds:
            readout, dt = trained[(key[0], key[1], seed)]
            for cell in readout:
                rows.append(
                    {
                        "agent": v["agent"],
                        "factor": v["factor"],
                        "value": v["value"],
                        "is_default": True,
                        "seed": seed,
                        "train_seconds": dt,
                        **cell,
                    }
                )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby(["agent", "factor", "value", "is_default"])
        .agg(
            n_cells=("reward_default", "size"),
            reward_mean=("reward_default", "mean"),
            reward_sd=("reward_default", "std"),
            skill_gain_mean=("skill_gain_total", "mean"),
            completion_mean=("completion_rate", "mean"),
            unique_actions_mean=("unique_actions", "mean"),
        )
        .reset_index()
        .sort_values(["agent", "factor", "value"])
    )
    summary.to_csv(OUT_CSV.with_name("sensitivity_summary.csv"), index=False)

    paired = paired_vs_default(df)
    if paired is not None:
        paired.to_csv(OUT_CSV.with_name("sensitivity_paired.csv"), index=False)

    total = time.time() - t_start
    runtime["total_seconds"] = round(total, 2)
    runtime["n_unique_trainings"] = len(train_keys) * len(list(seeds))
    runtime["episodes_per_run"] = episodes
    runtime["seeds"] = list(seeds)
    runtime["fixed_factors"] = FIXED_NOTE
    (OUT_CSV.with_name("sensitivity_runtime.json")).write_text(
        json.dumps(runtime, indent=2)
    )

    if verbose:
        print(f"\n-> {OUT_CSV}")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
        if paired is not None:
            print(paired.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
        print(f"total runtime: {total:.1f}s "
              f"({len(train_keys)} unique configs x {len(list(seeds))} seeds)")
    return df


def paired_vs_default(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Paired t + Wilcoxon of each value against the factor's default value."""
    from scipy import stats

    rows = []
    for (agent, factor), grp in df.groupby(["agent", "factor"]):
        piv = grp.pivot_table(
            index=["seed", "student_id", "eval_seed"],
            columns="value",
            values="reward_default",
        )
        defaults = [c for c in piv.columns if bool(
            grp.loc[grp["value"] == c, "is_default"].any()
        )]
        if len(defaults) != 1 or piv[defaults[0]].isna().any():
            continue
        ref = piv[defaults[0]]
        for value in piv.columns:
            if value == defaults[0]:
                continue
            col = piv[value].dropna()
            ref2 = ref[col.index]
            if len(col) < 3:
                continue
            t, p_t = stats.ttest_rel(col, ref2)
            try:
                _, p_w = stats.wilcoxon(col, ref2)
            except ValueError:          # pragma: no cover - all differences zero
                p_w = float("nan")
            rows.append(
                {
                    "agent": agent,
                    "factor": factor,
                    "value": value,
                    "default_value": float(defaults[0]),
                    "n_cells": int(len(col)),
                    "mean_diff_vs_default": float((col - ref2).mean()),
                    "t_stat": float(t),
                    "p_paired_t": float(p_t),
                    "p_wilcoxon": float(p_w),
                    "significant_05": bool(
                        min(p_t, p_w if p_w == p_w else 1.0) < 0.05
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Hyper-parameter sensitivity (train split)")
    p.add_argument("--episodes", type=int, default=SENSITIVITY_EPISODES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SENSITIVITY_SEEDS))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_study(
        episodes=args.episodes,
        seeds=tuple(args.seeds),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
