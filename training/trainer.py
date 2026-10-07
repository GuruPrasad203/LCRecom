"""Shared training loop for every RL agent.

One *episode* = one complete practice session of ``EPISODE_LEN`` recommended
problems for one simulated student.

Per episode we log:

    reward           sum of (clipped) rewards
    skill_gain       total mastery gained across all topics
    completions      problems solved
    hints            hints requested
    solve_time       mean solve time (minutes)
    calibration      fraction of recommendations in the "appropriate" band
    repeat_rate      fraction of steps that repeated the previous topic
    frustration      end-of-episode frustration
    avg_mastery      mean mastery over the 12 topics
    epsilon          exploration rate (tabular agents)

Artifacts per run:

    results/training/<name>_seed<s>.csv   learning curve
    models/<name>_seed<s>.npz|.pt         learned policy
    results/tensorboard/<name>_seed<s>/   TensorBoard logs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from config import EPISODE_LEN, MODELS_DIR, RESULTS_DIR, TENSORBOARD_DIR, TOPICS
from data.generate_data import history_lookup
from environment.coding_env import CodingPracticeEnv

TRAINING_DIR = RESULTS_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_COLUMNS = [
    "episode", "reward", "skill_gain", "completions", "hints", "solve_time",
    "calibration", "repeat_rate", "frustration", "avg_mastery", "epsilon",
    "mastery_gain",
]


def build_history() -> dict[str, list[dict]]:
    return history_lookup()


def run_episode(
    env: CodingPracticeEnv,
    agent,
    *,
    student,
    history: Optional[list[dict]],
    env_seed: int,
    explore: bool = True,
) -> dict[str, Any]:
    """Play one full episode; returns per-episode metrics."""
    options = {"student_id": student.student_id}
    if history is not None:
        options["history"] = history
    state, info = env.reset(seed=env_seed, options=options)
    init_mastery = env.mastery.copy()
    if explore and hasattr(agent, "begin_episode"):
        agent.begin_episode()

    total_reward = 0.0
    skill_gain = 0.0
    completions = 0
    hints = 0
    times: list[float] = []
    appropriate = 0
    repeats = 0
    steps = 0

    while True:
        action = agent.act(state, explore=explore) if explore else agent.recommend(state)
        next_state, reward, terminated, truncated, sinfo = env.step(action)
        if explore and hasattr(agent, "observe"):
            agent.observe(state, action, reward, next_state, terminated or truncated)

        total_reward += reward
        skill_gain += sinfo["skill_gain"]
        completions += int(sinfo["success"])
        hints += sinfo["hints_used"]
        times.append(sinfo["solve_time"])
        appropriate += int(sinfo["challenge"] == "appropriate")
        repeats += int(sinfo["repeated_topic"])
        steps += 1

        state = next_state
        if terminated or truncated:
            break

    if explore and hasattr(agent, "end_episode"):
        agent.end_episode()

    return {
        "reward": total_reward,
        "skill_gain": skill_gain,
        "completions": completions,
        "hints": hints,
        "solve_time": float(np.mean(times)),
        "calibration": appropriate / max(steps, 1),
        "repeat_rate": repeats / max(steps - 1, 1) if steps > 1 else 0.0,
        "frustration": env.frustration,
        "avg_mastery": float(env.mastery.mean()),
        "epsilon": float(getattr(agent, "epsilon", 0.0)),
        "mastery_gain": float(env.mastery.mean() - init_mastery.mean()),
    }


def train(
    agent,
    env: CodingPracticeEnv,
    *,
    episodes: int,
    seed: int,
    name: str,
    save: bool = True,
    log_every: int = 200,
    verbose: bool = True,
    extra_suffix: str = "",
    history: Optional[dict[str, list[dict]]] = None,
    save_model: bool = True,
) -> pd.DataFrame:
    """Train ``agent`` in ``env`` for ``episodes`` episodes (train split only)."""
    student_rng = np.random.default_rng(seed)
    history = history if history is not None else build_history()
    students = [s for s in env.students if s.split == "train"]
    if not students:
        raise ValueError(
            "training requires students from the 'train' split - the held-out "
            "test students must never be used for training or tuning"
        )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = TENSORBOARD_DIR / f"{name}{extra_suffix}_seed{seed}"
        writer = SummaryWriter(log_dir=str(tb_dir))
    except Exception:  # pragma: no cover - tensorboard optional
        writer = None

    rows: list[dict[str, Any]] = []
    # --- early stopping / best-checkpoint selection --------------------
    # Selection uses *deterministic greedy* rollouts on a fixed subset of
    # TRAINING students (never the held-out test split).  Training reward
    # itself cannot be used for selection because it is measured under the
    # epsilon-greedy *behaviour* policy (epsilon is still large mid-training)
    # whereas the deployed policy is greedy.
    trailing_window = 200
    checkpoint_every = 100
    val_students = students[:4]
    val_seed_base = seed * 7919 + 13
    best_score = -np.inf
    best_snap: dict | None = None
    best_episode = -1

    def greedy_val_score() -> float:
        scores = []
        for vi, st in enumerate(val_students):
            m = run_episode(
                env, agent, student=st,
                history=history.get(st.student_id),
                env_seed=val_seed_base + vi, explore=False,
            )
            scores.append(m["reward"])
        return float(np.mean(scores))
    t0 = time.time()
    for ep in range(episodes):
        student = students[int(student_rng.integers(len(students)))]
        metrics = run_episode(
            env,
            agent,
            student=student,
            history=history.get(student.student_id),
            env_seed=seed * 100_003 + ep,
            explore=True,
        )
        metrics["episode"] = ep
        rows.append(metrics)

        if (ep + 1) % checkpoint_every == 0 or ep == episodes - 1:
            score = greedy_val_score()
            if score > best_score:
                best_score = score
                best_episode = ep
                best_snap = agent.snapshot() if hasattr(agent, "snapshot") else None

        if writer is not None:
            writer.add_scalar("episode/reward", metrics["reward"], ep)
            writer.add_scalar("episode/skill_gain", metrics["skill_gain"], ep)
            writer.add_scalar("episode/completion_rate",
                              metrics["completions"] / EPISODE_LEN, ep)
            writer.add_scalar("episode/calibration", metrics["calibration"], ep)
            writer.add_scalar("episode/avg_mastery", metrics["avg_mastery"], ep)
            writer.add_scalar("hyper/epsilon", metrics["epsilon"], ep)

        if verbose and (ep % log_every == 0 or ep == episodes - 1):
            window = rows[-min(50, len(rows)):]
            mr = np.mean([r["reward"] for r in window])
            mg = np.mean([r["skill_gain"] for r in window])
            mc = np.mean([r["completions"] for r in window]) / EPISODE_LEN
            print(
                f"  [{name}{extra_suffix} seed={seed}] ep {ep:5d} "
                f"reward={mr:7.3f} skill_gain={mg:6.3f} completion={mc:5.2f} "
                f"eps={metrics['epsilon']:.3f} "
                f"({time.time() - t0:5.1f}s)"
            )

    df = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    if save:
        csv_path = TRAINING_DIR / f"{name}{extra_suffix}_seed{seed}.csv"
        df.to_csv(csv_path, index=False)
        final_score = greedy_val_score()
        if save_model:
            # keep the final weights for inspection ...
            _save_policy(agent, name, seed, extra_suffix=f"{extra_suffix}_last")
        # ... but ship the best *greedy-val* checkpoint as the primary model
        if best_snap is not None:
            agent.restore(best_snap)
        if save_model:
            _save_policy(agent, name, seed, extra_suffix)
        meta = {
            "best_episode": best_episode,
            "best_greedy_val_reward": round(best_score, 4),
            "final_greedy_val_reward": round(final_score, 4),
            "selection_metric": "greedy rollout reward on 4 TRAINING students",
            "val_students": [st.student_id for st in val_students],
        }
        (TRAINING_DIR / f"{name}{extra_suffix}_seed{seed}.json").write_text(
            __import__("json").dumps(meta, indent=2)
        )
        if verbose:
            print(f"  [{name}{extra_suffix} seed={seed}] best checkpoint at episode "
                  f"{best_episode} (greedy val reward {best_score:.2f} vs final {final_score:.2f})")
    if writer is not None:
        writer.close()
    return df


def _save_policy(agent, name: str, seed: int, extra_suffix: str = "") -> Path:
    if not hasattr(agent, "save"):
        raise TypeError(f"agent {name} has no save()")
    is_tabular = isinstance(getattr(agent, "Q", None), np.ndarray)
    ext = ".npz" if is_tabular else ".pt"
    path = MODELS_DIR / f"{name}{extra_suffix}_seed{seed}{ext}"
    agent.save(path)
    return path


def load_history() -> dict[str, list[dict]]:
    return build_history()
