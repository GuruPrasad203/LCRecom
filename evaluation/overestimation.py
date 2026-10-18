"""CP-6 — Q-value overestimation study: DQN vs Double DQN.

Classic source of DQN's overestimation: the target

    y_DQN   = r + gamma * max_a' Q_target(s', a')

takes a **max over the same noisy estimates** used for learning, so the
maximum of noisy values is biased upward.  Double DQN decouples selection
from evaluation:

    y_DDQN  = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

This study measures both biases directly, on *identical* transitions.

Protocol
--------
* agents     : DQN and Double DQN, seed 0 (the main experiment's checkpoints)
* states     : 100 pre-step (state, greedy action) pairs per agent from greedy
               rollouts on TRAINING students (5 students x 20 steps,
               eval_seed=100)  -> 200 pairs total
* samples    : 32 independent one-step transitions per pair, drawn by exact
               snapshot/restore of the environment's mutable state with a
               fresh RNG stream per sample (so the 32 samples really are
               distinct draws from the transition distribution)
* belief     : max_a Q_online(s, a)   (what the agent's greedy value says)
* targets    : E[y_DQN] and E[y_DDQN] over the 32 samples, computed from the
               *same* agent's networks
* overest.   : belief - E[y_rule]   (positive = the agent over-believes)

Output::

    results/overestimation.csv           one row per (agent, pair)
    results/overestimation_summary.csv   mean +/- sd per (agent, rule)
    visualization/figures/overestimation.png

Usage::

    python evaluation/overestimation.py
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    HISTORY_WINDOW,
    MODELS_DIR,
    N_ERROR_WINDOW,
    READOUT_EVAL_SEEDS,
    RESULTS_DIR,
)
from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from environment.reward import RewardConfig  # noqa: E402
from evaluation.audit_dqn import env_seed_for  # noqa: E402

OUT_CSV = RESULTS_DIR / "overestimation.csv"
FIGURES_DIR = ROOT / "visualization" / "figures"

N_SAMPLES = 32
N_STUDENTS = 5            # 5 students x 20 steps = 100 pairs per agent
ALGOS = {
    "dqn": ("DQN", "dqn_seed0.pt"),
    "double_dqn": ("Double DQN", "double_dqn_seed0.pt"),
}

# every attribute of CodingPracticeEnv mutated by step()
MUTABLE = (
    "mastery", "frustration", "consec_failures", "current_difficulty",
    "recent", "recent_errors", "steps", "attempted", "last_topic_idx",
    "last_problem_id", "same_topic_streak", "student", "history",
    "_episode_log",
)


def load_agent(key: str):
    from agents.dqn_agent import DQNAgent
    from agents.double_dqn_agent import DoubleDQNAgent

    cls = DoubleDQNAgent if key == "double_dqn" else DQNAgent
    return cls.load(MODELS_DIR / ALGOS[key][1])


# --------------------------------------------------------------------------
# Exact snapshot / restore of the env's dynamic state
# --------------------------------------------------------------------------
def snapshot(env: CodingPracticeEnv) -> dict:
    return {
        "mastery": env.mastery.copy(),
        "frustration": float(env.frustration),
        "consec_failures": int(env.consec_failures),
        "current_difficulty": int(env.current_difficulty),
        "recent": list(env.recent),
        "recent_errors": list(env.recent_errors),
        "steps": int(env.steps),
        "attempted": list(env.attempted),
        "last_topic_idx": env.last_topic_idx,
        "last_problem_id": env.last_problem_id,
        "same_topic_streak": int(env.same_topic_streak),
        "student": env.student,
        "history": env.history,
        "_episode_log": env._episode_log,
    }


def restore(env: CodingPracticeEnv, snap: dict) -> None:
    env.mastery = snap["mastery"].copy()
    env.frustration = snap["frustration"]
    env.consec_failures = snap["consec_failures"]
    env.current_difficulty = snap["current_difficulty"]
    env.recent = deque(snap["recent"], maxlen=HISTORY_WINDOW)
    env.recent_errors = deque(snap["recent_errors"], maxlen=N_ERROR_WINDOW)
    env.steps = snap["steps"]
    env.attempted = list(snap["attempted"])
    env.last_topic_idx = snap["last_topic_idx"]
    env.last_problem_id = snap["last_problem_id"]
    env.same_topic_streak = snap["same_topic_streak"]
    env.student = snap["student"]
    # history / _episode_log are append-only logs; step() appends into them,
    # so restore the same list object identity is not required for dynamics
    env.history = list(snap["history"])
    env._episode_log = list(snap["_episode_log"])


# --------------------------------------------------------------------------
# Pair collection: (state, greedy action) from the agent's own rollout
# --------------------------------------------------------------------------
def collect_pairs(agent, env: CodingPracticeEnv, history: dict,
                  student_ids: list[str], eval_seed: int) -> list[dict]:
    pairs: list[dict] = []
    for s_idx, sid in enumerate(student_ids):
        state, _ = env.reset(
            seed=env_seed_for(eval_seed, s_idx),
            options={"student_id": sid, "history": history.get(sid, [])},
        )
        while True:
            action = agent.recommend(state)
            snap = snapshot(env)
            pairs.append({
                "student_id": sid,
                "step": int(env.steps),
                "action": int(action),
                "state": state.copy(),
                "snap": snap,
            })
            state, _r, term, trunc, _info = env.step(action)
            if term or trunc:
                break
    return pairs


# --------------------------------------------------------------------------
# 32-sample expected targets per pair
# --------------------------------------------------------------------------
def sample_targets(env: CodingPracticeEnv, snap: dict, action: int,
                   agent, base_seed: int, pair_idx: int) -> dict:
    y_max, y_double, ys = [], [], []
    agree = 0
    for i in range(N_SAMPLES):
        restore(env, snap)
        # independent stream per sample: dynamics are identical, only the
        # RNG differs -> 32 distinct draws from the transition distribution
        env.np_random = np.random.default_rng(base_seed + pair_idx * 1009 + i)
        _s2, r, term, trunc, _info = env.step(action)
        done = float(term or trunc)
        s2 = _s2.astype(np.float32)

        with torch.no_grad():
            x = torch.as_tensor(s2, device=agent.device).unsqueeze(0)
            q_target = agent.target(x).squeeze(0)
            q_online = agent.online(x).squeeze(0)
            v_max = float(q_target.max().item())                       # DQN rule
            a_star = int(q_online.argmax().item())
            a_targ = int(q_target.argmax().item())
            v_double = float(q_target[a_star].item())                  # DDQN rule
            agree += int(a_star == a_targ)

        g = agent.cfg.gamma
        y_max.append(float(r) + g * (1.0 - done) * v_max)
        y_double.append(float(r) + g * (1.0 - done) * v_double)
        ys.append(float(r))

    return {
        "E_y_dqn_rule": float(np.mean(y_max)),
        "E_y_ddqn_rule": float(np.mean(y_double)),
        "sd_y_dqn_rule": float(np.std(y_max, ddof=1)),
        "sd_y_ddqn_rule": float(np.std(y_double, ddof=1)),
        "E_reward": float(np.mean(ys)),
        "argmax_agreement": agree / N_SAMPLES,
    }


def belief_max_q(agent, state: np.ndarray) -> float:
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32,
                            device=agent.device).unsqueeze(0)
        return float(agent.online(x).max().item())


# --------------------------------------------------------------------------
def run_study(verbose: bool = True) -> pd.DataFrame:
    problems = load_problems()
    pool = load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=pool, mode="train",
                            reward_config=RewardConfig())
    train_students = [s for s in pool if s.split == "train"]
    ids = [s.student_id for s in train_students[:N_STUDENTS]]
    eval_seed = READOUT_EVAL_SEEDS[0]           # fixed geometry: 100 pairs/agent

    rows: list[dict] = []
    t0 = time.time()
    for key, (label, _) in ALGOS.items():
        agent = load_agent(key)
        pairs = collect_pairs(agent, env, history, ids, eval_seed)
        if verbose:
            print(f"  [{label}] {len(pairs)} (state, greedy action) pairs "
                  f"from {len(ids)} train students")
        for pi, p in enumerate(pairs):
            t = sample_targets(env, p["snap"], p["action"], agent,
                               base_seed=7_000_000, pair_idx=pi)
            belief = belief_max_q(agent, p["state"])
            rows.append({
                "agent": key,
                "pair_idx": pi,
                "student_id": p["student_id"],
                "step": p["step"],
                "action": p["action"],
                "belief_max_q": belief,
                **t,
                "overest_dqn_rule": belief - t["E_y_dqn_rule"],
                "overest_ddqn_rule": belief - t["E_y_ddqn_rule"],
                "rule_gap": t["E_y_dqn_rule"] - t["E_y_ddqn_rule"],
                "n_samples": N_SAMPLES,
            })
        if verbose:
            print(f"  [{label}] sampled {len(pairs)} x {N_SAMPLES} transitions "
                  f"({time.time() - t0:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary_rows = []
    for agent_key, grp in df.groupby("agent"):
        for rule, col in (("dqn_rule", "overest_dqn_rule"),
                          ("ddqn_rule", "overest_ddqn_rule")):
            v = grp[col]
            summary_rows.append({
                "agent": agent_key,
                "rule": rule,
                "n_pairs": int(len(v)),
                "overestimation_mean": float(v.mean()),
                "overestimation_sd": float(v.std(ddof=1)),
                "frac_overestimating": float((v > 0).mean()),
                "belief_mean": float(grp["belief_max_q"].mean()),
                "target_mean": float((grp["belief_max_q"] - v).mean()),
                "argmax_agreement": float(grp["argmax_agreement"].mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_CSV.with_name("overestimation_summary.csv"), index=False)

    runtime = {"total_seconds": round(time.time() - t0, 2),
               "n_pairs": int(len(df)), "n_samples": N_SAMPLES}
    import json
    (OUT_CSV.with_name("overestimation_runtime.json")).write_text(
        json.dumps(runtime, indent=2)
    )

    if verbose:
        print(f"\n-> {OUT_CSV}")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
        print(f"runtime: {runtime['total_seconds']:.1f}s")
    return df


def plot_overestimation(df: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    palette = {"dqn": "#1f3b73", "double_dqn": "#2a9d8f"}   # navy / teal

    data, labels, colors = [], [], []
    for agent_key, (label, _) in ALGOS.items():
        grp = df[df["agent"] == agent_key]
        for rule, col in (("DQN target", "overest_dqn_rule"),
                          ("Double target", "overest_ddqn_rule")):
            data.append(grp[col].to_numpy())
            labels.append(f"{label}\n{rule}")
            colors.append(palette[agent_key])

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="white", lw=1.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax.axhline(0.0, color="#444", lw=1.0, ls="--")
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Overestimation  max$_a$Q$_{online}(s,a)$ $-$ E[target]")
    ax.set_title("Q-value overestimation by target rule\n"
                 "(200 pre-step states, 32 transition samples each, seed 0)")
    ax.spines[["top", "right"]].set_visible(False)
    # annotate means
    for i, d in enumerate(data, start=1):
        m = d.mean()
        ax.plot([i], [m], marker="D", ms=5, color="#e76f51", zorder=5)
        ax.annotate(f"mean {m:+.2f}", (i, m), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color="#e76f51")
    fig.tight_layout()
    out = FIGURES_DIR / "overestimation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    df = run_study()
    png = plot_overestimation(df)
    print(f"-> {png}")


if __name__ == "__main__":
    main()
