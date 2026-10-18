"""CP-5 — ablation study (TRAIN split only), Q-Learning + DQN only.

Two pre-registered ablations against the fully-configured model, plus the
full model re-read out on the identical cells as the reference:

====================  ==========================================================
variant               change w.r.t. ``full``
====================  ==========================================================
full                  the deployed agents, trained under the default reward,
                      26-dim state (reference; models come from the 5-seed
                      main experiment, training/CP-3)
reward_unshaped       reward = skill_gain_weight*skill_gain only: every shaping
                      term (success bonus, challenge, hint, time, frustration,
                      repetition penalties) set to 0.  Clip stays +-2.
                      Q-Learning + DQN.
state_ablated         observation = first 20 dims of the 26-dim state (the 6
                      ``error/<type>`` distribution features removed; deep net
                      input 26 -> 20).  DQN only - the tabular agents use
                      ``state_to_index`` bins that never contained the error
                      features, so there is nothing to ablate there.
====================  ==========================================================

Protocol (identical to the main experiment so ``full`` is directly reusable):

* episodes : ``TRAIN_EPISODES`` = 2000, seeds = (0,1,2,3,4)   (5 seeds)
* selection: greedy-val checkpoint selection on 4 TRAINING students
* readout  : greedy rollouts on 6 fixed TRAINING students x 2 eval seeds
             = 60 matched cells per (agent, variant); reward always scored
             under the default ``RewardConfig`` (common yardstick)
* stats    : Welch t-test (independent samples) of each ablation vs ``full``
* leakage  : resolve_students() rejects test ids; trainer refuses test pools

Output::

    results/ablation.csv              one row per (agent, variant, seed, cell)
    results/ablation_summary.csv      mean +/- sd per (agent, variant)
    results/ablation_stats.csv        Welch t vs full (+ Cohen's d)
    visualization/figures/ablation_reward_unshaped.png
    visualization/figures/ablation_state_ablated.png

Usage::

    python training/ablation.py [--episodes 2000] [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dqn_agent import DQNAgent, QNetwork, ReplayBuffer  # noqa: E402
from agents.q_learning_agent import QLearningAgent  # noqa: E402
from config import (  # noqa: E402
    DQNConfig,
    READOUT_EVAL_SEEDS,
    RESULTS_DIR,
    TabularConfig,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
)
from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from environment.reward import RewardConfig  # noqa: E402
from environment.state import STATE_DIM  # noqa: E402
from evaluation.audit_dqn import env_seed_for  # noqa: E402
from evaluation.strategies import build_strategy  # noqa: E402
from training.common import make_env  # noqa: E402
from training.reward_sensitivity import (  # noqa: E402
    _greedy_rollout,
    resolve_students,
    sensitivity_student_ids,
)
from training.trainer import train  # noqa: E402

OUT_CSV = RESULTS_DIR / "ablation.csv"
FIGURES_DIR = ROOT / "visualization" / "figures"

ABLATIONS = ("reward_unshaped", "state_ablated")
VARIANT_LABELS = {
    "full": "Full model (reference)",
    "reward_unshaped": "Ablation: reward = skill gain only",
    "state_ablated": "Ablation: state w/o error-type features",
}
ABLATED_DIM = STATE_DIM - 6  # 26 - 6 error-type one-hot features = 20

UNSHAPED_REWARD = RewardConfig(
    skill_gain_weight=40.0,
    success_bonus=0.0,
    challenge_appropriate=0.0,
    challenge_mismatch=0.0,
    hint_penalty=0.0,
    time_penalty=0.0,
    frustration_penalty=0.0,
    repetition_topic_penalty=0.0,
    repetition_problem_penalty=0.0,
    repetition_streak_penalty=0.0,
    # clip_min / clip_max stay at +-2 (kept, per the ablation definition)
)


# --------------------------------------------------------------------------
# State-ablated DQN: identical algorithm, 20-dim observation
# --------------------------------------------------------------------------
class StateAblatedDQNAgent(DQNAgent):
    """DQN whose observation drops the 6 ``error/<type>`` features.

    Only the observation passed to the network/replay changes; epsilon,
    learning rules, seeds and all other hyper-parameters are untouched.
    """

    name = "DQN (state-ablated)"

    def __init__(self, config: DQNConfig | None = None, seed: int = 0,
                 device: str | None = None) -> None:
        super().__init__(config=config, seed=seed, device=device)
        cfg = self.cfg
        # rebuild network + buffer at the ablated dimensionality
        self.online = QNetwork(
            state_dim=ABLATED_DIM, hidden1=cfg.hidden1, hidden2=cfg.hidden2
        ).to(self.device)
        self.target = QNetwork(
            state_dim=ABLATED_DIM, hidden1=cfg.hidden1, hidden2=cfg.hidden2
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, ABLATED_DIM, seed=seed)

    @staticmethod
    def _trim(state) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)[:ABLATED_DIM]

    def act(self, state, explore: bool = True) -> int:
        return super().act(self._trim(state), explore=explore)

    def recommend(self, state) -> int:
        return super().recommend(self._trim(state))

    def q_values(self, state) -> np.ndarray:
        return super().q_values(self._trim(state))

    def observe(self, state, action, reward, next_state, done) -> None:
        return super().observe(self._trim(state), action, reward,
                               self._trim(next_state), done)


# --------------------------------------------------------------------------
# Readout on the shared cells
# --------------------------------------------------------------------------
def greedy_readout(agent) -> list[dict]:
    ids = sensitivity_student_ids()
    students = resolve_students(ids)                      # <- leakage guard
    problems = load_problems()
    history = history_lookup(problems)
    pool = load_students()
    env = CodingPracticeEnv(problems=problems, students=pool, mode="train",
                            reward_config=RewardConfig())
    rows = []
    for s_idx, student in enumerate(students):
        for eval_seed in READOUT_EVAL_SEEDS:
            es = env_seed_for(eval_seed, s_idx)
            h = history.get(student.student_id, [])
            r, actions, log = _greedy_rollout(
                env, agent, student=student, history=h, env_seed=es
            )
            rows.append({
                "student_id": student.student_id,
                "eval_seed": eval_seed,
                "env_seed": es,
                "reward_default": r,
                "skill_gain_total": float(sum(i["skill_gain"] for i in log)),
                "completion_rate": float(np.mean([i["success"] for i in log])),
                "unique_actions": len(set(actions)),
                "n_steps": len(actions),
            })
    return rows


# --------------------------------------------------------------------------
# Study driver
# --------------------------------------------------------------------------
def run_study(
    episodes: int = TRAIN_EPISODES,
    seeds: Sequence[int] = TRAIN_SEEDS,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    runtime: dict[str, float] = {}
    t_start = time.time()

    # ---- full reference: CP-3 models, same readout cells -----------------
    for seed in seeds:
        for agent_name in ("q_learning", "dqn"):
            agent = build_strategy(agent_name, model_seed=seed)
            for cell in greedy_readout(agent):
                rows.append({"agent": agent_name, "variant": "full",
                             "seed": seed, **cell})
        if verbose:
            print(f"  [ablation] full reference read out (seed={seed})")

    # ---- reward_unshaped: Q-Learning + DQN -------------------------------
    for agent_name in ("q_learning", "dqn"):
        for seed in seeds:
            env = make_env()
            env.reward_config = UNSHAPED_REWARD
            if agent_name == "q_learning":
                agent = QLearningAgent(config=TabularConfig(), seed=seed)
            else:
                agent = DQNAgent(config=DQNConfig(), seed=seed)
            t0 = time.time()
            if verbose:
                print(f"  [ablation] {agent_name:<12} reward_unshaped  "
                      f"seed={seed} ({episodes} eps)")
            train(agent, env, episodes=episodes, seed=seed,
                  name=f"{agent_name}_abl_reward_unshaped",
                  verbose=False, save=True, save_model=False)
            runtime[f"{agent_name}_reward_unshaped_seed{seed}"] = round(
                time.time() - t0, 2)
            for cell in greedy_readout(agent):
                rows.append({"agent": agent_name, "variant": "reward_unshaped",
                             "seed": seed, **cell})

    # ---- state_ablated: DQN only (tabular bins never used the features) --
    for seed in seeds:
        env = make_env()
        agent = StateAblatedDQNAgent(config=DQNConfig(), seed=seed)
        t0 = time.time()
        if verbose:
            print(f"  [ablation] dqn          state_ablated       "
                  f"seed={seed} ({episodes} eps)")
        train(agent, env, episodes=episodes, seed=seed,
              name="dqn_abl_state_ablated",
              verbose=False, save=True, save_model=False)
        runtime[f"dqn_state_ablated_seed{seed}"] = round(time.time() - t0, 2)
        for cell in greedy_readout(agent):
            rows.append({"agent": "dqn", "variant": "state_ablated",
                         "seed": seed, **cell})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby(["agent", "variant"])
        .agg(n_cells=("reward_default", "size"),
             reward_mean=("reward_default", "mean"),
             reward_sd=("reward_default", "std"),
             skill_gain_mean=("skill_gain_total", "mean"),
             completion_mean=("completion_rate", "mean"),
             unique_actions_mean=("unique_actions", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_CSV.with_name("ablation_summary.csv"), index=False)

    stats = welch_vs_full(df)
    stats.to_csv(OUT_CSV.with_name("ablation_stats.csv"), index=False)

    runtime["total_seconds"] = round(time.time() - t_start, 2)
    runtime["episodes_per_run"] = episodes
    runtime["seeds"] = list(seeds)
    print(f"\ntotal ablation runtime: {runtime['total_seconds']:.1f}s")

    if verbose:
        print(f"\n-> {OUT_CSV}")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
        print("\nWelch t-test vs full:")
        print(stats.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    return df


def welch_vs_full(df: pd.DataFrame) -> pd.DataFrame:
    """Welch (unequal-variance, independent) t-test, ablation vs full."""
    from scipy import stats as sps

    rows = []
    for (agent, variant), grp in df.groupby(["agent", "variant"]):
        if variant == "full":
            continue
        full = df[(df["agent"] == agent) & (df["variant"] == "full")]
        a = grp["reward_default"].to_numpy()
        b = full["reward_default"].to_numpy()
        t, p = sps.ttest_ind(a, b, equal_var=False)
        # Cohen's d (pooled SD)
        pooled = np.sqrt(((a.std(ddof=1) ** 2) * (len(a) - 1)
                          + (b.std(ddof=1) ** 2) * (len(b) - 1))
                         / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
        rows.append({
            "agent": agent,
            "ablation": variant,
            "n_ablation_cells": int(len(a)),
            "n_full_cells": int(len(b)),
            "ablation_mean": float(a.mean()),
            "ablation_sd": float(a.std(ddof=1)),
            "full_mean": float(b.mean()),
            "full_sd": float(b.std(ddof=1)),
            "mean_diff_vs_full": float(a.mean() - b.mean()),
            "welch_t": float(t),
            "p_welch": float(p),
            "cohens_d": float(d),
            "significant_05": bool(p < 0.05),
        })
    return pd.DataFrame(rows)


def plot_ablations(summary: pd.DataFrame) -> list[Path]:
    """One bar chart per ablation: mean +/- sd reward, full vs ablated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    palette = {"full": "#1f3b73", "ablated": "#2a9d8f"}   # navy / teal
    paths: list[Path] = []

    for variant in ABLATIONS:
        sub = summary[summary["variant"].isin(["full", variant])]
        if sub.empty:
            continue
        scope = ("q_learning", "dqn") if variant == "reward_unshaped" else ("dqn",)
        agents = [a for a in scope if a in set(sub["agent"])]
        x = np.arange(len(agents))
        width = 0.36
        full_vals, abl_vals, full_sd, abl_sd = [], [], [], []
        for a in agents:
            f = sub[(sub["agent"] == a) & (sub["variant"] == "full")].iloc[0]
            v = sub[(sub["agent"] == a) & (sub["variant"] == variant)].iloc[0]
            full_vals.append(f["reward_mean"]); full_sd.append(f["reward_sd"])
            abl_vals.append(v["reward_mean"]); abl_sd.append(v["reward_sd"])

        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
        b1 = ax.bar(x - width / 2, full_vals, width, yerr=full_sd,
                    color=palette["full"], label="Full (reference)",
                    edgecolor="white", capsize=4, error_kw={"lw": 1.2})
        b2 = ax.bar(x + width / 2, abl_vals, width, yerr=abl_sd,
                    color=palette["ablated"], label=VARIANT_LABELS[variant],
                    edgecolor="white", capsize=4, error_kw={"lw": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", " ").title() for a in agents])
        ax.set_ylabel("Greedy readout reward (mean $\\pm$ sd)")
        ax.set_title(f"Ablation: {VARIANT_LABELS[variant]}\n"
                     "(train split, 5 seeds x 12 cells, error bars = sd)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        out = FIGURES_DIR / f"ablation_{variant}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Ablation study (train split)")
    p.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    df = run_study(episodes=args.episodes, seeds=tuple(args.seeds),
                   verbose=not args.quiet)
    summary = pd.read_csv(OUT_CSV.with_name("ablation_summary.csv"))
    pngs = plot_ablations(summary)
    for pth in pngs:
        print(f"-> {pth}")


if __name__ == "__main__":
    main()
