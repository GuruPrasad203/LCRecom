"""Phase 6 — the ten figures of the final report.

Every figure is produced from the artifacts written by Phases 1-5 (nothing is
re-simulated except the two greedy action-distribution readouts, which reuse
the *locked* readout geometry of ``config.py``).

====  =========================================  =================================
  #   figure                                     source
====  =========================================  =================================
  1   learning curves                            results/training/*_seed*.csv
  2   skill-gain bars (mean +/- sd, 30 cells)    results/evaluation/summary_by_strategy.csv
  3   reward bars                                results/evaluation/summary_by_strategy.csv
  4   2x2 metrics panel                          results/evaluation/summary_by_strategy.csv
  5   skill-gain box plots (30 cells)            results/evaluation/cell_means.csv
  6   reward-sensitivity chart                   results/reward_sensitivity.csv
  7   DQN vs Q-Learning 12x3 action heatmaps     greedy readout (config geometry)
  8   calibration - frustration scatter          results/evaluation/cell_means.csv
  9   significance heatmap (paired p-values)      results/evaluation/per_run.csv
 10   per-topic mastery-gain heatmaps            results/evaluation/per_run.csv
====  =========================================  =================================

Usage::

    python visualization/plots.py            # writes visualization/figures/*.png
    python visualization/plots.py 3 7        # only the named figure numbers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    N_ACTIONS,
    N_TOPICS,
    READOUT_EVAL_SEEDS,
    READOUT_N_STUDENTS,
    RESULTS_DIR,
    TOPICS,
    TRAIN_SEEDS,
)
from evaluation.strategies import (  # noqa: E402
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    STRATEGY_TYPE,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
EVAL_DIR = RESULTS_DIR / "evaluation"
TRAIN_DIR = RESULTS_DIR / "training"

sns.set_theme(style="whitegrid", context="paper")

STRATEGY_COLORS = {
    "Random": "#9e9e9e",
    "Difficulty-Based": "#c9b458",
    "Prerequisite-Based": "#5d6d7e",
    "SARSA": "#1f77b4",
    "Q-Learning": "#2ca02c",
    "DQN": "#ff7f0e",
    "Double DQN": "#d62728",
}
LABELS = [STRATEGY_LABELS[n] for n in STRATEGY_ORDER]


def _colors(labels: list[str]) -> list[str]:
    return [STRATEGY_COLORS[l] for l in labels]


def _summary() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "summary_by_strategy.csv")
    df["order"] = df["strategy"].map({l: i for i, l in enumerate(LABELS)})
    return df.sort_values("order").reset_index(drop=True)


def _cells() -> pd.DataFrame:
    return pd.read_csv(EVAL_DIR / "cell_means.csv")


# --------------------------------------------------------------------- #
# 1. learning curves
# --------------------------------------------------------------------- #
def fig_learning_curves() -> plt.Figure:
    curves = {
        "q_learning": "Q-Learning",
        "sarsa": "SARSA",
        "dqn": "DQN",
        "double_dqn": "Double DQN",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, label in curves.items():
        runs = []
        for seed in TRAIN_SEEDS:
            p = TRAIN_DIR / f"{name}_seed{seed}.csv"
            if not p.exists():
                continue
            s = pd.read_csv(p)["reward"].rolling(100, min_periods=20).mean()
            runs.append(s.to_numpy(dtype=float))
        if not runs:
            continue
        arr = np.vstack(runs)
        valid = ~np.isnan(arr).all(axis=0)   # leading rolling-window NaNs
        arr = arr[:, valid]
        x = np.flatnonzero(valid)
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax.plot(x, mean, color=STRATEGY_COLORS[label], lw=2.0, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=STRATEGY_COLORS[label],
                        alpha=0.15, linewidth=0)
    ax.set_xlabel("training episode")
    ax.set_ylabel("episode reward (100-episode moving average)")
    ax.set_title("Learning curves: mean $\\pm$ sd over 3 training seeds")
    ax.legend(frameon=True)
    ax.set_xlim(0, None)
    return fig


# --------------------------------------------------------------------- #
# 2. skill-gain bars
# --------------------------------------------------------------------- #
def fig_skill_gain_bars() -> plt.Figure:
    df = _summary().sort_values("skill_gain_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(df))
    ax.bar(x, df["skill_gain_mean"], yerr=df["skill_gain_std"],
           color=_colors(list(df["strategy"])), capsize=4, edgecolor="white")
    for xi, m in zip(x, df["skill_gain_mean"]):
        ax.text(xi, m + df["skill_gain_std"].max() * 0.03, f"{m:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["strategy"], rotation=20, ha="right")
    ax.set_ylabel("mean mastery gain per topic")
    ax.set_title("Skill gain (mean $\\pm$ sd over 30 held-out cells)")
    ax.set_ylim(0, None)
    return fig


# --------------------------------------------------------------------- #
# 3. reward bars
# --------------------------------------------------------------------- #
def fig_reward_bars() -> plt.Figure:
    df = _summary().sort_values("total_reward_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(df))
    ax.bar(x, df["total_reward_mean"], yerr=df["total_reward_std"],
           color=_colors(list(df["strategy"])), capsize=4, edgecolor="white")
    for xi, m in zip(x, df["total_reward_mean"]):
        ax.text(xi, m + df["total_reward_std"].max() * 0.03, f"{m:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["strategy"], rotation=20, ha="right")
    ax.set_ylabel("total session reward")
    ax.set_title("Reward collected on the held-out test students "
                 "(mean $\\pm$ sd, 30 cells)")
    ax.axhline(0.0, color="black", lw=0.8)
    return fig


# --------------------------------------------------------------------- #
# 4. 2x2 metrics panel
# --------------------------------------------------------------------- #
def fig_metrics_panel() -> plt.Figure:
    df = _summary()
    metrics = [
        ("skill_gain", "Skill gain per topic"),
        ("completion_rate", "Completion rate"),
        ("calibration", "Calibration (appropriate challenge)"),
        ("frustration_end", "End-of-session frustration"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        d = df.sort_values(f"{metric}_mean", ascending=False)
        x = np.arange(len(d))
        ax.bar(x, d[f"{metric}_mean"], yerr=d[f"{metric}_std"],
               color=_colors(list(d["strategy"])), capsize=3, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(d["strategy"], rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
    fig.suptitle("Key metrics (mean $\\pm$ sd over 30 held-out cells)", y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# 5. skill-gain box plots
# --------------------------------------------------------------------- #
def fig_skill_gain_boxplots() -> plt.Figure:
    cells = _cells()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = [l for l in LABELS if l in set(cells["strategy"])]
    data = [cells.loc[cells.strategy == l, "skill_gain"].to_numpy() for l in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4))
    for patch, label in zip(bp["boxes"], order):
        patch.set_facecolor(STRATEGY_COLORS[label])
        patch.set_alpha(0.85)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("mean mastery gain per topic")
    ax.set_title("Distribution of skill gain over the 30 held-out cells")
    return fig


# --------------------------------------------------------------------- #
# 6. reward sensitivity
# --------------------------------------------------------------------- #
def fig_reward_sensitivity() -> plt.Figure:
    df = pd.read_csv(RESULTS_DIR / "reward_sensitivity.csv")
    order = (
        df.groupby("variant")["reward_default"].mean().sort_values(ascending=False).index.tolist()
    )
    pretty = {
        "default": "default\n(reference)",
        "gain_half": "gain\nx0.5",
        "gain_double": "gain\nx2",
        "time_penalty_off": "time pen.\noff",
        "repetition_x2": "repetition\nx2",
        "success_bonus_low": "success\nbonus low",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    data = [df.loc[df.variant == v, "reward_default"].to_numpy() for v in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, v in zip(bp["boxes"], order):
        patch.set_facecolor(STRATEGY_COLORS["Q-Learning"] if v == "default" else "#c9b458")
        patch.set_alpha(0.9)
    ax.axhline(
        df.loc[df.variant == "default", "reward_default"].mean(),
        color="black", ls="--", lw=1, label="default mean",
    )
    ax.set_xticklabels([pretty[v] for v in order], fontsize=8)
    ax.set_ylabel("reward scored under the DEFAULT reward")
    ax.set_title("(a) Greedy readout, 6 train students x 2 eval seeds")
    ax.legend(fontsize=8)

    ax = axes[1]
    piv = df.pivot_table(index=["seed", "student_id", "eval_seed"],
                         columns="variant", values="reward_default")
    ref = piv["default"]
    variants = [v for v in order if v != "default"]
    diff = [(piv[v] - ref).to_numpy() for v in variants]
    means = [d.mean() for d in diff]
    sds = [d.std(ddof=1) for d in diff]
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=sds, color="#c9b458", capsize=4, edgecolor="white")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[v] for v in variants], fontsize=8)
    ax.set_ylabel("paired difference vs default")
    ax.set_title("(b) Same cells, paired with the default reward\n"
                 "(all $p > 0.28$ paired t / Wilcoxon)")
    fig.suptitle("Reward sensitivity: conclusions do not depend on the reward weights",
                 y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# 7. action heatmaps: DQN vs Q-Learning
# --------------------------------------------------------------------- #
def _action_probabilities(name: str) -> np.ndarray:
    """Mean greedy action probability over the 6 train students x 2 eval seeds,
    averaged over the 3 model seeds (12 x 3 = 36 cells)."""
    from evaluation.audit_dqn import (  # local import: heavy module
        AUDIT_EVAL_SEEDS,
        _greedy_rollout,
        audit_students,
        env_seed_for,
    )
    from data.generate_data import history_lookup, load_problems, load_students
    from environment.coding_env import CodingPracticeEnv
    from evaluation.strategies import build_strategy

    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    studs = audit_students()
    seeds = (0, 1, 2) if name in ("sarsa", "q_learning", "dqn", "double_dqn") else (0,)

    totals = np.zeros(N_ACTIONS, dtype=float)
    for seed in seeds:
        rec = build_strategy(name, model_seed=seed, random_seed=seed)
        counts = np.zeros(N_ACTIONS, dtype=float)
        for s_idx, st in enumerate(studs):
            for eval_seed in AUDIT_EVAL_SEEDS:
                _, actions, _ = _greedy_rollout(
                    env, rec, student=st,
                    history=history.get(st.student_id),
                    env_seed=env_seed_for(eval_seed, s_idx),
                )
                counts += np.bincount(actions, minlength=N_ACTIONS)
        totals += counts / max(counts.sum(), 1)
    return (totals / len(seeds)).reshape(N_TOPICS, 3)


def fig_action_heatmaps() -> plt.Figure:
    dqn = _action_probabilities("dqn")
    ql = _action_probabilities("q_learning")
    vmax = max(dqn.max(), ql.max())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2), sharey=True)
    for ax, mat, title in (
        (axes[0], dqn, "DQN (deep RL)"),
        (axes[1], ql, "Q-Learning (tabular RL)"),
    ):
        sns.heatmap(mat, ax=ax, cmap="viridis", vmin=0.0, vmax=vmax,
                    annot=True, fmt=".2f", linewidths=0.5,
                    xticklabels=["Easy", "Medium", "Hard"],
                    yticklabels=TOPICS, cbar_kws={"label": "greedy action share"})
        ax.set_title(title)
        ax.set_xlabel("difficulty")
    axes[0].set_ylabel("topic")
    fig.suptitle("Where the greedy policies spend their 240 recommendations "
                 "(12 topics x 3 difficulties, 3 seeds averaged)", y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# 8. calibration - frustration scatter
# --------------------------------------------------------------------- #
def fig_calibration_frustration() -> plt.Figure:
    cells = _cells()
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in LABELS:
        sub = cells[cells.strategy == label]
        if sub.empty:
            continue
        ax.scatter(sub["calibration"], sub["frustration_end"], s=26, alpha=0.45,
                   color=STRATEGY_COLORS[label], edgecolor="none", label=label)
        ax.scatter(sub["calibration"].mean(), sub["frustration_end"].mean(),
                   marker="*", s=340, color=STRATEGY_COLORS[label],
                   edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_xlabel("calibration (share of appropriately challenging problems)")
    ax.set_ylabel("frustration at the end of the session")
    ax.set_title("Calibration vs frustration — dots: 30 cells, stars: strategy means")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_ylim(bottom=0)
    return fig


# --------------------------------------------------------------------- #
# 9. significance heatmap
# --------------------------------------------------------------------- #
def _pairwise_p(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    from evaluation.compare_models import paired_test

    n = len(LABELS)
    p = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = LABELS[i], LABELS[j]
            if cells[cells.strategy == a].empty or cells[cells.strategy == b].empty:
                continue
            res = paired_test(cells, metric, a, b)
            p[j, i] = res["p_paired_t"]
    return pd.DataFrame(p, index=LABELS, columns=LABELS)


def fig_significance_heatmap() -> plt.Figure:
    cells = _cells()
    p_skill = _pairwise_p(cells, "skill_gain")
    p_reward = _pairwise_p(cells, "total_reward")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, pmat, title in (
        (axes[0], p_skill, "Skill gain"),
        (axes[1], p_reward, "Total reward"),
    ):
        with np.errstate(divide="ignore"):
            score = -np.log10(np.clip(pmat.to_numpy(dtype=float), 1e-12, 1.0))
        mask = np.triu(np.ones_like(score, dtype=bool))   # lower triangle only
        sns.heatmap(score, ax=ax, cmap="magma", mask=mask,
                    annot=pmat.to_numpy(), fmt=".1e",
                    linewidths=0.5, vmin=0, vmax=6,
                    xticklabels=LABELS, yticklabels=LABELS,
                    cbar_kws={"label": "$-\\log_{10} p$ (paired t-test)"})
        ax.set_title(f"{title}: paired t-test, 30 matched cells\n"
                     "(annotated = raw p; lower triangle only)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    fig.suptitle("Statistical significance of every pairwise comparison", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# 10. per-topic mastery-gain heatmaps
# --------------------------------------------------------------------- #
def fig_mastery_gain_heatmaps() -> plt.Figure:
    per_run = pd.read_csv(EVAL_DIR / "per_run.csv")
    init_cols = [f"init_{t}" for t in TOPICS]
    final_cols = [f"final_{t}" for t in TOPICS]

    gain = pd.DataFrame(
        (per_run[final_cols].to_numpy() - per_run[init_cols].to_numpy()),
        columns=TOPICS,
    )
    gain.insert(0, "strategy", per_run["strategy"].to_numpy())
    mat = gain.groupby("strategy")[TOPICS].mean().reindex(LABELS)
    rel = mat.sub(mat.mean(axis=1), axis=0)          # topic profile within strategy

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    sns.heatmap(mat, ax=axes[0], cmap="YlGnBu", annot=True, fmt=".3f",
                annot_kws={"fontsize": 7},
                linewidths=0.4, yticklabels=LABELS, xticklabels=TOPICS,
                cbar_kws={"label": "mean mastery gain"})
    axes[0].set_title("(a) absolute per-topic mastery gain")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha="right")

    sns.heatmap(rel, ax=axes[1], cmap="RdBu_r", center=0, annot=True, fmt=".3f",
                annot_kws={"fontsize": 7},
                linewidths=0.4, yticklabels=LABELS, xticklabels=TOPICS,
                cbar_kws={"label": "gain $-$ row mean"})
    axes[1].set_title("(b) topic profile (deviation from each strategy's mean)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=40, ha="right")
    axes[0].set_ylabel("strategy")
    fig.suptitle("Per-topic mastery gain on the held-out test students", y=1.03)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# registry / driver
# --------------------------------------------------------------------- #
FIGURES: dict[int, tuple[str, Callable[[], plt.Figure]]] = {
    1: ("01_learning_curves", fig_learning_curves),
    2: ("02_skill_gain_bars", fig_skill_gain_bars),
    3: ("03_reward_bars", fig_reward_bars),
    4: ("04_metrics_panel_2x2", fig_metrics_panel),
    5: ("05_skill_gain_boxplots", fig_skill_gain_boxplots),
    6: ("06_reward_sensitivity", fig_reward_sensitivity),
    7: ("07_action_heatmaps_dqn_vs_qlearning", fig_action_heatmaps),
    8: ("08_calibration_frustration_scatter", fig_calibration_frustration),
    9: ("09_significance_heatmap", fig_significance_heatmap),
    10: ("10_mastery_gain_heatmaps", fig_mastery_gain_heatmaps),
}


def render(which: list[int] | None = None, out_dir: Path = FIGURES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx in (which or sorted(FIGURES)):
        name, fn = FIGURES[idx]
        fig = fn()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"  [{idx:>2}/10] {path.name}")
    return written


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Render the report figures")
    p.add_argument("figures", nargs="*", type=int, default=None,
                   help="figure numbers (default: all ten)")
    p.add_argument("--out", default=str(FIGURES_DIR))
    args = p.parse_args()
    written = render(args.figures or None, Path(args.out))
    print(f"\n{len(written)} figure(s) -> {Path(args.out)}")


if __name__ == "__main__":
    main()
