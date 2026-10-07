"""Aggregate the evaluation runs into the final comparison table.

Aggregation protocol
--------------------
Every strategy is summarised over the same 30 *cells* = 10 held-out students
x 3 evaluation seeds; RL strategies are first averaged over their 3 training
seeds inside each cell.  All paired significance tests therefore compare
30 matched observations per strategy.

Research questions answered here:

    RQ1  RL vs non-RL baselines (skill gain)
    RQ2  Q-Learning vs SARSA (on-policy vs off-policy)
    RQ3  DQN vs tabular Q-Learning (function approximation)
    RQ4  Double DQN vs DQN (overestimation correction)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR  # noqa: E402
from evaluation.metrics import HIGHER_IS_BETTER, LOWER_IS_BETTER, SUMMARY_METRICS  # noqa: E402
from evaluation.strategies import STRATEGY_LABELS, STRATEGY_ORDER, STRATEGY_TYPE  # noqa: E402

EVAL_DIR = RESULTS_DIR / "evaluation"
TABLE_COLUMNS = {
    "skill_gain": "Skill Gain",
    "completion_rate": "Completion",
    "avg_hints": "Hints",
    "mean_solve_time": "Time",
    "calibration": "Calibration",
    "unproductive_repeat_rate": "Unprod. Repeat",
    "frustration_end": "Frustration",
    "total_reward": "Reward",
}

# ------------------------------------------------------------------ helpers
def load_runs(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or EVAL_DIR / "per_run.csv")


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (strategy, student, eval_seed): mean over training seeds."""
    metrics = list(TABLE_COLUMNS) + ["repetition_rate", "too_easy_rate",
                                     "too_hard_rate", "median_solve_time",
                                     "unique_topics", "skill_gain_total"]
    return (
        df.groupby(["strategy", "student_id", "eval_seed"])[metrics]
        .mean()
        .reset_index()
    )


def summary_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in STRATEGY_ORDER:
        label = STRATEGY_LABELS[name]
        sub = cells[cells.strategy == label]
        row = {"Strategy": label, "Type": STRATEGY_TYPE[name], "n": len(sub)}
        for metric, pretty in TABLE_COLUMNS.items():
            row[pretty] = f"{sub[metric].mean():.4f} ± {sub[metric].std(ddof=1):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def raw_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in STRATEGY_ORDER:
        label = STRATEGY_LABELS[name]
        sub = cells[cells.strategy == label]
        row = {"strategy": label, "type": STRATEGY_TYPE[name], "n": len(sub)}
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------------------------- statistics
def paired_test(cells: pd.DataFrame, metric: str, a: str, b: str) -> dict:
    """Two-sided paired t-test + Wilcoxon signed-rank on the 30 cells."""
    from scipy import stats

    a = STRATEGY_LABELS.get(a, a)
    b = STRATEGY_LABELS.get(b, b)
    piv = cells.pivot_table(
        index=["student_id", "eval_seed"], columns="strategy", values=metric
    )
    x, y = piv[a].to_numpy(), piv[b].to_numpy()
    t, p_t = stats.ttest_rel(x, y)
    try:
        w, p_w = stats.wilcoxon(x, y)
    except ValueError:  # pragma: no cover - all differences zero
        w, p_w = np.nan, np.nan
    diff = x - y
    return {
        "metric": metric,
        "A": a,
        "B": b,
        "mean_A": float(x.mean()),
        "mean_B": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "t_stat": float(t),
        "p_paired_t": float(p_t),
        "wilcoxon_W": float(w) if w == w else None,
        "p_wilcoxon": float(p_w) if p_w == p_w else None,
        "A_better": bool(diff.mean() > 0) if metric in HIGHER_IS_BETTER
        else bool(diff.mean() < 0),
    }


def best_strategy(
    cells: pd.DataFrame, metric: str, candidates: list[str]
) -> str:
    """Highest cell-mean of ``metric`` over the 30 matched cells."""
    means = {
        name: float(cells.loc[cells.strategy == STRATEGY_LABELS[name], metric].mean())
        for name in candidates
    }
    return max(means, key=means.get)


def answer_research_questions(
    cells: pd.DataFrame,
    tests: list[dict],
    best_pair: tuple[str, str] | None = None,
) -> dict:
    lookup = {(t["metric"], t["A"], t["B"]): t for t in tests}
    L = STRATEGY_LABELS

    def rq(metric, a, b):
        t = lookup[(metric, L[a], L[b])]
        return {
            "a": L[a],
            "b": L[b],
            "mean_a": round(t["mean_A"], 4),
            "mean_b": round(t["mean_B"], 4),
            "diff": round(t["mean_diff"], 4),
            "p_t": t["p_paired_t"],
            "p_wilcoxon": t["p_wilcoxon"],
            "significant_05": bool(min(t["p_paired_t"], t["p_wilcoxon"] or 1.0) < 0.05),
        }

    if best_pair is None:
        best_pair = ("q_learning", "prerequisite")
    best_rl, best_baseline = best_pair
    answers = {
        "RQ1": {
            "question": "Does RL-based recommendation improve student skill more "
                        "than difficulty-based recommendation?",
            "vs_difficulty": rq("skill_gain", "q_learning", "difficulty"),
            "vs_prerequisite": rq("skill_gain", "q_learning", "prerequisite"),
            "vs_random": rq("skill_gain", "q_learning", "random"),
            "best_rl_vs_best_baseline": rq("skill_gain", best_rl, best_baseline),
            "best_pair_selection": {
                "rule": "highest mean skill_gain over the 30 matched cells",
                "best_rl": L[best_rl],
                "best_baseline": L[best_baseline],
            },
        },
        "RQ2": {
            "question": "Does Q-Learning outperform SARSA for adaptive "
                        "coding-practice recommendation?",
            "skill_gain": rq("skill_gain", "q_learning", "sarsa"),
            "reward": rq("total_reward", "q_learning", "sarsa"),
            "calibration": rq("calibration", "q_learning", "sarsa"),
        },
        "RQ3": {
            "question": "Does DQN outperform tabular Q-Learning when the state "
                        "space becomes larger (continuous 26-dim state)?",
            "skill_gain": rq("skill_gain", "dqn", "q_learning"),
            "calibration": rq("calibration", "dqn", "q_learning"),
            "reward": rq("total_reward", "dqn", "q_learning"),
        },
        "RQ4": {
            "question": "Can the RL agent maintain an appropriate challenge "
                        "level while reducing frustration? (and: does Double DQN "
                        "improve over DQN?)",
            "calibration_vs_prereq": rq("calibration", "dqn", "prerequisite"),
            "frustration_vs_random": rq("frustration_end", "dqn", "random"),
            "skill_gain_double_vs_single": rq("skill_gain", "double_dqn", "dqn"),
            "reward_double_vs_single": rq("total_reward", "double_dqn", "dqn"),
        },
    }
    return answers


# ------------------------------------------------------------------- main
def main() -> None:
    df = load_runs()
    cells = cell_means(df)

    table = summary_table(cells)
    raw = raw_summary(cells)

    table_csv = EVAL_DIR / "comparison_table.csv"
    table_md = EVAL_DIR / "comparison_table.md"
    raw.to_csv(EVAL_DIR / "summary_by_strategy.csv", index=False)
    cells.to_csv(EVAL_DIR / "cell_means.csv", index=False)
    table.to_csv(table_csv, index=False)
    table.to_markdown(table_md, index=False)

    # the headline comparison: best RL strategy vs best baseline, chosen on
    # the same 30 cells by mean skill gain (a read-out, not a tuning step)
    rl_names = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"]
    base_names = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] == "Baseline"]
    best_rl = best_strategy(cells, "skill_gain", rl_names)
    best_baseline = best_strategy(cells, "skill_gain", base_names)

    # paired tests (all pairs that answer an RQ + skill-gain leaderboard)
    pairs = [
        ("skill_gain", "q_learning", "difficulty"),
        ("skill_gain", "q_learning", "prerequisite"),
        ("skill_gain", "q_learning", "random"),
        ("skill_gain", "dqn", "prerequisite"),
        ("skill_gain", "q_learning", "sarsa"),
        ("skill_gain", "dqn", "q_learning"),
        ("skill_gain", "double_dqn", "dqn"),
        ("total_reward", "q_learning", "sarsa"),
        ("total_reward", "dqn", "q_learning"),
        ("total_reward", "double_dqn", "dqn"),
        ("calibration", "q_learning", "sarsa"),
        ("calibration", "dqn", "q_learning"),
        ("calibration", "dqn", "prerequisite"),
        ("completion_rate", "dqn", "random"),
        ("avg_hints", "dqn", "difficulty"),
        ("frustration_end", "dqn", "random"),
        ("unproductive_repeat_rate", "dqn", "q_learning"),
        ("skill_gain", best_rl, best_baseline),
    ]
    # de-duplicate while keeping the declared order
    seen: set[tuple[str, str, str]] = set()
    pairs = [p for p in pairs if not (p in seen or seen.add(p))]
    tests = [paired_test(cells, m, a, b) for m, a, b in pairs]
    pd.DataFrame(tests).to_csv(EVAL_DIR / "significance_tests.csv", index=False)

    answers = answer_research_questions(
        cells, tests, best_pair=(best_rl, best_baseline)
    )
    (EVAL_DIR / "research_questions.json").write_text(json.dumps(answers, indent=2))

    print(table.to_string(index=False))
    print(f"\n  best RL: {STRATEGY_LABELS[best_rl]}   "
          f"best baseline: {STRATEGY_LABELS[best_baseline]}")
    print(f"\n-> {table_csv}")
    print(f"-> {table_md}")
    print(f"-> {EVAL_DIR / 'significance_tests.csv'}")
    print(f"-> {EVAL_DIR / 'research_questions.json'}")


if __name__ == "__main__":
    main()
