"""Streamlit frontend for Adaptive Coding-Practice Recommendation project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in st.session_state.get("_sys_path", []):
    import sys
    sys.path.insert(0, str(ROOT))

from config import (
    N_ACTIONS,
    N_TOPICS,
    READOUT_EVAL_SEEDS,
    READOUT_N_STUDENTS,
    RESULTS_DIR,
    TOPICS,
    TRAIN_SEEDS,
)  # noqa: E402
from evaluation.compare_models import best_strategy, cell_means, load_runs
from evaluation.strategies import STRATEGY_LABELS, STRATEGY_ORDER, STRATEGY_TYPE
from visualization.plots import FIGURES, render


st.set_page_config(page_title="LCRecom — Adaptive Coding RL", layout="wide")

st.title("🎯 LCRecom — Adaptive Coding Recommendation")
st.caption("Reinforcement Learning for adaptive coding practice recommendations")


def load_results():
    """Load all result CSVs."""
    eval_dir = RESULTS_DIR / "evaluation"
    train_dir = RESULTS_DIR / "training"

    data = {}

    data["summary"] = pd.read_csv(eval_dir / "summary_by_strategy.csv")
    data["cell_means"] = pd.read_csv(eval_dir / "cell_means.csv")
    data["per_run"] = pd.read_csv(eval_dir / "per_run.csv")

    se_path = eval_dir / "reward_sensitivity.csv"
    data["reward_sensitivity"] = pd.read_csv(se_path) if se_path.exists() else pd.DataFrame()

    # Training curves
    training_curves = {}
    for name in ["dqn", "double_dqn", "sarsa", "q_learning"]:
        runs = []
        for seed in TRAIN_SEEDS:
            p = train_dir / f"{name}_seed{seed}.csv"
            if p.exists():
                s = pd.read_csv(p)["reward"]
                runs.append(s)
        if runs:
            training_curves[name] = pd.concat(runs, axis=1)
    data["training_curves"] = training_curves

    return data


@st.cache_data
def get_data():
    return load_results()


def render_figures_tab(data):
    st.subheader("Report Figures")
    col1, col2 = st.columns(2)
    fig_names = [
        ("01", "Learning Curves"),
        ("02", "Skill Gain Bars"),
        ("03", "Reward Bars"),
        ("04", "Metrics Panel 2x2"),
        ("05", "Skill Gain Boxplots"),
        ("06", "Reward Sensitivity"),
        ("07", "DQN vs Q-Learning Heatmaps"),
        ("08", "Calibration vs Frustration"),
        ("09", "Significance Heatmap"),
        ("10", "Per-Topic Mastery Gain"),
    ]

    for idx, (code, label) in enumerate(fig_names):
        fig_path = Path(f"visualization/figures/{code}_*.png")
        fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
        paths = sorted(fig_dir.glob(f"{code}_*.png"))
        if paths:
            with col1 if idx % 2 == 0 else col2:
                st.image(str(paths[0]), caption=label, use_container_width=True)


def render_metrics_tab(data):
    st.subheader("Evaluation Metrics")

    tab1, tab2, tab3 = st.tabs(["Summary", "Cell Means", "Per-Run"])

    with tab1:
        st.dataframe(data["summary"], use_container_width=True)

    with tab2:
        st.dataframe(data["cell_means"], use_container_width=True)

    with tab3:
        strategy = st.selectbox("Strategy", STRATEGY_ORDER, key="per_run_strategy")
        filtered = data["per_run"][data["per_run"]["strategy"] == strategy]
        st.dataframe(filtered, use_container_width=True)


def render_demo_tab(data):
    st.subheader("Demo / Trace")

    from evaluation.strategies import STRATEGY_ORDER, STRATEGY_TYPE
    strategy_options = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"]
    strategy_name = st.selectbox(
        "Strategy",
        strategy_options,
        key="demo_strategy",
    )
    student_id = st.selectbox(
        "Student",
        sorted(data["per_run"]["student_id"].unique()) if "student_id" in data["per_run"].columns else [],
        key="demo_student",
    )
    eval_seed = st.selectbox("Eval Seed", [100, 200, 300], key="demo_seed")

    if st.button("Run Demo Trace") and strategy_name and student_id:
        with st.spinner("Running demo..."):
            import sys
            from pathlib import Path as _Path

            root = _Path(__file__).resolve().parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

            from config import EPISODE_LEN, EVAL_SEEDS
            from data.generate_data import history_lookup, load_problems, load_students
            from environment.coding_env import CodingPracticeEnv
            from evaluation.compare_models import best_strategy, cell_means, load_runs
            from evaluation.strategies import (
                STRATEGY_LABELS,
                STRATEGY_ORDER,
                STRATEGY_TYPE,
                build_strategy,
            )
            from evaluation import best_pair

            problems = load_problems()
            students = load_students(split="test")
            by_id = {s.student_id: s for s in students}

            sid = student_id
            if sid not in by_id:
                st.error(f"Student {sid} not found in test split")
                return

            student = by_id[sid]
            s_idx = [s.student_id for s in students].index(student.student_id)
            env_seed = int(eval_seed * 100_003 + s_idx)
            history = history_lookup(problems).get(student.student_id, [])

            env = CodingPracticeEnv(problems=problems, students=students, mode="eval")

            rec = build_strategy(strategy_name, model_seed=0, random_seed=0)

            from environment.state import (
                IDX_CONSEC_FAIL,
                IDX_DIFFICULTY,
                IDX_FRUSTRATION,
                IDX_RECENT_ACCURACY,
                IDX_STREAK,
                MASTERY_SLICE,
            )

            DIFF_NAMES = {0: "Easy", 1: "Medium", 2: "Hard"}

            def _state_line(state, topic_idx):
                return (
                    f"acc={float(state[IDX_RECENT_ACCURACY]):.2f} "
                    f"frust={float(state[IDX_FRUSTRATION]):.2f} "
                    f"last_diff={DIFF_NAMES[int(round(float(state[IDX_DIFFICULTY]) * 2))]} "
                    f"consec_fail={int(round(float(state[IDX_CONSEC_FAIL]) * 5))} "
                    f"streak={int(round(float(state[IDX_STREAK]) * 5))} "
                    f"mastery={float(state[MASTERY_SLICE][topic_idx]):.3f}"
                )

            def _reward_line(comps):
                keys = [
                    ("gain", "skill_gain"),
                    ("success", "success"),
                    ("challenge", "challenge"),
                    ("hint", "hint"),
                    ("time", "time"),
                    ("frust", "frustration"),
                    ("rep_topic", "repetition_topic"),
                    ("rep_problem", "repetition_problem"),
                    ("streak", "repetition_streak"),
                ]
                body = "  ".join(f"{lbl}={comps[k]:+.2f}" for lbl, k in keys)
                return f"{body}  | raw={comps['raw_total']:+.2f} clipped={comps['clipped']:+.2f}"

            st.write(f"## {STRATEGY_LABELS[strategy_name]} — {student_id}")
            st.write(f"Env seed: {env_seed}, History length: {len(history)}")

            total = 0.0
            summary_fields = {
                "skill_gain_total": 0.0,
                "skill_gain_mean": 0.0,
                "completion_rate": 0.0,
                "calibration": 0.0,
                "unique_topics": 0,
                "frustration_end": 0.0,
            }
            initial_mastery = env.mastery.copy()

            for step in range(EPISODE_LEN):
                action = rec.recommend(state)
                topic_idx, difficulty = action // 3, action % 3 + 1
                state_snapshot = _state_line(state, topic_idx)
                next_state, reward, terminated, truncated, si = env.step(action)
                total += reward

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Step {step+1}**")
                    st.markdown(f"- **State**: {state_snapshot}")
                    st.markdown(f"- **Action**: [{action:>2}] ({si['action_label']})")
                    st.markdown(f"- **Problem**: #{si['problem_id']} \"{si['problem_title']}\" "
                              f"({si['expected_time']:.0f} min, mastery {si['mastery_before']:.3f}, challenge {si['challenge']})")

                with col2:
                    outcome = "PASS" if si["success"] else "FAIL"
                    err = si["error_type"] or "-"
                    st.markdown(f"- **Outcome**: {outcome} in {si['solve_time']:.1f} min "
                              f"({si['solve_time'] / max(si['expected_time'], 1e-6):.2f}x)")
                    st.markdown(f"- **Hints**: {si['hints_used']}, **Wrong**: {si['wrong_submissions']}")
                    st.markdown(f"- **Error**: {err}")
                    st.markdown(f"- **Reward**: {reward:+.2f}")
                    st.markdown(f"- **Components**: {_reward_line(si['reward_components'])}")

                state = next_state
                if terminated or truncated:
                    break

            final_mastery = env.mastery.copy()
            diff = final_mastery - initial_mastery

            summary_fields["skill_gain_total"] = float(diff.sum())
            summary_fields["skill_gain_mean"] = float(diff.mean())
            unique_topics = len({h["topic"] for h in env.history})
            summary_fields["unique_topics"] = unique_topics
            summary_fields["completion_rate"] = float(
                sum(bool(h["success"]) for h in env.history) / max(len(env.history), 1)
            )
            summary_fields["calibration"] = float(
                sum(h["challenge"] == "appropriate" for h in env.history)
                / max(len(env.history), 1)
            )
            summary_fields["frustration_end"] = float(env.frustration)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reward", f"{total:+.2f}")
                st.metric("Skill Gain (total)", f"{summary_fields['skill_gain_total']:.4f}")
            with col2:
                st.metric("Completion Rate", f"{summary_fields['completion_rate']:.2f}")
                st.metric("Unique Topics", summary_fields["unique_topics"])
            with col3:
                st.metric("Calibration", f"{summary_fields['calibration']:.2f}")
                st.metric("Frustration End", f"{summary_fields['frustration_end']:.2f}")

            st.markdown("### Mastery Progression")
            fig, ax = plt.subplots(figsize=(8, 4))
            episodes = list(range(1, len(env.history) + 1))
            ax.plot(episodes, env.mastery, "b-", marker="o", label="Mastery")
            ax.axhline(initial_mastery[0], color="gray", linestyle="--", label="Initial")
            ax.set_xlabel("Recommendation Step")
            ax.set_ylabel("Mastery")
            ax.set_title(f"Mastery Progression — {student_id} ({STRATEGY_LABELS[strategy_name]})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()


def main():
    data = get_data()

    menu = st.sidebar.radio("Navigation", [
        "Report Figures",
        "Evaluation Metrics",
        "Demo / Trace",
        "Training Curves",
    ])

    if menu == "Report Figures":
        render_figures_tab(data)

    elif menu == "Evaluation Metrics":
        render_metrics_tab(data)

    elif menu == "Demo / Trace":
        render_demo_tab(data)

    elif menu == "Training Curves":
        st.subheader("Training Curves")
        strategy = st.selectbox("Strategy", ["DQN", "Double DQN", "SARSA", "Q-Learning"])
        if strategy in data["training_curves"]:
            fig, ax = plt.subplots(figsize=(9, 5))
            df = data["training_curves"][strategy]
            # Plot mean ± sd across seeds
            means = df.mean(axis=1)
            stds = df.std(axis=1)
            x = df.index
            ax.plot(x, means, lw=2.0, label=strategy)
            ax.fill_between(x, means - stds, means + stds, alpha=0.15, linewidth=0)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Reward (100-episode moving average)")
            ax.set_title(f"Learning Curves: {strategy}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()


if __name__ == "__main__":
    main()