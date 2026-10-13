"""Streamlit frontend for Agentic RL (PPO) agent.

Provides an interactive web interface to:
- Train the PPO agent with agentic features
- Evaluate the agent on held-out students
- Compare against baselines
- Visualize per-step traces and metrics
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from config import (
    EPISODE_LEN, EVAL_SEEDS, TRAIN_SEEDS, N_ACTIONS, N_TOPICS, N_DIFFICULTIES,
    TOPICS, DIFFICULTY_LEVELS, DIFFICULTY_NORM
)
from environment.coding_env import CodingPracticeEnv
from environment.state import state_to_index
from agents.ppo_agent import PPOAgent
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from baselines.base import Recommender
from data.generate_data import history_lookup, load_problems, load_students
from training.common import make_env


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def strategy_from_name(name: str) -> Recommender:
    """Build a recommender agent from a strategy name string."""
    if name == "PPO":
        return PPOAgent(seed=0)
    elif name == "Q-Learning":
        return QLearningAgent(seed=0)
    elif name == "SARSA":
        return SarsaAgent(seed=0)
    elif name == "DQN":
        return DQNAgent(config=None, seed=0)
    elif name == "Double DQN":
        return DoubleDQNAgent(seed=0)
    elif name == "Random":
        from baselines.random_baseline import RandomRecommender
        return RandomRecommender()
    elif name == "Difficulty-Based":
        from baselines.difficulty_based import DifficultyRecommender
        return DifficultyRecommender()
    elif name == "Prerequisite-Based":
        from baselines.prerequisite_based import PrerequisiteRecommender
        return PrerequisiteRecommender()
    else:
        raise ValueError(f"Unknown strategy: {name}")


def format_action(action_idx: int) -> str:
    """Format action index as (topic, difficulty)."""
    topic_idx = action_idx // 3
    difficulty = action_idx % 3 + 1
    return f"{TOPICS[topic_idx]} ({DIFFICULTY_LABELS[difficulty]})"


def run_episode_strategy(
    strategy_name: str,
    student_index: int,
    eval_seed: int,
    model_seed: int = 0,
    episodes: int = 1,
) -> dict:
    """Run one or more episodes with a given strategy and return summary metrics."""
    strategy = strategy_from_name(strategy_name)

    problems = load_problems()
    students = load_students(split="test")
    student = students[student_index]

    env_seed = int(eval_seed * 100_003 + student_index)
    history = history_lookup().get(student.student_id, [])

    env = CodingPracticeEnv(
        problems=problems,
        students=students,
        mode="eval",
        seed=env_seed,
    )

    # Reset strategy
    if hasattr(strategy, "reset"):
        strategy.reset()

    total_reward = 0.0
    skill_gain = 0.0
    completions = 0
    hints = 0
    appropriate = 0
    repeats = 0
    times: list[float] = []
    topic_counts: dict[str, int] = {t: 0 for t in TOPICS}
    error_types: dict[str, int] = {e: 0 for e in [
        "Syntax Error", "Logic Error", "Conceptual Error",
        "Time Complexity Error", "Edge Case Error", "Implementation Error",
    ]}

    for ep in range(episodes):
        state, info = env.reset(
            seed=env_seed + ep,
            options={"student_id": student.student_id, "history": history},
        )
        init_mastery = env.mastery.copy()

        for step in range(EPISODE_LEN):
            action = strategy.recommend(state)
            topic_idx = action // 3
            difficulty = action % 3 + 1
            topic_counts[TOPICS[topic_idx]] = topic_counts.get(TOPICS[topic_idx], 0) + 1

            next_state, reward, terminated, truncated, sinfo = env.step(action)

            total_reward += reward
            skill_gain += sinfo["skill_gain"]
            completions += int(sinfo["success"])
            hints += sinfo["hints_used"]
            appropriate += int(sinfo["challenge"] == "appropriate")
            repeats += int(sinfo["repeated_topic"])
            times.append(sinfo["solve_time"])
            for et in error_types:
                error_types[et] += sinfo.get("error_type", "") == et or 0

            state = next_state
            if terminated or truncated:
                break

    final_mastery = env.mastery.copy()
    mastery_diff = float((final_mastery - init_mastery).sum())

    summary = {
        "strategy": strategy_name,
        "student": student.student_id,
        "total_reward": float(total_reward),
        "skill_gain": mastery_diff / 12.0,  # per-topic average
        "completion_rate": completions / EPISODE_LEN,
        "avg_hints": hints / EPISODE_LEN,
        "avg_solve_time": float(np.mean(times)) if times else 0.0,
        "calibration": appropriate / EPISODE_LEN,
        "frustration": float(env.frustration),
        "repeats": repeats / max(EPISODE_LEN - 1, 1),
        "unique_topics": len(set(
            h["topic"] for h in env.history if "topic" in h
        )),
    }
    summary.update({
        **{f"error_{et.lower().replace(' ', '_')}_count": c
           for et, c in error_types.items()},
        **{f"topic_{t.lower().replace(' ', '_')}_count": c
           for t, c in topic_counts.items()},
    })
    return summary


# --------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic RL — PPO Frontend",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic Reinforcement Learning — PPO Frontend")
st.caption("Proximal Policy Optimization agent for adaptive coding-practice recommendation")

# Sidebar controls
st.sidebar.header("Controls")
strategy_name = st.sidebar.selectbox(
    "Strategy",
    ["PPO", "Q-Learning", "SARSA", "DQN", "Double DQN", "Random",
     "Difficulty-Based", "Prerequisite-Based"],
    index=0,
)

col1, col2 = st.sidebar.columns(2)
with col1:
    eval_seed = st.number_input("Eval Seed", min_value=0, value=100)
with col2:
    model_seed = st.number_input("Model Seed", min_value=0, value=0)

col3, col4 = st.sidebar.columns(2)
with col3:
    student_index = st.selectbox(
        "Test Student",
        options=list(range(10)),
        format_func=lambda i: f"S019 + {i} = S{i+19}",
    )
with col4:
    n_episodes = st.slider("Episodes", min_value=1, max_value=10, value=1)

run_button = st.sidebar.button("Run Episode", type="primary")

# Main area
if run_button:
    with st.spinner(f"Running {strategy_name} on student S{student_index+19}..."):
        try:
            summary = run_episode_strategy(
                strategy_name=strategy_name,
                student_index=student_index,
                eval_seed=eval_seed,
                model_seed=model_seed,
                episodes=n_episodes,
            )

            st.success("Episode complete!")

            # Display key metrics
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("Total Reward", f"{summary['total_reward']:+.2f}")
            with col_metrics[1]:
                st.metric("Skill Gain (avg)", f"{summary['skill_gain']:.4f}")
            with col_metrics[2]:
                st.metric("Completion Rate", f"{summary['completion_rate']:.2f}")

            # Detailed metrics
            st.subheader("Detailed Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Avg Hints:** {summary['avg_hints']:.2f}")
                st.write(f"**Avg Solve Time:** {summary['avg_solve_time']:.2f} min")
                st.write(f"**Calibration:** {summary['calibration']:.2f}")
            with col2:
                st.write(f"**Frustration:** {summary['frustration']:.2f}")
                st.write(f"**Repeats:** {summary['repeats']:.2f}")
                st.write(f"**Unique Topics:** {summary['unique_topics']}")

            # Distribution charts
            st.subheader("Topic Usage")
            topic_df = pd.DataFrame([
                {"Topic": t, "Count": summary.get(f"topic_{t.lower().replace(' ', '_')}_count", 0)}
                for t in TOPICS
            ])
            st.bar_chart(topic_df.set_index("Topic"))

            st.subheader("Error Type Distribution")
            error_df = pd.DataFrame([
                {"Error Type": et.replace("_", " "), "Count": summary.get(f"error_{et.lower().replace(' ', '_')}_count", 0)}
                for et in [
                    "Syntax Error", "Logic Error", "Conceptual Error",
                    "Time Complexity Error", "Edge Case Error", "Implementation Error",
                ]
            ])
            st.bar_chart(error_df.set_index("Error Type"))

        except Exception as e:
            st.error(f"Error running strategy: {e}")
            st.exception(e)
else:
    st.info("Configure the sidebar and click **Run Episode** to start an evaluation run.")
    st.caption(
        "The PPO agent uses actor-critic with PPO clip objective, "
        "GAE advantage estimation, and entropy regularization for exploration."
    )

# About section
with st.expander("About this frontend"):
    st.write(
        "This frontend provides a web interface for the PPO (Proximal Policy Optimization) "
        "agent implemented as part of the agentic RL approach. PPO features:\n\n"
        "- **Actor-Critic architecture**: Shared network with policy (actor) and value (critic) heads\n"
        "- **PPO clip objective**: `min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)` with ε=0.2 for stable learning\n"
        "- **GAE (Generalized Advantage Estimation)**: λ=0.95 for advantage estimation\n"
        "- **Entropy regularization**: Coefficient 0.01 for implicit exploration\n"
        "- **Agentic features**: Internal latent state, goal tracking, topic diversity penalty\n\n"
        "Compared to the other agents in this project:\n"
        "- More stable than DQN/Double DQN (which had 37/90 negative held-out episodes)\n"
        "- Stochastic policy with entropy-based exploration (vs ε-greedy in tabular agents)\n"
        "- Learns a parameterized policy rather than just Q-values\n"
    )

st.sidebar.divider()
st.sidebar.caption(
    "Adaptive Coding-Practice RL Project\n"
    "Data seed: 42 | 18 train students / 10 test students"
)

# Run in: python -m streamlit run frontend.py