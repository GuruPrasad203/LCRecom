"""Central configuration for the Adaptive Coding-Practice RL project.

Every module in the project imports its constants from here so that
training, evaluation and visualisation always agree on:

* the topic / difficulty vocabulary (i.e. the action space),
* episode geometry (length, sliding-window sizes),
* hyper-parameters of the tabular and deep RL agents,
* global random seeds (reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"

for _p in (DATA_DIR, RESULTS_DIR, MODELS_DIR, TENSORBOARD_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Problem vocabulary -> action space
# --------------------------------------------------------------------------
TOPICS: list[str] = [
    "Arrays",
    "Strings",
    "Linked Lists",
    "Stack",
    "Queue",
    "Recursion",
    "Trees",
    "Graphs",
    "Dynamic Programming",
    "Greedy",
    "Binary Search",
    "Sorting",
]

DIFFICULTY_LABELS: dict[int, str] = {1: "Easy", 2: "Medium", 3: "Hard"}
DIFFICULTY_LEVELS: tuple[int, ...] = (1, 2, 3)

# Normalised difficulty used inside the reward / challenge model.
# 1 -> 0.10, 2 -> 0.50, 3 -> 0.90   (mastery is on [0, 1])
DIFFICULTY_NORM: dict[int, float] = {1: 0.10, 2: 0.50, 3: 0.90}

N_TOPICS = len(TOPICS)
N_DIFFICULTIES = len(DIFFICULTY_LEVELS)
N_ACTIONS = N_TOPICS * N_DIFFICULTIES  # 12 x 3 = 36

TOPIC_TO_INDEX: dict[str, int] = {t: i for i, t in enumerate(TOPICS)}

# --------------------------------------------------------------------------
# Error vocabulary
# --------------------------------------------------------------------------
ERROR_TYPES: list[str] = [
    "Syntax Error",
    "Logic Error",
    "Conceptual Error",
    "Time Complexity Error",
    "Edge Case Error",
    "Implementation Error",
]
N_ERROR_TYPES = len(ERROR_TYPES)

# --------------------------------------------------------------------------
# Topic prerequisite DAG (used by the prerequisite-based baseline and
# by the synthetic dataset generator)
# --------------------------------------------------------------------------
TOPIC_PREREQS: dict[str, list[str]] = {
    "Arrays": [],
    "Strings": ["Arrays"],
    "Stack": ["Arrays"],
    "Queue": ["Arrays"],
    "Linked Lists": ["Arrays"],
    "Sorting": ["Arrays"],
    "Binary Search": ["Arrays", "Sorting"],
    "Recursion": ["Arrays"],
    "Trees": ["Recursion"],
    "Graphs": ["Stack", "Queue", "Trees"],
    "Dynamic Programming": ["Recursion"],
    "Greedy": ["Sorting"],
}

# --------------------------------------------------------------------------
# Episode geometry
# --------------------------------------------------------------------------
EPISODE_LEN = 20          # coding problems recommended per episode
HISTORY_WINDOW = 10       # sliding window for "recent" statistics
MAX_HINTS = 3             # hints a simulated student may request
N_ERROR_WINDOW = 12       # window used for the error-type distribution

# --------------------------------------------------------------------------
# Random seeds
# --------------------------------------------------------------------------
GLOBAL_SEED = 42
DATA_SEED = 42
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)
EVAL_SEEDS: tuple[int, ...] = (100, 200, 300)

# --------------------------------------------------------------------------
# Tabular hyper-parameters (SARSA / Q-Learning)
# --------------------------------------------------------------------------
@dataclass
class TabularConfig:
    alpha: float = 0.18            # learning rate
    gamma: float = 0.95            # discount factor
    epsilon_start: float = 1.0     # initial exploration rate
    epsilon_end: float = 0.05      # final exploration rate
    epsilon_decay_episodes: int = 1200  # linear decay over this many episodes


# --------------------------------------------------------------------------
# Deep RL hyper-parameters (DQN / Double DQN)
# --------------------------------------------------------------------------
@dataclass
class DQNConfig:
    hidden1: int = 128
    hidden2: int = 128
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 1_000
    train_every: int = 4
    target_update_every: int = 250   # hard target-network update interval
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000  # linear decay over this many env steps
    max_grad_norm: float = 10.0


# --------------------------------------------------------------------------
# Experiment geometry
# --------------------------------------------------------------------------
TRAIN_EPISODES = 2000       # episodes per RL run
N_TRAIN_SEEDS = 3           # independent training seeds (mean ± std)
N_EVAL_STUDENTS = 10        # held-out test students
SENSITIVITY_EPISODES = 600  # episodes for the reward-sensitivity study

# --------------------------------------------------------------------------
# Greedy readout geometry (TRAINING students only -- never the test split)
# Shared by evaluation/audit_dqn.py and training/reward_sensitivity.py so
# both studies score policies on *identical* (student, eval_seed) cells.
# --------------------------------------------------------------------------
READOUT_N_STUDENTS = 6          # first N students of the train split
READOUT_EVAL_SEEDS: tuple[int, ...] = (100, 200)


def ensure_dirs() -> None:
    """Re-create all output folders (idempotent)."""
    for p in (DATA_DIR, RESULTS_DIR, MODELS_DIR, TENSORBOARD_DIR):
        p.mkdir(parents=True, exist_ok=True)
