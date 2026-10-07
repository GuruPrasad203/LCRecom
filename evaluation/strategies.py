"""Factory for the seven recommendation strategies under comparison.

| Strategy          | Type                      | Source                     |
|-------------------|---------------------------|----------------------------|
| Random            | baseline                  | uniform over 36 actions    |
| Difficulty-Based  | baseline                  | heuristic rules            |
| Prerequisite-Based| baseline                  | topic-DAG rules            |
| SARSA             | on-policy tabular RL      | models/sarsa_seed*.npz     |
| Q-Learning        | off-policy tabular RL     | models/q_learning_seed*.npz|
| DQN               | deep RL                   | models/dqn_seed*.pt        |
| Double DQN        | improved deep RL          | models/double_dqn_seed*.pt |
"""

from __future__ import annotations

from pathlib import Path

from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent
from baselines.base import Recommender
from baselines.difficulty_based import DifficultyRecommender
from baselines.prerequisite_based import PrerequisiteRecommender
from baselines.random_baseline import RandomRecommender
from config import MODELS_DIR, TabularConfig

STRATEGY_ORDER = [
    "random",
    "difficulty",
    "prerequisite",
    "sarsa",
    "q_learning",
    "dqn",
    "double_dqn",
]

STRATEGY_LABELS = {
    "random": "Random",
    "difficulty": "Difficulty-Based",
    "prerequisite": "Prerequisite-Based",
    "sarsa": "SARSA",
    "q_learning": "Q-Learning",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
}

STRATEGY_TYPE = {
    "random": "Baseline",
    "difficulty": "Baseline",
    "prerequisite": "Baseline",
    "sarsa": "On-policy RL",
    "q_learning": "Off-policy RL",
    "dqn": "Deep RL",
    "double_dqn": "Improved Deep RL",
}

# strategies that need a trained model (and therefore a model seed)
TRAINED = {"sarsa", "q_learning", "dqn", "double_dqn"}


def build_strategy(name: str, model_seed: int = 0, random_seed: int = 0) -> Recommender:
    if name == "random":
        return RandomRecommender(seed=random_seed)
    if name == "difficulty":
        return DifficultyRecommender()
    if name == "prerequisite":
        return PrerequisiteRecommender()
    if name == "sarsa":
        return SarsaAgent.load(MODELS_DIR / f"sarsa_seed{model_seed}.npz", config=TabularConfig())
    if name == "q_learning":
        return QLearningAgent.load(
            MODELS_DIR / f"q_learning_seed{model_seed}.npz", config=TabularConfig()
        )
    if name == "dqn":
        return DQNAgent.load(MODELS_DIR / f"dqn_seed{model_seed}.pt")
    if name == "double_dqn":
        return DoubleDQNAgent.load(MODELS_DIR / f"double_dqn_seed{model_seed}.pt")
    raise ValueError(f"unknown strategy: {name}")


def model_seeds_for(name: str, seeds: tuple[int, ...] | list[int]) -> list[int]:
    """Baselines have no trained parameters -> a single (dummy) seed."""
    return list(seeds) if name in TRAINED else [0]


def available_models(names: list[str] | None = None) -> dict[str, bool]:
    """Check which trained artifacts exist on disk."""
    out = {}
    for name in (names or STRATEGY_ORDER):
        if name not in TRAINED:
            out[name] = True
        else:
            ext = ".npz" if name in ("sarsa", "q_learning") else ".pt"
            out[name] = (MODELS_DIR / f"{name}_seed0{ext}").exists()
    return out
