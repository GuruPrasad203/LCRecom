"""Train the PPO agent with agentic RL features.

Usage:
    python training/train_ppo.py [--episodes 2000] [--seeds 0 1 2]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent  # noqa: E402
from training.common import make_env, training_arg_parser  # noqa: E402
from training.trainer import train  # noqa: E402


def main() -> None:
    args = training_arg_parser("Train PPO agent with agentic RL").parse_args()
    for seed in args.seeds:
        env = make_env()
        agent = PPOAgent(seed=seed)
        train(
            agent,
            env,
            episodes=args.episodes,
            seed=seed,
            name="PPO",
            verbose=not args.quiet,
        )
    print("PPO training complete -> results/training/ppo_seed*.csv, models/ppo_seed*.pt")


if __name__ == "__main__":
    main()