"""Train the DQN agent (deep Q-learning on the continuous 26-dim state).

Usage:
    python training/train_dqn.py [--episodes 2000] [--seeds 0 1 2]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dqn_agent import DQNAgent  # noqa: E402
from config import DQNConfig  # noqa: E402
from training.common import make_env, training_arg_parser  # noqa: E402
from training.trainer import train  # noqa: E402


def main() -> None:
    args = training_arg_parser("Train DQN").parse_args()
    cfg = DQNConfig()
    for seed in args.seeds:
        env = make_env()
        agent = DQNAgent(config=cfg, seed=seed)
        train(
            agent,
            env,
            episodes=args.episodes,
            seed=seed,
            name="dqn",
            verbose=not args.quiet,
        )
    print("DQN training complete -> results/training/dqn_seed*.csv, models/dqn_seed*.pt")


if __name__ == "__main__":
    main()
