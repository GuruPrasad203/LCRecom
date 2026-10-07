"""Train the Double DQN agent (overestimation-corrected deep Q-learning).

Usage:
    python training/train_double_dqn.py [--episodes 2000] [--seeds 0 1 2]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.double_dqn_agent import DoubleDQNAgent  # noqa: E402
from config import DQNConfig  # noqa: E402
from training.common import make_env, training_arg_parser  # noqa: E402
from training.trainer import train  # noqa: E402


def main() -> None:
    args = training_arg_parser("Train Double DQN").parse_args()
    cfg = DQNConfig()
    for seed in args.seeds:
        env = make_env()
        agent = DoubleDQNAgent(config=cfg, seed=seed)
        train(
            agent,
            env,
            episodes=args.episodes,
            seed=seed,
            name="double_dqn",
            verbose=not args.quiet,
        )
    print("Double DQN training complete -> results/training/double_dqn_seed*.csv")


if __name__ == "__main__":
    main()
