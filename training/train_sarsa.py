"""Train the tabular SARSA agent (on-policy TD control).

Usage:
    python training/train_sarsa.py [--episodes 2000] [--seeds 0 1 2]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.sarsa_agent import SarsaAgent  # noqa: E402
from config import TabularConfig  # noqa: E402
from training.common import make_env, training_arg_parser  # noqa: E402
from training.trainer import train  # noqa: E402


def main() -> None:
    args = training_arg_parser("Train SARSA").parse_args()
    cfg = TabularConfig()
    for seed in args.seeds:
        env = make_env()
        agent = SarsaAgent(config=cfg, seed=seed)
        train(
            agent,
            env,
            episodes=args.episodes,
            seed=seed,
            name="sarsa",
            verbose=not args.quiet,
        )
    print("SARSA training complete -> results/training/sarsa_seed*.csv, models/sarsa_seed*.npz")


if __name__ == "__main__":
    main()
