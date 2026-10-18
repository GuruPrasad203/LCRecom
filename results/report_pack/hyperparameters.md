# Hyper-parameters (source of truth: `config.py`)

## Tabular agents (SARSA, Q-Learning)

| parameter | value |
|---|---|
| alpha | 0.18 |
| gamma | 0.95 |
| epsilon_start | 1 |
| epsilon_end | 0.05 |
| epsilon_decay_episodes | 1200 |

## Deep agents (DQN, Double DQN)

| parameter | value |
|---|---|
| hidden1 | 128 |
| hidden2 | 128 |
| lr | 0.001 |
| gamma | 0.95 |
| batch_size | 64 |
| buffer_size | 50000 |
| warmup_steps | 1000 |
| train_every | 4 |
| target_update_every | 250 |
| epsilon_start | 1 |
| epsilon_end | 0.05 |
| epsilon_decay_steps | 30000 |
| max_grad_norm | 10 |

## Reward weights (`environment/reward.py::RewardConfig`)

| parameter | value |
|---|---|
| skill_gain_weight | 40 |
| success_bonus | 0.8 |
| challenge_appropriate | 0.5 |
| challenge_mismatch | -0.6 |
| hint_penalty | 0.18 |
| time_penalty | 0.8 |
| time_tolerance | 1.15 |
| frustration_penalty | 0.45 |
| repetition_topic_penalty | 0.2 |
| repetition_problem_penalty | 0.35 |
| repetition_streak_penalty | 0.3 |
| clip_min | -2 |
| clip_max | 2 |

## Experiment geometry

* `EPISODE_LEN = 20` · `TRAIN_EPISODES = 2000` · `N_TRAIN_SEEDS = 5` (seeds [0, 1, 2, 3, 4])
* `EVAL_SEEDS = [100, 200, 300]` · `N_EVAL_STUDENTS = 10`
* `SENSITIVITY_EPISODES = 600` · `READOUT_N_STUDENTS = 6` · `READOUT_EVAL_SEEDS = [100, 200]`
* `GLOBAL_SEED = 42`, `DATA_SEED = 42`

## OFAT sensitivity grids (every grid contains its config.py default)

| agent | factor | grid | default | n configs |
|---|---|---|---|---|
| dqn | buffer_size | 10000, 50000, 100000 | 50000 | 3 |
| dqn | hidden | 64, 128, 256 | 128 | 3 |
| dqn | target_update_every | 100, 250, 500 | 250 | 3 |
| q_learning | alpha | 0.05, 0.1, 0.18, 0.3, 0.5 | 0.18 | 5 |
| q_learning | epsilon_decay_episodes | 600, 1200, 2400 | 1200 | 3 |
| q_learning | gamma | 0.8, 0.9, 0.95, 0.99 | 0.95 | 4 |

Fixed during the sensitivity study:

* **q_learning**: epsilon_start=1.0, epsilon_end=0.05 fixed; only the listed factor varies
* **dqn**: lr=1e-3, batch_size=64, gamma=0.95, warmup=1000, train_every=4 FIXED

Protocol: train split, [0, 1] training seeds x 600 episodes, 34 unique trainings, **364.7 s total** (per-run seconds in `sensitivity_runtime.json`).

## Study protocols

| study | agents | episodes | seeds | readout cells | runtime |
|---|---|---|---|---|---|
| main comparison | 4 RL + 3 baselines | 2000 (RL) | 5 train x 3 eval | 30 matched test cells | 21 s (evaluate.py wall-clock) |
| hyper-parameter sensitivity (OFAT) | Q-Learning, DQN | 600 | 2 train x 2 eval | 24 cells/config | 364.7 s |
| reward-weight sensitivity (earlier phase) | Q-Learning | 600 | 2 x 2 | 24 cells/variant | not instrumented (no runtime artifact) |
| ablations | Q-Learning, DQN (+full refs) | 2000 | 5 train x 2 eval | 60 cells/variant (train split) | 587.9 s (driver wall-clock) |
| overestimation | DQN, Double DQN snapshots | — | 1 pair/seed (3 seeds) x 32 samples | 200 pairs | 4.4 s |

**PPO** (`agents/ppo_agent.py`, `training/train_ppo.py`) is implemented and has 3-seed checkpoints (`models/PPO_seed*.pt`) from the earlier run; it is excluded from the main comparison (see REPORT_DATA.md, mandated note (b)).
