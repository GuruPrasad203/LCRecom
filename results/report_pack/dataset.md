# Dataset

All data is synthetic and generated from a fixed seed (`DATA_SEED=42`); the frozen copies are the three CSVs listed below (locked in `results/LOCK.json`).

## Files (frozen in `data/`)

| file | rows | columns | content |
|---|---|---|---|
| students.csv | 28 | 22 | learner profiles: ability, learning rate, frustration/hint/time parameters, initial mastery of all 12 topics |
| problems.csv | 180 | 6 | 12 topics x 3 difficulties x 5 problems per (topic, difficulty) cell = 180 problems, with prerequisite_topics and expected_time |
| interactions.csv | 1120 | 12 | historical practice log (success, solve_time, hints, error_type, skill_gain) |

## Splits

| split | n | id range |
|---|---|---|
| test | 10 | S019–S028 |
| train | 18 | S001–S018 |

Main comparison = **test split** (10 students x 3 evaluation seeds = 30 matched cells; RL strategies averaged over their 5 training seeds inside each cell).  Sensitivity / ablation / reward-weight readouts use the **train split** (first 6 train students x 2 eval seeds) — stated per study.

## Action space

* 12 topics x 3 difficulties = **36 discrete actions** (one action = recommend one topic at one difficulty).
* Episode = **20 consecutive recommendations** (`EPISODE_LEN=20`), session statistics over the last `HISTORY_WINDOW=10` attempts, error distribution over the last `N_ERROR_WINDOW=12`, at most `MAX_HINTS=3` hints per problem.

## Topic prerequisite DAG

Used only by (i) the dataset generator / simulator, (ii) the Prerequisite-Based baseline, and (iii) the diagnostic `prereq_violation_rate` readout. **No RL agent observes the DAG.**

| topic | prerequisites |
|---|---|
| Arrays | — |
| Strings | Arrays |
| Linked Lists | Arrays |
| Stack | Arrays |
| Queue | Arrays |
| Recursion | Arrays |
| Trees | Recursion |
| Graphs | Stack, Queue, Trees |
| Dynamic Programming | Recursion |
| Greedy | Sorting |
| Binary Search | Arrays, Sorting |
| Sorting | Arrays |

## State vector (26 dims — deep agents; tabular agents use the coarsened `state_to_index`)

| idx | feature | range |
|---|---|---|
| 0–11 | mastery/<topic> (12 topics) | [0,1] |
| 12 | recent_accuracy | [0,1] |
| 13 | avg_solve_time_norm | [0,1] |
| 14 | recent_hint_usage | [0,1] |
| 15 | recent_error_rate | [0,1] |
| 16 | current_difficulty_norm | [0,1] |
| 17 | consecutive_failures_norm | [0,1] |
| 18 | frustration | [0,1] |
| 19 | same_topic_streak_norm | [0,1] |
| 20–25 | error/<type> (6 types) | [0,1] |

## Reward (per step, clipped)

`r = clip( 40 * skill_gain + 0.8*success + challenge (+0.5 appropriate / −0.6 mismatch) − 0.18*hints − time_penalty (0.8 if solve_time > 1.15×expected, else 0) − 0.45*frustration − repetition penalties (topic 0.20 / problem 0.35 / streak 0.30), −2 … +2 )`

Weights are the `RewardConfig` defaults in `environment/reward.py`; their sensitivity is studied in `reward_sensitivity*` artifacts.
