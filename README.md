# Adaptive Coding-Practice Recommendation with Reinforcement Learning

A reproducible study of *which recommendation strategy best improves a
simulated student's coding skills*: three hand-crafted baselines, two tabular
RL agents (SARSA, Q-Learning) and two deep RL agents (DQN, Double DQN), all
compared on identical students, seeds, horizon and metrics.

---

## 1. Repository layout

```
config.py                     single source of truth: vocabulary, geometry,
                              hyper-parameters, seeds, readout geometry
data/                         generated dataset (180 problems, 28 students,
                              1120 historical interactions) + generator
environment/                  gymnasium env, reward, state, action space,
                              student simulator
agents/                       SARSA, Q-Learning, DQN, Double DQN
baselines/                    random / difficulty-based / prerequisite-based
training/                     shared trainer, per-agent entry points,
                              reward-sensitivity study
evaluation/                   evaluate.py, compare_models.py, audit_dqn.py,
                              lock.py, metrics, strategies
visualization/plots.py        the ten report figures  -> visualization/figures/
tests/                        77 pytest checks (contract + leakage guards)
demo.py                       printable per-step trace of one test student
results/                      every artifact (see §7)
models/                       trained checkpoints (frozen by Phase 3 LOCK)
```

---

## 2. Exact protocol

### 2.1 Data and splits

| item | value |
|---|---|
| problems | 180 = 12 topics x 15 (5 per difficulty), every one of the 36 action categories populated |
| students | 28 → **18 train** (S001–S018) / **10 held-out test** (S019–S028) |
| history | 40 logged interactions per student (1120 rows) → seeds the 10-step sliding window at `reset()` |
| generator seed | `DATA_SEED = 42` (`python data/generate_data.py` reproduces the CSVs byte-for-byte) |

The test split is touched **exactly once**, by `evaluation/evaluate.py` in
Phase 4, after the LOCK.

### 2.2 State, action, geometry

| item | value |
|---|---|
| action | `(topic, difficulty)` category → `topic_idx * 3 + (difficulty - 1)` = **36 actions** |
| observation | **26-dim** float32 in [0,1]: 12 mastery + recent accuracy / avg time / hints / error rate / last difficulty / consecutive failures / frustration / topic streak + 6 error-type shares |
| tabular state | coarsening of the same vector → **972 states** (3·3·3·3·2·2·3) → Q-table 972 x 36 |
| episode | **20 recommendations** (fixed horizon, truncation; `terminated` is never set) |
| windows | `HISTORY_WINDOW = 10`, `N_ERROR_WINDOW = 12`, `MAX_HINTS = 3` |

### 2.3 Reward (frozen, `environment/reward.py`)

```
R = 40 * skill_gain + 0.80 * 1{solved} + 0.50 * 1{appropriate}
                 - 0.60 * 1{too_easy or too_hard}
                 - 0.18 * hints - 0.80 * max(0, time_ratio - 1.15)
                 - 0.45 * frustration - repetition terms (0.20 / 0.35 / 0.30)
     clipped to [-2, +2]
```

The decomposition, the clipping and its agreement with `env.step()` are
asserted by audit check 3 and `tests/test_env.py`.

### 2.4 Agents and hyper-parameters

| agent | hyper-parameters (all from `config.py`) |
|---|---|
| SARSA / Q-Learning | `alpha 0.18`, `gamma 0.95`, epsilon 1.0 → 0.05 linearly over 1200 episodes |
| DQN / Double DQN | MLP 26-128-128-36, Adam `lr 1e-3`, `gamma 0.95`, batch 64, buffer 50 000, warm-up 1000, train every 4 steps, hard target update every 250 steps, epsilon 1.0 → 0.05 over 30 000 env steps, Huber loss, grad-clip 10 |

Double DQN differs from DQN **only** in the target
(`Q_target(s', argmax_a Q_online(s', a))`); everything else is shared.

Baselines: `RandomRecommender` (uniform over 36), `DifficultyRecommender`
(heuristic rules), `PrerequisiteRecommender` (topic-DAG order).

### 2.5 Training and checkpoint selection

* `TRAIN_EPISODES = 2000`, `TRAIN_SEEDS = (0, 1, 2)` — three independent runs
  per agent, mean ± sd everywhere.
* Each episode samples a **train** student with `env_seed = seed * 100003 + ep`.
* **Selection rule (not a hyper-parameter):** every 100 episodes the trainer
  runs *deterministic greedy* rollouts on the **first 4 TRAINING students**
  (env seeds `seed * 7919 + 13 + i`) and keeps the checkpoint with the highest
  mean reward. Training reward itself is unusable for selection because it is
  measured under the ε-greedy behaviour policy, not the deployed greedy one.
  The selection record is written to `results/training/<name>_seed<s>.json`
  (`best_episode`, `best_greedy_val_reward`, `final_greedy_val_reward`).
* `train()` raises if its student pool contains no train-split student.
* Only the selected checkpoint is shipped as `models/<name>_seed<s>.{npz,pt}`;
  the final weights are kept alongside as `*_last_*` for inspection.

### 2.6 Evaluation protocol (Phase 4 — the only test-split contact)

```
3 model seeds (RL) x 10 test students x 3 eval seeds = 90 runs / RL strategy
1 x 10 x 3                                     = 30 runs / baseline
450 runs total, 20 steps each, identical workload for every strategy
env_seed = eval_seed * 100_003 + student_index   (identical across strategies)
eval seeds = (100, 200, 300)
```

Aggregation: RL runs are averaged over model seeds inside each of the
**30 matched cells** (10 students x 3 eval seeds); every significance test
compares 30 paired observations — two-sided paired *t* **and** Wilcoxon
signed-rank.

### 2.7 Reward-sensitivity design (Phase 2, train split only)

Q-Learning, `SENSITIVITY_EPISODES = 600`, seeds `(0, 1)`, six variants:

| variant | change |
|---|---|
| `default` | none — the deployed reward, untouched |
| `gain_half` | `skill_gain_weight 40 → 20` |
| `gain_double` | `skill_gain_weight 40 → 80` |
| `time_penalty_off` | `time_penalty 0.80 → 0` |
| `repetition_x2` | repetition penalties x2 (0.40 / 0.70 / 0.60) |
| `success_bonus_low` | `success_bonus 0.80 → 0.20` |

Readout: greedy rollouts on **6 fixed train students x 2 eval seeds = 12
cells** per (variant, seed), 24 rows per variant, each scored twice — under
the default reward (`reward_default`, the common yardstick) and under the
variant's own reward (`reward_variant`). Paired *t* + Wilcoxon against
`default` decide whether the reward should change.

### 2.8 Greedy readout geometry (shared, `config.py`)

`READOUT_N_STUDENTS = 6`, `READOUT_EVAL_SEEDS = (100, 200)`,
`env_seed = eval_seed * 100_003 + student_index` — used identically by the
DQN audit and the sensitivity study.

### 2.9 Leakage guards

1. `training.trainer.train()` raises `ValueError` when no train student is left.
2. `training.reward_sensitivity.resolve_students()` rejects any id outside the
   train split **before** an environment exists (tested with S019+).
3. The audit readouts assert their student ids ⊆ train ids (test).
4. `split="test"` is read only in `evaluation/evaluate.py` (Phase 4) and
   `demo.py` (read-only display of a Phase-4 result).
5. Phase 3 `results/LOCK.json` freezes data, models, `config.py` and the
   design code (environment/, agents/, training/) — `python evaluation/lock.py
   verify` re-hashes them and the suite fails if anything drifts.

### 2.10 Tests (77)

| file | what it gates |
|---|---|
| `tests/test_data.py` | dataset shape, ranges, reproducibility, action-space alignment |
| `tests/test_env.py` | spaces, determinism, challenge bands, reward decomposition/bounds |
| `tests/test_agents.py` | hand-calculated SARSA/Q-Learning updates, terminal targets, ε-decay, greedy `recommend()`, save/load |
| `tests/test_dqn_contract.py` | the 10 audit checks (preprocessing parity, action roundtrip, reward decomposition, terminal/target/replay invariants, `recommend()` greediness under a hostile rng, checkpoint identity, action-distribution table, deterministic step trace) + leakage guard |
| `tests/test_reward_sensitivity.py` | variant definitions, test-id rejection, harness behaviour |
| `tests/test_lock.py` | the LOCK manifest still matches the working tree |
| `tests/test_visualization.py` | all ten figures render from the frozen artifacts |

```bash
pytest tests -q          # 77 passed
```

---

## 3. Pipeline (phase order)

```bash
python -m pytest tests -q

# Phase 1 — DQN audit -> retrain -> re-audit
python evaluation/audit_dqn.py                 # 10 checks -> results/audit/
python training/train_dqn.py       --episodes 2000 --seeds 0 1 2
python training/train_double_dqn.py --episodes 2000 --seeds 0 1 2
python evaluation/audit_dqn.py                 # re-run, must end 0-fail

# Phase 2 — reward sensitivity (train split)
python training/reward_sensitivity.py          # -> results/reward_sensitivity*.csv

# Phase 3 — freeze everything
python evaluation/lock.py                      # -> results/LOCK.json
python evaluation/lock.py verify

# Phase 4 — the only test-split contact
python evaluation/evaluate.py                  # -> results/evaluation/per_run.csv

# Phase 5 — statistics
python evaluation/compare_models.py            # tables + paired tests + RQ1-RQ4

# Phase 6 — figures
python visualization/plots.py                  # -> visualization/figures/*.png

# Phase 7 — demo
python demo.py --out results/demo_trace.txt
```

---

## 4. Checkpoint selection vs. hyper-parameter tuning

Greedy-val selection on 4 training students is **model-checkpoint selection**
(picking which weights to ship). It is applied identically to every agent and
never reads the test split. Hyper-parameters are declared once in `config.py`
and were not changed after the Phase 3 LOCK.

---

## 5. Figures

| # | file | source |
|---|---|---|
| 1 | `01_learning_curves.png` | `results/training/*_seed*.csv` |
| 2 | `02_skill_gain_bars.png` | `results/evaluation/summary_by_strategy.csv` |
| 3 | `03_reward_bars.png` | same |
| 4 | `04_metrics_panel_2x2.png` | same |
| 5 | `05_skill_gain_boxplots.png` | `results/evaluation/cell_means.csv` |
| 6 | `06_reward_sensitivity.png` | `results/reward_sensitivity.csv` |
| 7 | `07_action_heatmaps_dqn_vs_qlearning.png` | greedy readout (§2.8) |
| 8 | `08_calibration_frustration_scatter.png` | `cell_means.csv` |
| 9 | `09_significance_heatmap.png` | paired t-tests on `per_run.csv` |
| 10 | `10_mastery_gain_heatmaps.png` | `per_run.csv` init/final mastery columns |

---

## 6. Requirements

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn,
                                    # torch, gymnasium, matplotlib, seaborn,
                                    # tensorboard, pytest
python data/generate_data.py        # only if data/ is missing (seed 42)
```

Python 3.11+ (developed on 3.14).

---

## 7. Artifacts

```
results/LOCK.json                       Phase 3 manifest (md5 per file)
results/audit/audit_report.json         10 audit checks (25 verdicts)
results/audit/action_distribution.csv   36-row action counts per deep checkpoint
results/audit/greedy_train_reward.csv   12 greedy cells per deep checkpoint
results/audit/reference_readout.csv     same readout for all 7 strategies
results/reward_sensitivity.csv          Phase 2 readout (72 rows)
results/reward_sensitivity_summary.csv  per-variant means
results/reward_sensitivity_paired.csv   paired t / Wilcoxon vs default
results/training/<agent>_seed<s>.csv    learning curves + selection JSONs
results/evaluation/per_run.csv          450 held-out runs
results/evaluation/cell_means.csv       30 matched cells per strategy
results/evaluation/comparison_table.*   mean ± sd table
results/evaluation/significance_tests.csv  paired t + Wilcoxon
results/evaluation/research_questions.json RQ1–RQ4 answers
results/demo_trace.txt                  Phase 7 per-step trace
visualization/figures/*.png             the ten figures
```

See **REPORT.md** for the results.
# LCRecom
