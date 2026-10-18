# README — repro & pack contents

## Exact commands (run from the repo root, in order)

```bash
# 0. environment (CPU-only is fine)
#    pip install -r requirements.txt   (numpy pandas scipy torch matplotlib pytest)

# 1. archive the original 3-seed run, then retrain at 5 seeds
#    (results_archive_3seed/ holds the frozen v1 artifacts)
python training/train_sarsa.py
python training/train_qlearning.py
python training/train_dqn.py
python training/train_double_dqn.py
python training/train_ppo.py            # exists; excluded from the main table

# 2. post-training sanity + freeze everything
python evaluation/audit_dqn.py           # 25 passed / 0 failed
python evaluation/lock.py                # freezes data+models+config+design (74 files)

# 3. test-split evaluation + statistics
python evaluation/evaluate.py            # 690 runs -> evaluation/per_run.csv
python evaluation/compare_models.py      # tables + paired t/Wilcoxon + Welch/Cohen d + stats_tests.md

# 4. studies
python training/hyperparam_sensitivity.py  # OFAT (Q-Learning, DQN); ~365 s
python training/ablation.py                # 2 ablations x 5 seeds; ~588 s
python evaluation/overestimation.py        # 200 pairs; ~4 s

# 5. figures + tests
python visualization/plots.py            # 13 figures, 300 dpi
python -m pytest tests -q                # 80 passed

# 6. this pack
python report/build_report_pack.py       # results/report_pack/ + results/report_pack.zip
```

Notes:

* Do **not** regenerate `data/` — it is frozen by `results/LOCK.json` (`data/generate_data.py` exists and is seed-deterministic, DATA_SEED=42, but regeneration would risk float/platform drift against the lock).
* The lock trip-wire test (`tests/test_lock.py`) fails if any locked file changes; re-lock only with a recorded reason (the CP-7 re-lock happened *before* the 5-seed evaluation).
* The earlier 3-seed artifacts are in `results_archive_3seed/` (commit `336b7dd`); the archived run reproduced byte-identically over a 50-episode gate before the 5-seed retrain.

## Pack contents

| file | what |
|---|---|
| `REPORT_DATA.md` | the report draft data: protocol, all tables, RQ answers, mandated notes (a)–(d), caveats |
| `metrics_main.{csv,md}` | every main metric mean ± sd + definitions + convergence/training time |
| `stats_tests.md`, `significance_tests.csv`, `welch_vs_best_baseline.csv`, `research_questions.json` | significance statistics |
| `comparison_table.{csv,md}` | formatted headline table |
| `evaluation/per_run.csv` (690 rows), `evaluation/cell_means.csv`, `evaluation/summary_by_strategy.csv` | raw → aggregated evidence |
| `sensitivity.csv`, `sensitivity_summary.csv`, `sensitivity_paired.csv`, `sensitivity_runtime.json` | OFAT hyper-parameter study |
| `reward_sensitivity*.csv` | earlier reward-weight study |
| `ablation.csv`, `ablation_summary.csv`, `ablation_stats.csv`, `ablation.md` | ablation study |
| `overestimation.csv`, `overestimation_summary.csv`, `overestimation_runtime.json` | overestimation study |
| `audit/audit_report.json` | 25-check training-health audit |
| `training/*.csv`, `training/*.json` | per-run training curves + selection/timing records |
| `figures/*.png` | all 16 figures at 300 dpi |
| `dataset/` | `students.csv`, `problems.csv`, `interactions.csv`, `prerequisite_table.csv` |
| `system_info.md`, `dataset.md`, `hyperparameters.md` | machine, data & configuration cards |
| `code_snippets.md` | verbatim source excerpts + pseudocode |
| `architecture.md` | Mermaid pipeline diagram |
| `README_REPRO.md` | this file |

## Integrity

The zip was created *after* this pack, so it contains every pack file plus the full `results/` tree (training curves, tensorboard event files, LOCK.json) — everything under `results/` at build time, nothing estimated or hand-edited.
