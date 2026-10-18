# Architecture — data pipeline

```mermaid
flowchart TD
    A["data/ — synthetic students (28), problems (180), interactions (1120)<br/>seed 42, split train 18 / test 10"] --> B["environment/CodingPracticeEnv<br/>36 actions, 20-step episodes, 26-dim state, clipped shaped reward"]
    B --> C["training/trainer.py<br/>SARSA / Q-Learning / DQN / Double DQN<br/>2000 episodes x seeds 0-4"]
    C --> D["models/ — 5 checkpoints per agent + PPO (3-seed, excluded)"]
    D --> E["evaluation/lock.py — freeze data + models + config + design (74 files)"]
    E --> F["evaluation/evaluate.py — 690 runs<br/>10 test students x 3 eval seeds x (5 model seeds RL / 1 baseline)"]
    F --> G["evaluation/compare_models.py<br/>30 matched cells; paired t + Wilcoxon; Welch t + Cohen's d"]
    C --> H["training/hyperparam_sensitivity.py — OFAT, 17 configs"]
    C --> I["training/ablation.py — reward-unshaped + state-ablated"]
    C --> J["evaluation/overestimation.py — frozen net snapshots"]
    F --> K["visualization/plots.py — 13 figures, 300 dpi"]
    G --> K
    H --> K
    I --> L["ablation_*.png"]
    J --> M["overestimation.png"]
    K --> N["report/build_report_pack.py"]
    L --> N
    M --> N
    N --> O["results/report_pack/ + results/report_pack.zip"]
    ```

Supporting pieces:

* `evaluation/audit_dqn.py` — post-training sanity checks (checkpoint identity, action-distribution verdicts, deterministic step traces): **25 passed / 0 failed**.
* `evaluation/metrics.py` — one `RunMetrics` per run; `prereq_violation_rate` reads the mastery snapshot taken before each `env.step`.
* `training/reward_sensitivity.py` — earlier reward-weight study (artifacts `reward_sensitivity*.csv`).
* `tests/` — 80 tests, including the lock trip-wire (`tests/test_lock.py`) and per-figure render tests.
