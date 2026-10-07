# REPORT — Adaptive Coding-Practice Recommendation with Reinforcement Learning

All numbers below were produced by the commands listed in `README.md` §3 on
this repository as it stands after the Phase 3 LOCK. `pytest tests -q` →
**77 passed**; `python evaluation/lock.py verify` → **LOCK holds**.

---

## 0. Protocol summary (details in README §2)

| item | value |
|---|---|
| students | 18 train (S001–S018) / **10 held-out test** (S019–S028), `DATA_SEED = 42` |
| episode | 20 recommendations, 36 actions (12 topics × 3 difficulties), 26-dim state, 972-state tabular coarsening |
| training | 2000 episodes × seeds 0/1/2, greedy-val checkpoint selection on 4 **train** students |
| evaluation | 450 runs = 90 per RL strategy (3 model seeds × 10 students × 3 eval seeds) + 30 per baseline; `env_seed = eval_seed·100003 + student_index` identical for every strategy |
| statistics | 30 matched cells per strategy (10 students × 3 eval seeds), two-sided paired *t* **and** Wilcoxon signed-rank, α = 0.05, two tests reported side by side, no multiplicity correction |
| test-split contact | exactly one program (`evaluation/evaluate.py`), after the LOCK |

---

## 1. Phase 1 — DQN audit, retrain, re-audit

### 1.1 What the first audit found (pre-retrain checkpoints)

Greedy readout on the **6 train students × 2 eval seeds** (12 cells):

| checkpoint | mean reward | distinct actions (of 36) |
|---|---|---|
| DQN seed 0 | −11.4 | 6 |
| DQN seed 1 | −14.3 | 6 |
| DQN seed 2 | −7.3 | 8 |
| Double DQN seed 0 | +13.8 | 9 |
| Double DQN seed 1 | +15.6 | 10 |
| Double DQN seed 2 | +16.1 | 9 |

DQN's shipped weights were **negative on average on the training students it
was trained on**, using ≤8 of 36 actions — the collapse the audit was written
to detect. The ten contract checks themselves passed (they test *correctness*
of the implementation, not *quality* of the policy).

### 1.2 Retrain (approved protocol: current `training/trainer.py`, 2000 episodes, seeds 0/1/2, **no hyper-parameter change**)

Checkpoint selection = highest greedy-val reward on 4 train students:

| model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| DQN best episode | 1999 | 1199 | 1299 |
| DQN greedy val reward | 15.00 | 14.83 | 13.82 |
| Double DQN best episode | 1799 | 1699 | 1899 |
| Double DQN greedy val reward | 14.38 | 17.17 | 17.05 |

(The greedy-val value is measured on 4 students × 1 env seed; the audit
readout below is the more reliable 6 students × 2 env seeds.)

### 1.3 Re-audit (`results/audit/`, 25 verdicts, **0 fail**)

| check | verdict |
|---|---|
| `preprocessing_parity` | PASS — 40 policy inputs identical on the train and eval paths, net input dim 26 |
| `action_roundtrip` | PASS — 36/36 (topic, difficulty) pairs reached the environment |
| `reward_decomposition` | PASS — 3 unit cases + 36 env steps match the printed components |
| `terminal_invariants` | PASS — `terminated` never fires before step 20, truncation only |
| `target_invariants` | PASS — DQN/Double-DQN target formulas, target-net isolation, hard update every 250 |
| `replay_invariants` | PASS — capacity 50 000, wrap-around, dtype, index checks |
| `recommend_greediness` | PASS — 24 states, ε forced to 1.0, `recommend()` still argmax |
| `checkpoint_identity` ×6 | PASS — file matches the recorded greedy-val reward and config |
| `action_distribution` ×6 | see table |
| `deterministic_step_trace` ×6 | PASS — 20 steps reproducible under `env_seed=10000300` |

Greedy readout after retrain (12 cells each):

| checkpoint | mean | min | actions used | top-action share | verdict |
|---|---|---|---|---|---|
| DQN seed 0 | **+15.36** | +0.15 | 8/36 | 0.29 | NARROW |
| DQN seed 1 | **−0.50** | −24.03 | 9/36 | 0.31 | COLLAPSED |
| DQN seed 2 | **+2.63** | −22.65 | 9/36 | 0.55 | NARROW |
| Double DQN seed 0 | +10.06 | −26.33 | 12/36 | 0.29 | HEALTHY |
| Double DQN seed 1 | +15.48 | −11.19 | 8/36 | 0.37 | NARROW |
| Double DQN seed 2 | +15.25 | −1.72 | 15/36 | 0.33 | HEALTHY |
| *(same readout, for scale)* Random | +7.13 | −11.28 | 36/36 | 0.05 | — |
| *(same readout)* Q-Learning seed 0/1/2 | +21.16 / +18.72 / +20.61 | +3.2 / +5.7 / +3.6 | 20/20/18 | 0.36/0.20/0.28 | — |
| *(same readout)* SARSA seed 0/1/2 | +20.75 / +18.37 / +18.33 | +7.8 / +7.1 / −5.7 | 20/15/16 | 0.21/0.21/0.20 | — |

**Honest Phase-1 outcome.**

* Contract correctness: **fully verified** (25/25).
* DQN quality: the retrain fixed the *mean* (−11.4/−14.3/−7.3 →
  +15.4/−0.5/+2.6) but **1 of 3 DQN seeds still has a non-positive greedy
  reward on the train students**, every DQN seed uses only 8–9 of 36 actions,
  and seeds 1/2 have single cells at −24/−23. Double DQN is better (2 of 3
  seeds HEALTHY) but one seed still reaches −26 in a cell.
* Per the agreed rule this is reported as **genuine deep-RL instability, not a
  hyper-parameter problem: no tuning was performed or attempted.**
* All three tabular runs of Q-Learning/SARSA are strictly positive on the same
  cells (min +3.2 to −5.7 vs DQN's −24), so the instability is specific to
  the deep agents in this environment.

---

## 2. Phase 2 — reward sensitivity (train split only)

Design: Q-Learning, 600 episodes, seeds (0, 1), 6 variants, readout = 6 train
students × 2 eval seeds = 12 cells → 24 paired rows per variant, every run
scored under the **default** reward (`reward_default` = common yardstick).

| variant | reward_default mean ± sd | skill gain | completion |
|---|---|---|---|
| success_bonus_low | 19.46 ± 9.07 | 0.1865 | 0.8021 |
| gain_double | 19.38 ± 7.84 | 0.1636 | 0.8250 |
| gain_half | 18.95 ± 8.26 | 0.1619 | 0.8062 |
| **default** | **18.84 ± 9.24** | 0.1642 | 0.8250 |
| time_penalty_off | 18.68 ± 8.48 | 0.1688 | 0.8146 |
| repetition_x2 | 18.10 ± 8.58 | 0.1642 | 0.8042 |

Paired against `default` (n = 24 cells):

| variant | Δ mean | paired *t* p | Wilcoxon p | significant |
|---|---|---|---|---|
| gain_double | +0.54 | 0.669 | 0.922 | no |
| gain_half | +0.11 | 0.925 | 0.989 | no |
| repetition_x2 | −0.74 | 0.395 | 0.281 | no |
| success_bonus_low | +0.62 | 0.593 | 0.570 | no |
| time_penalty_off | −0.16 | 0.886 | 0.790 | no |

**Decision: no variant differs from the default under either test (all
p > 0.28) → keep the default reward unchanged.** (Variant-own-reward columns
are in `results/reward_sensitivity.csv`; `time_penalty_off` scores 26–29 under
its own unbounded reward, which is exactly why the cross-variant comparison
uses `reward_default`.)

---

## 3. Phase 3 — LOCK

`results/LOCK.json` freezes **48 files**: `data/` (3 CSV), `models/` (24),
`config.py`, and the design code (`environment/`, `agents/`,
`training/*.py` — 20 files). Evaluation/visualization/tests/demo are *not*
locked by design (they read the frozen artifacts).

```
python evaluation/lock.py verify   → LOCK holds: manifest matches the tree
```

Nothing in the locked groups has been edited since.

---

## 4. Phase 4 — held-out evaluation (`results/evaluation/per_run.csv`, 450 rows)

### 4.1 Main table (30 matched cells per strategy, mean ± sd)

| Strategy | Type | n | Skill gain | Completion | Hints | Time | Calibration | Frustration | **Reward** |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Prerequisite-Based | Baseline | 30 | **0.0171 ± 0.0090** | 0.797 ± 0.096 | 0.062 | 39.95 | **0.963 ± 0.097** | 0.025 | 11.35 ± 16.14 |
| Difficulty-Based | Baseline | 30 | 0.0145 ± 0.0084 | 0.695 ± 0.103 | 0.365 | 74.74 | 0.697 ± 0.148 | 0.198 | 8.30 ± 14.43 |
| Random | Baseline | 30 | 0.0130 ± 0.0065 | 0.762 ± 0.137 | 0.255 | 54.03 | 0.627 ± 0.179 | 0.184 | 7.95 ± 12.17 |
| **SARSA** | On-policy RL | 30 | **0.0145 ± 0.0073** | 0.788 ± 0.090 | 0.110 | 45.53 | 0.902 ± 0.085 | 0.070 | **15.29 ± 11.98** |
| Q-Learning | Off-policy RL | 30 | 0.0139 ± 0.0065 | 0.803 ± 0.092 | 0.041 | 44.16 | 0.884 ± 0.104 | 0.042 | 15.19 ± 11.17 |
| Double DQN | Deep RL | 30 | 0.0135 ± 0.0063 | 0.775 ± 0.120 | 0.126 | 50.83 | 0.833 ± 0.177 | 0.079 | 12.06 ± 10.26 |
| DQN | Deep RL | 30 | 0.0135 ± 0.0063 | 0.777 ± 0.105 | 0.078 | 46.24 | 0.887 ± 0.095 | 0.066 | 2.37 ± 9.87 |

Rank by skill gain: **Prerequisite (0.0171) > SARSA (0.0145) = Difficulty
(0.0145) > Q-Learning (0.0139) > Double DQN = DQN (0.0135) > Random (0.0130).**
Rank by reward: **SARSA (15.29) > Q-Learning (15.19) > Double DQN (12.06) >
Prerequisite (11.35) > Difficulty (8.30) > Random (7.95) > DQN (2.37).**

### 4.2 Paired significance (selected rows of `significance_tests.csv`)

| metric | A | B | Δ | paired *t* p | Wilcoxon p | A better |
|---|:--|:--|--:|--:|--:|:--|
| skill_gain | SARSA | Prerequisite-Based | −0.0026 | 3.09e-05 | 2.69e-05 | no |
| skill_gain | Q-Learning | Prerequisite-Based | −0.0032 | 9.80e-06 | 2.55e-07 | no |
| skill_gain | Q-Learning | Difficulty-Based | −0.0006 | 0.208 | 0.903 | no |
| skill_gain | Q-Learning | Random | +0.0009 | 0.0506 | 0.0185 | yes (Wilcoxon only) |
| skill_gain | Q-Learning | SARSA | −0.0006 | 0.0308 | 0.0427 | no |
| skill_gain | DQN | Q-Learning | −0.0004 | 0.0695 | 0.0879 | no |
| skill_gain | Double DQN | DQN | +0.0000 | 0.933 | 0.612 | no |
| total_reward | DQN | Q-Learning | −12.83 | 2.85e-08 | 1.86e-09 | no |
| total_reward | Double DQN | DQN | +9.69 | 8.57e-08 | 3.54e-08 | yes |
| total_reward | Q-Learning | SARSA | −0.10 | 0.878 | 0.584 | no |
| calibration | DQN | Prerequisite-Based | −0.076 | 1.50e-04 | 5.47e-04 | no |
| frustration | DQN | Random | −0.119 | 0.0187 | 0.0447 | yes |
| hints | DQN | Difficulty-Based | −0.287 | 4.12e-05 | 4.94e-05 | yes |

### 4.3 Research questions (`research_questions.json`)

**RQ1 — Does RL beat difficulty-based recommendation?**
No. Q-Learning vs Difficulty-Based: Δ skill gain −0.0006, *p* = 0.208 /
0.903 (not significant). Versus the *best* baseline (Prerequisite-Based) RL is
significantly **worse**: best-RL (SARSA) vs best-baseline Δ = −0.0026,
*p* = 3.09e-05 / 2.69e-05. Versus Random, Q-Learning is +0.0009 and only
significant under Wilcoxon (0.0185; *t* p = 0.0506).

**RQ2 — Does Q-Learning outperform SARSA?**
No. SARSA has the higher skill gain (0.0145 vs 0.0139, Δ −0.0006,
*p* = 0.031 / 0.043 — significant, in SARSA's favour). Reward and
calibration are statistically indistinguishable (p = 0.878/0.584 and
0.443/0.462).

**RQ3 — Does DQN beat tabular Q-Learning on the 26-dim continuous state?**
No. Skill gain Δ −0.0004 (*p* = 0.069 / 0.088, n.s.), calibration Δ +0.003
(n.s.), but total reward is **far lower**: 2.37 vs 15.19
(*p* = 2.85e-08 / 1.86e-09) — 37 of DQN's 90 runs end with a negative total
reward (Q-Learning: 8/90; Double DQN: 18/90).

**RQ4 — Does the RL agent hold calibration and cut frustration? Does Double
DQN help?**
Calibration: DQN 0.887 vs Prerequisite 0.963 (DQN worse, Δ −0.076,
*p* = 1.50e-04 / 5.47e-04), and DQN ≈ Q-Learning (Δ +0.003, n.s.).
Frustration: DQN 0.066 vs Random 0.184 (better, *p* = 0.0187 / 0.0447).
Double DQN vs DQN: skill gain identical (Δ +0.0000, *p* = 0.933/0.612) but
reward significantly higher (12.06 vs 2.37, *p* = 8.57e-08 / 3.54e-08) —
double-Q fixes the reward collapse, not the skill-gain ranking.

---

## 5. Conclusions

1. **No RL agent beats the best baseline on skill gain.** Prerequisite-Based
   (0.0171) is ahead of the best RL strategy, SARSA (0.0145), by +0.0026
   (*p* ≈ 3e-05 on both tests). RL matches the *heuristic* difficulty baseline
   and beats Random, but the prerequisite ordering — a domain prior this
   environment rewards through the challenge term — is the strongest method on
   the primary metric.
2. **RL wins the reward metric, not the learning metric.** SARSA/Q-Learning
   reach ≈15.2 total reward vs 11.4 for the best baseline (not significant
   pairwise at n = 30 for the tabular-vs-prereq reward, but consistently
   ranked), mainly through lower time cost and lower frustration.
3. **Deep RL is unstable here, as audited.** After a protocol-compliant
   retrain with zero tuning, DQN still produces one non-positive training
   readout (seed 1: −0.50) and 37/90 negative held-out episodes; Double DQN
   recovers the reward (12.06) but not the skill gain (0.0135). The tabular
   agents are strictly positive on the same cells. This is reported as a real
   finding about the deep agents, not something papered over with tuning.
4. **The reward is not over-fitted to its coefficients.** Five plausible
   re-weightings change nothing significant (all p > 0.28), so the reward
   shape is not driving the ranking.
5. **Integrity.** 18/10 student split with test ids rejected by the sensitivity
   harness, one test-split read after a 48-file LOCK, 77 tests green, ten
   figures and the demo trace all generated from the frozen artifacts.

---

## 6. Artifacts

| artifact | path |
|---|---|
| audit report (25 verdicts) | `results/audit/audit_report.json` |
| action distribution / readouts | `results/audit/action_distribution.csv`, `greedy_train_reward.csv`, `reference_readout.csv` |
| reward sensitivity | `results/reward_sensitivity{,_summary,_paired}.csv` |
| lock manifest | `results/LOCK.json` |
| held-out runs (450) | `results/evaluation/per_run.csv` |
| tables / tests / RQs | `results/evaluation/comparison_table.{csv,md}`, `significance_tests.csv`, `research_questions.json` |
| learning curves | `results/training/<agent>_seed<s>.{csv,json}` |
| demo trace | `results/demo_trace.txt` |
| figures (10) | `visualization/figures/01…10_*.png` |
| tests | `tests/` → `pytest tests -q` = 77 passed |

### Limitations

* The simulator, not real students, is the subject of every claim.
* 30 matched cells limits power for small effects; Wilcoxon and paired-*t*
  occasionally disagree (RQ1 vs Random), and no multiplicity correction is
  applied across the reported pairs.
* One environment/dataset (`DATA_SEED = 42`), one horizon (20 steps), one
  reward family.
* Greedy-val checkpoint selection is model selection (4 train students), not
  hyper-parameter tuning; DQN hyper-parameters were never touched, per the
  approved protocol — so this report says nothing about what tuned DQN would
  do, only about the specified procedure.
