# Ablation study — reward shaping and state features

**Scope: Q-Learning and DQN only.** SARSA and Double DQN are deliberately
excluded from the ablation study: Q-Learning is the representative tabular
agent and DQN the representative deep agent. Running every ablation on every
algorithm would multiply CPU cost without changing the conclusions drawn from
these two canonical agents. The state ablation additionally applies to DQN
only (see below).

* Protocol: train split only, `EPISODE_LEN` = 20 steps, **5 seeds**
  (0–4), **2000 episodes** per run — identical to the main experiment.
* Readout: greedy rollouts on 6 fixed TRAINING students × 2 eval seeds
  = **60 cells** per (agent, variant). Reward is always scored under the
  default `RewardConfig` (common yardstick; the shaping only ever changes
  the number returned by `compute_reward`, never the session dynamics).
* The **full (unablated) reference** uses the very same readout, the same
  5 seeds and the CP-3 models of the main experiment — no re-training, so
  reference and ablations are directly comparable.
* Statistics: **Welch t-test** (independent samples, unequal variance) of
  each ablation vs full, plus Cohen's *d*.

## Variants

| variant | change vs `full` | agents |
|---|---|---|
| `full` | none — deployed configuration (reference) | Q-Learning, DQN |
| `reward_unshaped` | reward = `skill_gain_weight × skill_gain` only; success bonus, challenge, hint, time, frustration and repetition terms all set to 0. Clip (±2) kept. | Q-Learning, DQN |
| `state_ablated` | observation = first 20 dims of the 26-dim state (the 6 `error/<type>` distribution features removed; network input 26 → 20) | DQN only |

**Why no prerequisite-state ablation:** the agents never observe the
prerequisite DAG. It exists only inside (a) the success model that shapes
transitions and (b) the Prerequisite-Based baseline. There is no prerequisite
information in the observation to remove, so such an ablation is not
well-defined for the agents.

**Why `state_ablated` is DQN-only:** the tabular agents coarsen the 26-dim
observation with `state_to_index` into bins over mastery/accuracy/failures/
difficulty/weak-topic/hints/streak — the error-type features were never part
of that discretisation, so masking them cannot change the tabular state
index. There is nothing to ablate.

## Results

Source: `results/ablation.csv` (per-cell), `results/ablation_summary.csv`
(mean ± sd), `results/ablation_stats.csv` (Welch tests).

### Reward ablation (`reward_unshaped` vs `full`)

| agent | variant | n cells | reward mean ± sd | Welch *t* | *p* | Cohen's *d* |
|---|---|---|---|---|---|---|
| Q-Learning | full | 60 | 20.53 ± 8.80 | — | — | — |
| Q-Learning | reward_unshaped | 60 | 10.85 ± 10.05 | −5.61 | < 0.0001 | −1.03 |
| DQN | full | 60 | 6.35 ± 15.68 | — | — | — |
| DQN | reward_unshaped | 60 | −9.35 ± 13.38 | −5.90 | < 0.0001 | −1.08 |

Both agents degrade significantly when every shaping term is removed:
the shaped reward is doing real work, and neither the reward-free skill-gain
signal nor the exploration schedule can fully compensate.

### State ablation (`state_ablated` vs `full`)

| agent | variant | n cells | reward mean ± sd | Welch *t* | *p* | Cohen's *d* |
|---|---|---|---|---|---|---|
| DQN | full | 60 | 6.35 ± 15.68 | — | — | — |
| DQN | state_ablated | 60 | 14.31 ± 8.62 | +3.45 | 0.0009 | +0.63 |

Removing the 6 error-type distribution features **improves** the DQN's
greedy readout reward significantly. The plausible reading: with 12 000
training transitions, the 6 sparse, highly-correlated error features add
variance without adding decision-relevant information, and the 26 → 20 input
reduces overfitting. The ablated policy is also less variable across cells
(sd 8.6 vs 15.7).

## Figures

* `visualization/figures/ablation_reward_unshaped.png` — bar chart,
  full vs reward-unshaped, Q-Learning + DQN, mean ± sd.
* `visualization/figures/ablation_state_ablated.png` — bar chart,
  full vs state-ablated, DQN, mean ± sd.

## Reproduce

```bash
python training/ablation.py            # 2000 eps x 5 seeds, ~588 s
```
