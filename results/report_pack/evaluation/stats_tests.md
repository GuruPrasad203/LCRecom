# Significance tests (test split, 30 matched cells)

Cells = 10 held-out students x 3 evaluation seeds; RL strategies averaged over 5 training seeds inside each cell.

## Paired t-test + Wilcoxon (matched cells)

| metric | A | B | mean A | mean B | diff | paired *t* | *p* | Wilcoxon *p* | better |
|---|---|---|---|---|---|---|---|---|---|
| skill_gain | Q-Learning | Difficulty-Based | 0.0137 | 0.0145 | -0.0008 | -1.831 | 0.07744 | 0.2801 | B |
| skill_gain | Q-Learning | Prerequisite-Based | 0.0137 | 0.0171 | -0.0034 | -5.571 | 5.196e-06 | 3.856e-07 | B |
| skill_gain | Q-Learning | Random | 0.0137 | 0.0130 | 0.0007 | 1.984 | 0.05681 | 0.02085 | A |
| skill_gain | DQN | Prerequisite-Based | 0.0128 | 0.0171 | -0.0043 | -5.993 | 1.622e-06 | 6.147e-08 | B |
| skill_gain | Q-Learning | SARSA | 0.0137 | 0.0142 | -0.0005 | -2.935 | 0.006466 | 0.009301 | B |
| skill_gain | DQN | Q-Learning | 0.0128 | 0.0137 | -0.0010 | -4.749 | 5.097e-05 | 1.824e-05 | B |
| skill_gain | Double DQN | DQN | 0.0138 | 0.0128 | 0.0011 | 3.821 | 0.0006498 | 0.001131 | A |
| total_reward | Q-Learning | SARSA | 14.8654 | 15.3460 | -0.4807 | -0.962 | 0.3439 | 0.3492 | B |
| total_reward | DQN | Q-Learning | 4.2246 | 14.8654 | -10.6407 | -6.714 | 2.296e-07 | 1.863e-09 | B |
| total_reward | Double DQN | DQN | 10.7981 | 4.2246 | 6.5735 | 5.094 | 1.956e-05 | 3.453e-05 | A |
| calibration | Q-Learning | SARSA | 0.8997 | 0.9047 | -0.0050 | -0.312 | 0.7574 | 0.6829 | B |
| calibration | DQN | Q-Learning | 0.8797 | 0.8997 | -0.0200 | -0.999 | 0.3263 | 0.5447 | B |
| calibration | DQN | Prerequisite-Based | 0.8797 | 0.9633 | -0.0837 | -5.172 | 1.574e-05 | 5.472e-05 | B |
| completion_rate | DQN | Random | 0.7837 | 0.7617 | 0.0220 | 1.146 | 0.2612 | 0.2594 | A |
| avg_hints | DQN | Difficulty-Based | 0.0747 | 0.3650 | -0.2903 | -4.866 | 3.682e-05 | 4.39e-05 | A |
| frustration_end | DQN | Random | 0.0583 | 0.1842 | -0.1259 | -2.703 | 0.01136 | 0.03212 | A |
| unproductive_repeat_rate | DQN | Q-Learning | 0.0165 | 0.0070 | 0.0095 | 2.081 | 0.04636 | 0.04507 | B |
| skill_gain | SARSA | Prerequisite-Based | 0.0142 | 0.0171 | -0.0029 | -5.347 | 9.678e-06 | 8.326e-07 | B |

## Welch t-test: each RL agent vs best baseline

Best RL = **SARSA**, best baseline = **Prerequisite-Based** (highest mean skill gain over the 30 cells; read-out, not a tuning step).  Welch test on independent cell sets (n = 30 each).

| metric | RL agent | baseline | mean (RL) | mean (base) | diff | Welch *t* | *p* | Cohen's *d* | sig. |
|---|---|---|---|---|---|---|---|---|---|
| skill_gain | SARSA | Prerequisite-Based | 0.0142 | 0.0171 | -0.0029 | -1.396 | 0.1683 | -0.360 | no |
| total_reward | SARSA | Prerequisite-Based | 15.3460 | 11.3514 | 3.9946 | 1.077 | 0.286 | 0.278 | no |
| completion_rate | SARSA | Prerequisite-Based | 0.7923 | 0.7967 | -0.0043 | -0.180 | 0.8576 | -0.047 | no |
| calibration | SARSA | Prerequisite-Based | 0.9047 | 0.9633 | -0.0587 | -2.549 | 0.01357 | -0.658 | yes |
| prereq_violation_rate | SARSA | Prerequisite-Based | 0.0663 | 0.0000 | 0.0663 | 3.975 | 0.0004284 | 1.026 | yes |
| skill_gain | Q-Learning | Prerequisite-Based | 0.0137 | 0.0171 | -0.0034 | -1.669 | 0.1009 | -0.431 | no |
| total_reward | Q-Learning | Prerequisite-Based | 14.8654 | 11.3514 | 3.5139 | 0.971 | 0.3361 | 0.251 | no |
| completion_rate | Q-Learning | Prerequisite-Based | 0.7980 | 0.7967 | 0.0013 | 0.055 | 0.956 | 0.014 | no |
| calibration | Q-Learning | Prerequisite-Based | 0.8997 | 0.9633 | -0.0637 | -2.955 | 0.004707 | -0.763 | yes |
| prereq_violation_rate | Q-Learning | Prerequisite-Based | 0.1043 | 0.0000 | 0.1043 | 4.877 | 3.569e-05 | 1.259 | yes |
| skill_gain | DQN | Prerequisite-Based | 0.0128 | 0.0171 | -0.0043 | -2.224 | 0.03077 | -0.574 | yes |
| total_reward | DQN | Prerequisite-Based | 4.2246 | 11.3514 | -7.1268 | -2.141 | 0.0379 | -0.553 | yes |
| completion_rate | DQN | Prerequisite-Based | 0.7837 | 0.7967 | -0.0130 | -0.532 | 0.5967 | -0.137 | no |
| calibration | DQN | Prerequisite-Based | 0.8797 | 0.9633 | -0.0837 | -3.340 | 0.00147 | -0.862 | yes |
| prereq_violation_rate | DQN | Prerequisite-Based | 0.0923 | 0.0000 | 0.0923 | 3.818 | 0.0006537 | 0.986 | yes |
| skill_gain | Double DQN | Prerequisite-Based | 0.0138 | 0.0171 | -0.0033 | -1.613 | 0.1127 | -0.417 | no |
| total_reward | Double DQN | Prerequisite-Based | 10.7981 | 11.3514 | -0.5533 | -0.156 | 0.8766 | -0.040 | no |
| completion_rate | Double DQN | Prerequisite-Based | 0.7700 | 0.7967 | -0.0267 | -0.986 | 0.3283 | -0.255 | no |
| calibration | Double DQN | Prerequisite-Based | 0.8637 | 0.9633 | -0.0997 | -3.262 | 0.001945 | -0.842 | yes |
| prereq_violation_rate | Double DQN | Prerequisite-Based | 0.0790 | 0.0000 | 0.0790 | 5.048 | 2.219e-05 | 1.303 | yes |

## Notes

* Prereq Viol. = fraction of recommendations whose topic had a prerequisite with mastery < 0.30 at decision time (lower is better; the agents never observe the DAG).
* All values are measured from `results/evaluation/per_run.csv` — nothing is estimated.
