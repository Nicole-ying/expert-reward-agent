# Training Feedback

## Final-policy outcome
score=281.528054, len=238.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[253.089023, 313.295781]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 27.707548 | 86.5% | 88.9% | 93.6% |
| vel_damping | -1.908979 | -6.0% | 6.0% | 93.7% |
| fuel_penalty | -1.199000 | -3.7% | 3.7% | 50.3% |
| angle_stab | -0.442452 | -1.4% | 1.4% | 100.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
