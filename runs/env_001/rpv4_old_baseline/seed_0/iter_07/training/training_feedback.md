# Training Feedback

## Final-policy outcome
score=-123.286761, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.643354, -102.020790]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 69.250000 | 12.7% | 62.9% | 3.2% |
| radial_reward | 133.008165 | 24.4% | 26.1% | 100.0% |
| vel_penalty | -38.708928 | -7.1% | 7.1% | 100.0% |
| angle_penalty | -11.956992 | -2.2% | 2.2% | 100.0% |
| time_penalty | -3.415000 | -0.6% | 0.6% | 100.0% |
| descent_reward | 2.864015 | 0.5% | 0.5% | 94.3% |
| proximity_bonus | 2.136442 | 0.4% | 0.4% | 20.7% |
| engine_penalty | -0.640000 | -0.1% | 0.1% | 4.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
