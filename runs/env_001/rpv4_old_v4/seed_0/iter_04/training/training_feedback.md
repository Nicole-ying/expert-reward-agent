# Training Feedback

## Final-policy outcome
score=-59.194438, len=977.900000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-113.828086, -7.745415]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proximity_reward | 4731.133226 | 99.5% | 99.5% | 100.0% |
| vel_damping | -21.755565 | -0.5% | 0.5% | 92.1% |
| progress_reward | 0.837051 | 0.0% | 0.0% | 100.0% |
| orientation_penalty | -0.558160 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
