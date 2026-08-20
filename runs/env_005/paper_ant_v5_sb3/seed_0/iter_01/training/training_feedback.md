# Training Feedback

## Final-policy outcome
score=-371.497217, len=320.000000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-1287.446200, 36.328565]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 41.808643 | 72.3% | 86.8% | 29.3% |
| upright | -7.206391 | -12.5% | 12.5% | 13.8% |
| height | -0.446474 | -0.8% | 0.8% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
