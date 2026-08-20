# Training Feedback

## Final-policy outcome
score=-63.735456, len=585.900000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-188.200601, 196.363887]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 76.437707 | 98.0% | 98.0% | 92.9% |
| progress_reward | 0.695133 | 0.9% | 2.0% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
