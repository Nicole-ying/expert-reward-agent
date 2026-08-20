# Training Feedback

## Final-policy outcome
score=-50.899432, len=407.800000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-265.483793, 60.151375]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 325.217109 | 48.8% | 53.3% | 100.0% |
| orientation_reward | 198.801061 | 29.9% | 29.9% | 100.0% |
| stability_penalty | -112.504495 | -16.9% | 16.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
