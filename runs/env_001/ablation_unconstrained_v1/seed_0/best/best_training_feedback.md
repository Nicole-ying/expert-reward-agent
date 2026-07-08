# Training Feedback

## Final-policy outcome
score=121.869068, len=837.850000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[29.151419, 204.519208]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 689.224037 | 52.0% | 52.0% | 75.4% |
| distance_reward | 629.048129 | 47.5% | 47.5% | 100.0% |
| stability_penalty | -6.188048 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
