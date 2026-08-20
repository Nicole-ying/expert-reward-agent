# Training Feedback

## Final-policy outcome
score=-110.724050, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.523014, -99.036193]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 6.335520 | 43.9% | 52.6% | 16.7% |
| distance_reward | 5.606580 | 38.9% | 40.3% | 100.0% |
| stability_penalty | -1.030940 | -7.2% | 7.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
