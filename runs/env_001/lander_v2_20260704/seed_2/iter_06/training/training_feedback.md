# Training Feedback

## Final-policy outcome
score=-33.723676, len=254.800000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-209.673175, 230.622357]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 169.973915 | 49.9% | 49.9% | 100.0% |
| distance_reward | -156.101764 | -45.8% | 45.8% | 100.0% |
| stability_penalty | -14.637592 | -4.3% | 4.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
