# Training Feedback

## Final-policy outcome
score=73.068896, len=414.050000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-456.050036, 266.210134]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 52.447056 | 93.2% | 93.2% | 98.6% |
| distance_progress | 2.175501 | 3.9% | 4.4% | 99.1% |
| stability_penalty | -1.346446 | -2.4% | 2.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
