# Training Feedback

## Final-policy outcome
score=-108.687525, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.839570, -91.911322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -66.484895 | -72.9% | 72.9% | 100.0% |
| landing_quality | 16.487447 | 18.1% | 18.1% | 100.0% |
| stability_penalty | -8.169597 | -9.0% | 9.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
