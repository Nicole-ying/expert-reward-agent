# Training Feedback

## Final-policy outcome
score=-73.816961, len=87.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-126.600305, -24.521012]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | 6.178687 | 59.1% | 63.0% | 100.0% |
| landing_quality | 2.544800 | 24.4% | 25.8% | 18.3% |
| stability_penalty | -1.173605 | -11.2% | 11.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
