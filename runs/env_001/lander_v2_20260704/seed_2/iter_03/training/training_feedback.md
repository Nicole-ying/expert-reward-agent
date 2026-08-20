# Training Feedback

## Final-policy outcome
score=-108.369798, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.882840, -78.933292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -66.460267 | -79.0% | 79.0% | 100.0% |
| landing_quality | 9.463003 | 11.3% | 11.3% | 42.4% |
| stability_penalty | -8.165946 | -9.7% | 9.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
