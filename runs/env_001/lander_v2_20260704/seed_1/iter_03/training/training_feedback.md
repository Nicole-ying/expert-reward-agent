# Training Feedback

## Final-policy outcome
score=-87.910428, len=69.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-131.026226, -59.852507]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_cost | -75.378420 | -57.8% | 57.8% | 100.0% |
| approach_landing_reward | 45.403254 | 34.8% | 34.8% | 99.1% |
| stability_cost | -9.646084 | -7.4% | 7.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
