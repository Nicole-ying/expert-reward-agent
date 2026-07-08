# Training Feedback

## Final-policy outcome
score=-113.994484, len=69.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-201.361576, 16.024257]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_cost | -75.732187 | -57.7% | 57.7% | 100.0% |
| approach_landing_reward | 44.684147 | 34.0% | 34.0% | 99.7% |
| stability_cost | -10.856522 | -8.3% | 8.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
