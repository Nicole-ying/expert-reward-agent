# Training Feedback

## Final-policy outcome
score=-53.287992, len=72.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-88.484955, -17.388169]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| target_proximity | -15.139127 | -76.7% | 76.7% | 100.0% |
| velocity_penalty | -4.241642 | -21.5% | 21.5% | 99.8% |
| angvel_penalty | -0.330935 | -1.7% | 1.7% | 99.7% |
| orientation_penalty | -0.020351 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
