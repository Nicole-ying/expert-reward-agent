# Training Feedback

## Final-policy outcome
score=-19.195913, len=10.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-48.286113, 3.325283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_cost | -35.691970 | -76.1% | 76.1% | 100.0% |
| height_management | -5.710450 | -12.2% | 12.2% | 88.6% |
| lateral_penalty | -3.620356 | -7.7% | 7.7% | 100.0% |
| forward_encouragement | -0.286118 | -0.6% | 4.0% | 100.0% |
| contact_reward | 0.000979 | 0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
