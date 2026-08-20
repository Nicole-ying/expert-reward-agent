# Training Feedback

## Final-policy outcome
score=-16.607985, len=143.250000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-100.455290, 34.857703]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality_reward | 7.945185 | 76.1% | 76.1% | 1.3% |
| potential_diff | 1.096048 | 10.5% | 15.3% | 100.0% |
| angle_penalty | -0.899998 | -8.6% | 8.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
