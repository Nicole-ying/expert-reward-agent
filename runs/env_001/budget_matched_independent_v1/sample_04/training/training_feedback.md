# Training Feedback

## Final-policy outcome
score=-559.734915, len=62.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-650.803456, -373.275576]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| orientation_stability | -6750.584479 | -55.9% | 55.9% | 100.0% |
| soft_landing_velocity | -3430.775048 | -28.4% | 28.4% | 100.0% |
| goal_proximity | -1890.378758 | -15.7% | 15.7% | 100.0% |
| contact_encouragement | 7.000000 | 0.1% | 0.1% | 1.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
