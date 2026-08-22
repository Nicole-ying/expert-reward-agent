# Training Feedback

## Final-policy outcome
score=-112.950415, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.423093, -95.165512]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 11.199824 | 85.8% | 88.8% | 100.0% |
| angvel_penalty | -0.875868 | -6.7% | 6.7% | 99.5% |
| landing_bonus | 0.403354 | 3.1% | 3.1% | 0.5% |
| fuel_penalty | -0.170000 | -1.3% | 1.3% | 5.0% |
| angle_penalty | -0.015237 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
