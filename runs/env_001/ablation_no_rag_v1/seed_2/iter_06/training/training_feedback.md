# Training Feedback

## Final-policy outcome
score=-112.793808, len=681.700000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-252.145943, 276.321943]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 990.000000 | 99.3% | 99.3% | 0.7% |
| potential_diff | 0.587719 | 0.1% | 0.5% | 100.0% |
| angle_penalty | -1.585910 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
