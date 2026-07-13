# Training Feedback

## Final-policy outcome
score=-26.298515, len=951.000000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-157.100879, 157.420870]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 1523.525591 | 93.2% | 93.2% | 82.7% |
| proximity | 67.770953 | 4.1% | 4.1% | 100.0% |
| fuel_penalty | -42.997500 | -2.6% | 2.6% | 90.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
