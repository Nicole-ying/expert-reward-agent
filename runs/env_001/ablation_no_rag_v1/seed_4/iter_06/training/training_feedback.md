# Training Feedback

## Final-policy outcome
score=93.238046, len=931.450000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[40.769013, 272.047153]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 135.500000 | 92.6% | 92.6% | 14.5% |
| target_proximity | -7.475840 | -5.1% | 5.1% | 100.0% |
| velocity_penalty | -2.283529 | -1.6% | 1.6% | 99.9% |
| engine_penalty | -0.464850 | -0.3% | 0.3% | 49.9% |
| angvel_penalty | -0.339097 | -0.2% | 0.2% | 99.3% |
| orientation_penalty | -0.332273 | -0.2% | 0.2% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
