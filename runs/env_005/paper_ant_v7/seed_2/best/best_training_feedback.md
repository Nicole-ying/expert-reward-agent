# Training Feedback

## Final-policy outcome
score=-102.079657, len=218.100000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-883.698747, 18.669044]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 159.081901 | 36.6% | 47.4% | 59.4% |
| upright_orientation_penalty | -177.884621 | -40.9% | 40.9% | 58.7% |
| lateral_drift_penalty | -36.084192 | -8.3% | 8.3% | 58.9% |
| action_energy_penalty | -7.974121 | -1.8% | 1.8% | 100.0% |
| height_health_penalty | -6.991385 | -1.6% | 1.6% | 20.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
