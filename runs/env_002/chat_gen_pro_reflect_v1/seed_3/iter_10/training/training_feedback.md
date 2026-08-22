# Training Feedback

## Final-policy outcome
score=286.724538, len=835.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[38.851592, 324.320376]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2897.527314 | 69.9% | 69.9% | 99.7% |
| balance_gate | 835.288824 | 20.2% | 20.2% | 100.0% |
| gait_quality | 292.400000 | 7.1% | 7.1% | 35.0% |
| gait_efficiency | 81.850494 | 2.0% | 2.0% | 34.8% |
| both_off_penalty | -25.970000 | -0.6% | 0.6% | 31.1% |
| action_penalty | -8.760994 | -0.2% | 0.2% | 100.0% |
| balance_modulation | -2.190617 | -0.1% | 0.1% | 100.0% |
| lidar_penalty | -0.079576 | -0.0% | 0.0% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
