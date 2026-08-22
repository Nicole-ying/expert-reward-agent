# Training Feedback

## Final-policy outcome
score=-59.116198, len=72.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-114.214464, -9.895323]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaping_reward | 11.253881 | 44.7% | 46.5% | 100.0% |
| success_bonus | 10.000000 | 39.7% | 39.7% | 0.3% |
| fuel_penalty | -2.640500 | -10.5% | 10.5% | 29.2% |
| crash_penalty | -0.500000 | -2.0% | 2.0% | 0.1% |
| angle_penalty | -0.301930 | -1.2% | 1.2% | 100.0% |
| ang_vel_penalty | -0.039459 | -0.2% | 0.2% | 100.0% |
| boundary_x_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| boundary_y_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
