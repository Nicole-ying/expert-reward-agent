# Training Feedback

## Final-policy outcome
score=-118.059196, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.398763, -95.050266]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 204.129179 | 59.3% | 61.6% | 3.1% |
| radial_reward | 88.889133 | 25.8% | 27.6% | 100.0% |
| vy_penalty | -19.152378 | -5.6% | 5.6% | 83.8% |
| vx_penalty | -8.537503 | -2.5% | 2.5% | 100.0% |
| time_penalty | -3.420000 | -1.0% | 1.0% | 100.0% |
| angle_penalty | -2.877625 | -0.8% | 0.8% | 100.0% |
| x_penalty | -2.100649 | -0.6% | 0.6% | 100.0% |
| descent_reward | 0.716764 | 0.2% | 0.2% | 94.3% |
| engine_penalty | -0.577500 | -0.2% | 0.2% | 5.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
