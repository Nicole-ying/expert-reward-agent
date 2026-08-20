# Training Feedback

## Final-policy outcome
score=131.350093, len=831.250000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[12.949437, 257.366190]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 6100.000000 | 99.3% | 99.3% | 7.3% |
| progress | 24.704294 | 0.4% | 0.5% | 100.0% |
| fuel_penalty | -7.933500 | -0.1% | 0.1% | 95.4% |
| vel_damping | -1.529314 | -0.0% | 0.0% | 99.9% |
| angle_stab | -0.563688 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
