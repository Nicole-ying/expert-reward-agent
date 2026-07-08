# Training Feedback

## Final-policy outcome
score=237.210338, len=428.250000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[108.033655, 279.230728]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 327.372166 | 90.6% | 90.6% | 100.0% |
| fuel_penalty | -13.787500 | -3.8% | 3.8% | 64.4% |
| tilt_penalty | -10.141923 | -2.8% | 2.8% | 100.0% |
| velocity_penalty | -8.431129 | -2.3% | 2.3% | 99.8% |
| rotation_penalty | -1.472777 | -0.4% | 0.4% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
