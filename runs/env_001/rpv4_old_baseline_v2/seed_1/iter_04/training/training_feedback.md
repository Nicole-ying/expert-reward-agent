# Training Feedback

## Final-policy outcome
score=-80.353504, len=813.750000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-222.921763, 195.269367]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 53.250000 | 52.0% | 52.0% | 1.8% |
| distance_penalty | -38.098124 | -37.2% | 37.2% | 100.0% |
| vel_penalty | -8.603585 | -8.4% | 8.4% | 100.0% |
| angle_penalty | -1.473811 | -1.4% | 1.4% | 100.0% |
| dist_reward | 0.378018 | 0.4% | 0.7% | 100.0% |
| success_bonus | 0.200000 | 0.2% | 0.2% | 0.0% |
| angvel_penalty | -0.031442 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
