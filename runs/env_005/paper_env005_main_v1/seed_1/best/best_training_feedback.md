# Training Feedback

## Final-policy outcome
score=-16.602316, len=11.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-25.541189, -7.891307]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| angular_velocity_penalty | -21.074904 | -71.4% | 71.4% | 100.0% |
| forward_velocity_reward | 0.954357 | 3.2% | 17.1% | 100.0% |
| vertical_velocity_penalty | -2.353788 | -8.0% | 8.0% | 100.0% |
| lateral_velocity_penalty | -1.037845 | -3.5% | 3.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
