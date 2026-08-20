# Training Feedback

## Final-policy outcome
score=-94.993338, len=1600.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-105.309170, -91.464349]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gait_quality | 1593.750000 | 76.5% | 76.5% | 99.6% |
| gait_bonus | 478.125000 | 23.0% | 23.0% | 99.6% |
| balance_penalty | -7.735648 | -0.4% | 0.4% | 100.0% |
| both_off_ground_penalty | -2.400000 | -0.1% | 0.1% | 0.8% |
| forward_progress | 0.286244 | 0.0% | 0.0% | 2.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
