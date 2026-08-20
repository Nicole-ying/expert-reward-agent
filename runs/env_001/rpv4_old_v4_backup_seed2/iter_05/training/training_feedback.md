# Training Feedback

## Final-policy outcome
score=-197.455359, len=185.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-292.101504, -148.214023]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -0.716674 | -49.3% | 49.3% | 15.8% |
| progress | -0.182647 | -12.6% | 27.1% | 100.0% |
| angular_vel_penalty | -0.342899 | -23.6% | 23.6% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
