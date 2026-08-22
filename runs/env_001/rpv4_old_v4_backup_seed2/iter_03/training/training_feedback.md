# Training Feedback

## Final-policy outcome
score=-222.850054, len=209.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-336.317345, -168.881892]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -0.735491 | -46.9% | 46.9% | 14.8% |
| progress | -0.298984 | -19.1% | 29.6% | 100.0% |
| angular_vel_penalty | -0.367962 | -23.5% | 23.5% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
