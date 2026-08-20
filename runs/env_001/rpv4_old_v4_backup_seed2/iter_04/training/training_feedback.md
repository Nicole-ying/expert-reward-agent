# Training Feedback

## Final-policy outcome
score=-295.326995, len=138.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-343.965185, -239.829467]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_guidance | -1.595910 | -18.9% | 53.9% | 100.0% |
| speed_penalty | -2.096809 | -24.9% | 24.9% | 63.2% |
| angular_vel_penalty | -1.265816 | -15.0% | 15.0% | 100.0% |
| progress | -0.026234 | -0.3% | 6.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
