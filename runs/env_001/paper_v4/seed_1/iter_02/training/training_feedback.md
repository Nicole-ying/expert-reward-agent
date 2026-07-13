# Training Feedback

## Final-policy outcome
score=-17.878810, len=759.350000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-181.708159, 222.063453]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 49.682843 | 97.3% | 97.3% | 1.2% |
| shaped_progress | 0.829294 | 1.6% | 2.1% | 99.8% |
| angular_vel_penalty | -0.330852 | -0.6% | 0.6% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
