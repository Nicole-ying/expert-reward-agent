# Training Feedback

## Final-policy outcome
score=-18.790023, len=126.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-91.681165, 31.492267]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_progress | 0.724467 | 83.6% | 90.9% | 100.0% |
| angular_vel_penalty | -0.051586 | -6.0% | 6.0% | 99.9% |
| angle_penalty | -0.027651 | -3.2% | 3.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
