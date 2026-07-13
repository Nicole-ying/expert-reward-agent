# Training Feedback

## Final-policy outcome
score=240.602907, len=334.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-13.715721, 287.722762]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality_improvement | 84.716626 | 64.1% | 95.8% | 97.8% |
| shaped_progress | 4.677420 | 3.5% | 3.8% | 95.5% |
| angular_vel_penalty | -0.461283 | -0.3% | 0.3% | 94.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
