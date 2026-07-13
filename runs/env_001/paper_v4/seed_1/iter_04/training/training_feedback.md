# Training Feedback

## Final-policy outcome
score=-29.475926, len=76.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-66.090146, 6.503554]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaped_progress | 24.534608 | 52.4% | 55.1% | 100.0% |
| landing_bonus | 20.359603 | 43.5% | 43.5% | 0.8% |
| angular_vel_penalty | -0.687755 | -1.5% | 1.5% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
