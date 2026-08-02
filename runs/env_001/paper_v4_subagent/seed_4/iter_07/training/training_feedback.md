# Training Feedback

## Final-policy outcome
score=150.812750, len=813.700000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[66.561835, 252.995116]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 1050.676873 | 98.6% | 98.6% | 99.4% |
| progress | 12.444459 | 1.2% | 1.4% | 100.0% |
| angle_penalty | -0.333011 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.067080 | -0.0% | 0.0% | 85.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
