# Training Feedback

## Final-policy outcome
score=120.019100, len=596.900000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[19.801820, 262.333619]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 3456.106416 | 99.8% | 99.8% | 7.1% |
| shaped_progress | 4.359156 | 0.1% | 0.1% | 99.6% |
| angular_vel_penalty | -0.725148 | -0.0% | 0.0% | 97.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
