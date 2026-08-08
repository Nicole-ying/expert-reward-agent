# Training Feedback

## Final-policy outcome
score=-61.410602, len=326.700000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-82.993346, -35.781411]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 78.977525 | 96.8% | 98.4% | 99.3% |
| stability_angle_penalty | -1.286859 | -1.6% | 1.6% | 3.3% |
| stability_angvel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| vertical_speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
