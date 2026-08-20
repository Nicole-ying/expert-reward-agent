# Training Feedback

## Final-policy outcome
score=-282.035627, len=663.900000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-1034.263527, -10.131371]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 1421.015355 | 86.3% | 88.0% | 96.6% |
| lateral_penalty | -99.651550 | -6.1% | 6.1% | 93.9% |
| angular_velocity_penalty | -65.178205 | -4.0% | 4.0% | 94.0% |
| height_hinge_penalty | -32.046613 | -1.9% | 1.9% | 12.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
