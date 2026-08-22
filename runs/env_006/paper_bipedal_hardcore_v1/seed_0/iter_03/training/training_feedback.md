# Training Feedback

## Final-policy outcome
score=-49.537949, len=163.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-74.961435, -28.224934]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_speed | 98.221734 | 99.1% | 99.1% | 98.9% |
| stability_tilt_hinge_penalty | -0.861119 | -0.9% | 0.9% | 15.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
