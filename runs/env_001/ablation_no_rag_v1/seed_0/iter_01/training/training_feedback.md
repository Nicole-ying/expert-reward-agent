# Training Feedback

## Final-policy outcome
score=-110.225592, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.839570, -92.492124]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -66.464309 | -89.3% | 89.3% | 100.0% |
| velocity_penalty | -7.296741 | -9.8% | 9.8% | 100.0% |
| tilt_penalty | -0.379022 | -0.5% | 0.5% | 100.0% |
| rotation_penalty | -0.322699 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
