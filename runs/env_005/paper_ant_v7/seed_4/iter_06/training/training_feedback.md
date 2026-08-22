# Training Feedback

## Final-policy outcome
score=-184.033639, len=577.300000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-1256.425335, 20.101762]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 1154.737013 | 83.3% | 84.7% | 93.0% |
| lateral_penalty | -102.223362 | -7.4% | 7.4% | 90.0% |
| angular_velocity_penalty | -85.318223 | -6.2% | 6.2% | 89.8% |
| height_hinge_penalty | -24.267300 | -1.8% | 1.8% | 10.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
