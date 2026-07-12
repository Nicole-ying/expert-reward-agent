# Training Feedback

## Final-policy outcome
score=231.371376, len=589.350000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[137.422777, 292.936248]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 1036.567659 | 76.8% | 76.8% | 100.0% |
| contact_bonus | 310.228859 | 23.0% | 23.0% | 53.3% |
| velocity_penalty | -2.417221 | -0.2% | 0.2% | 98.5% |
| angle_penalty | -0.422520 | -0.0% | 0.0% | 100.0% |
| angular_velocity_penalty | -0.186958 | -0.0% | 0.0% | 83.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
