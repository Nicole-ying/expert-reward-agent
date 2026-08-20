# Training Feedback

## Final-policy outcome
score=12.313339, len=161.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-181.890465, 175.626337]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 158.515021 | 53.4% | 63.7% | 100.0% |
| orientation_reward | 46.309150 | 15.6% | 15.6% | 100.0% |
| lateral_velocity_penalty | -23.634231 | -8.0% | 8.0% | 100.0% |
| angular_velocity_penalty | -20.519881 | -6.9% | 6.9% | 100.0% |
| height_deviation_penalty | -17.426705 | -5.9% | 5.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
