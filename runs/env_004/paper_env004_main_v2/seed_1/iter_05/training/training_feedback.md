# Training Feedback

## Final-policy outcome
score=3163.804808, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3093.144936, 3216.396881]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 2165.804545 | 92.2% | 92.2% | 99.6% |
| torso_angular_velocity_penalty | -105.646105 | -4.5% | 4.5% | 100.0% |
| torso_angle_penalty | -47.336195 | -2.0% | 2.0% | 100.0% |
| acceleration_bonus | 29.058110 | 1.2% | 1.2% | 46.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
