# Training Feedback

## Final-policy outcome
score=2657.562275, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2633.297261, 2673.226738]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 1658.971729 | 79.6% | 79.6% | 100.0% |
| forward_speed_bonus | 289.581663 | 13.9% | 13.9% | 100.0% |
| torso_angular_velocity_penalty | -102.516947 | -4.9% | 4.9% | 100.0% |
| torso_angle_penalty | -34.158730 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
