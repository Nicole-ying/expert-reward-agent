# Training Feedback

## Final-policy outcome
score=3038.318325, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2988.756371, 3076.578863]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 2039.817728 | 63.6% | 63.6% | 99.8% |
| forward_speed_bonus | 1096.804772 | 34.2% | 34.2% | 99.8% |
| torso_angular_velocity_penalty | -53.419309 | -1.7% | 1.7% | 100.0% |
| torso_angle_penalty | -18.755597 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
