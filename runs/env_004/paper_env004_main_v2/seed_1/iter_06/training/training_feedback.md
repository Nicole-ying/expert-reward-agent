# Training Feedback

## Final-policy outcome
score=2470.046398, len=816.900000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[1330.131390, 3117.666519]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 1655.275534 | 76.8% | 76.8% | 100.0% |
| forward_speed_bonus | 427.593129 | 19.8% | 19.8% | 100.0% |
| torso_angle_penalty | -40.444558 | -1.9% | 1.9% | 100.0% |
| torso_angular_velocity_penalty | -32.962332 | -1.5% | 1.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
