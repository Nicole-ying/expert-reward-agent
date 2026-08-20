# Training Feedback

## Final-policy outcome
score=3260.345083, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3230.610603, 3275.060345]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 2262.133283 | 77.7% | 77.7% | 99.2% |
| forward_speed_bonus | 535.547002 | 18.4% | 18.4% | 99.2% |
| torso_angular_velocity_penalty | -65.025472 | -2.2% | 2.2% | 100.0% |
| torso_angle_penalty | -48.941591 | -1.7% | 1.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
