# Training Feedback

## Final-policy outcome
score=858.497184, len=791.500000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[363.240187, 1035.687272]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_reward | 238.286070 | 51.4% | 51.4% | 100.0% |
| forward_progress | 206.177945 | 44.5% | 44.5% | 58.9% |
| torso_angular_velocity_penalty | -9.780114 | -2.1% | 2.1% | 100.0% |
| torso_angle_penalty | -9.317436 | -2.0% | 2.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
