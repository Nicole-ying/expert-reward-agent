# Training Feedback

## Final-policy outcome
score=3168.343329, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3143.717645, 3184.238977]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 2170.141191 | 95.8% | 95.8% | 100.0% |
| torso_angular_velocity_penalty | -61.797295 | -2.7% | 2.7% | 100.0% |
| torso_angle_penalty | -32.676780 | -1.4% | 1.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
