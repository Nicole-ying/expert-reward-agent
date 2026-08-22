# Training Feedback

## Final-policy outcome
score=-131.921973, len=376.250000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-1204.768640, 236.198029]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 295.115233 | 55.7% | 63.7% | 100.0% |
| orientation_reward | 29.537333 | 5.6% | 20.3% | 100.0% |
| lateral_velocity_penalty | -39.032825 | -7.4% | 7.4% | 100.0% |
| angular_velocity_penalty | -35.488919 | -6.7% | 6.7% | 100.0% |
| height_boundary_penalty | -10.383432 | -2.0% | 2.0% | 35.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
