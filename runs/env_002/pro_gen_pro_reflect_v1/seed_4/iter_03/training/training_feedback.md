# Training Feedback

## Final-policy outcome
score=-92.753400, len=1600.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-97.528593, -86.086546]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gait_cycle_reward | 271.530000 | 99.3% | 99.3% | 49.3% |
| torque_penalty | -1.866633 | -0.7% | 0.7% | 100.0% |
| tilt_penalty | -0.000918 | -0.0% | 0.0% | 0.0% |
| angvel_penalty | -0.000147 | -0.0% | 0.0% | 99.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
