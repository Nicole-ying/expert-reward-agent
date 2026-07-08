# Training Feedback

## Final-policy outcome
score=-87.669632, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-115.820961, -47.855786]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| target_proximity | 6399.664858 | 100.0% | 100.0% | 100.0% |
| velocity_penalty | -0.708993 | -0.0% | 0.0% | 100.0% |
| orientation_penalty | -0.136366 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.104646 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
