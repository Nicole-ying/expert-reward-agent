# Training Feedback

## Final-policy outcome
score=-111.880830, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-127.611244, -95.059093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_delta | 56.062585 | 83.8% | 86.7% | 100.0% |
| velocity_danger | -8.531347 | -12.7% | 12.7% | 100.0% |
| orientation_penalty | -0.389229 | -0.6% | 0.6% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
