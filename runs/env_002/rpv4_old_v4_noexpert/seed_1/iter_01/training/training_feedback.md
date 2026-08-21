# Training Feedback

## Final-policy outcome
score=-17.267416, len=375.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-92.274511, 277.514208]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 222.614715 | 97.2% | 97.2% | 94.3% |
| upright_stability_penalty | -6.525055 | -2.8% | 2.8% | 67.3% |
| vertical_oscillation_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 14/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
