# Training Feedback

## Final-policy outcome
score=-49.895602, len=349.350000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-1124.206240, 252.460301]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_upright_product | 366.563706 | 92.0% | 97.9% | 86.1% |
| torque_penalty | -5.851495 | -1.5% | 1.5% | 100.0% |
| height_survival_penalty | -2.474791 | -0.6% | 0.6% | 16.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
