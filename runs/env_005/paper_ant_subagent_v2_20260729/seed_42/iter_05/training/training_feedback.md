# Training Feedback

## Final-policy outcome
score=-345.873939, len=306.400000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-1269.077816, -4.889502]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| lateral_factor | 196.085967 | 29.4% | 29.4% | 100.0% |
| height_factor | 184.219840 | 27.7% | 27.7% | 100.0% |
| angular_factor | 130.669497 | 19.6% | 19.6% | 100.0% |
| upright_factor | 64.764749 | 9.7% | 9.7% | 21.5% |
| forward_reward | 46.202188 | 6.9% | 6.9% | 14.5% |
| gate | 42.677627 | 6.4% | 6.4% | 21.5% |
| action_cost | 1.373896 | 0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
