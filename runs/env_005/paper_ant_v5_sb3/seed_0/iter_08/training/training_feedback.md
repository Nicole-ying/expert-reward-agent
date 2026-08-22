# Training Feedback

## Final-policy outcome
score=-228.023737, len=215.400000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-2106.247513, 26.922948]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 65.935228 | 89.9% | 95.5% | 49.8% |
| height_pen | -3.281502 | -4.5% | 4.5% | 42.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
