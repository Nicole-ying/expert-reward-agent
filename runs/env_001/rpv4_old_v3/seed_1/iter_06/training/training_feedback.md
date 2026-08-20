# Training Feedback

## Final-policy outcome
score=-108.177815, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.756541, -91.564287]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.127200 | -92.3% | 92.3% | 100.0% |
| goal_approach | 1.119989 | 5.4% | 5.6% | 100.0% |
| landing_comfort | 0.428127 | 2.1% | 2.1% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
