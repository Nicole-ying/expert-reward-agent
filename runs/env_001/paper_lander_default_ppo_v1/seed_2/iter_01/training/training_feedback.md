# Training Feedback

## Final-policy outcome
score=-113.995984, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -132.945396 | -93.0% | 93.0% | 100.0% |
| stability_penalty | -9.826903 | -6.9% | 6.9% | 100.0% |
| soft_landing_proxy | 0.200000 | 0.1% | 0.1% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
