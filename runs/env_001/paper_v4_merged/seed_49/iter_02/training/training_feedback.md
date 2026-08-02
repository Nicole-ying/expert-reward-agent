# Training Feedback

## Final-policy outcome
score=-115.677399, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.820934, -92.536457]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 0.777300 | 56.6% | 56.6% | 1.2% |
| angvel_penalty | -0.298458 | -21.7% | 21.7% | 1.0% |
| progress | 0.199347 | 14.5% | 16.7% | 53.5% |
| efficiency | -0.069000 | -5.0% | 5.0% | 5.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
