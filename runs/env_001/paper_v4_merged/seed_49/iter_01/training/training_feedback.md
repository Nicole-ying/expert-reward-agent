# Training Feedback

## Final-policy outcome
score=-113.713580, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.449349, -92.527644]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.120390 | 45.9% | 47.5% | 100.0% |
| soft_landing | 0.966077 | 39.6% | 39.6% | 0.9% |
| angvel_penalty | -0.218513 | -9.0% | 9.0% | 0.7% |
| efficiency | -0.096000 | -3.9% | 3.9% | 7.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
