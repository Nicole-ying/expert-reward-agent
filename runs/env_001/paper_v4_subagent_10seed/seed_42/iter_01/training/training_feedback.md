# Training Feedback

## Final-policy outcome
score=163.329391, len=436.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-74.671665, 272.150063]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 28.042383 | 93.7% | 93.7% | 9.9% |
| progress_delta | 1.180373 | 3.9% | 5.8% | 98.2% |
| orientation_penalty | -0.151964 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
