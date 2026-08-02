# Training Feedback

## Final-policy outcome
score=-271.102448, len=463.800000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-1115.005493, 46.154214]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 314.041572 | 91.1% | 97.4% | 80.7% |
| height_reward | -9.051687 | -2.6% | 2.6% | 14.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
