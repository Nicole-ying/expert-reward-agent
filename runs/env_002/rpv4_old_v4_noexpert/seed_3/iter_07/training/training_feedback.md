# Training Feedback

## Final-policy outcome
score=274.390805, len=1069.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[33.782973, 305.578804]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1197.463867 | 99.4% | 99.5% | 100.0% |
| balance_penalty | -6.482423 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
