# Training Feedback

## Final-policy outcome
score=285.187709, len=1110.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[281.447822, 288.588786]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 756.201792 | 95.9% | 95.9% | 100.0% |
| balance_penalty | -32.246434 | -4.1% | 4.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
