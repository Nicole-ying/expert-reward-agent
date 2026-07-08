# Training Feedback

## Final-policy outcome
score=280.500184, len=908.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[73.299102, 302.718978]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 489.238862 | 94.7% | 94.7% | 100.0% |
| stability_penalty | -27.381375 | -5.3% | 5.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
