# Training Feedback

## Final-policy outcome
score=275.976735, len=906.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[0.782889, 312.170857]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 476.164066 | 98.6% | 98.6% | 100.0% |
| energy_penalty | -6.471032 | -1.3% | 1.3% | 100.0% |
| stability_penalty | -0.247776 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
