# Training Feedback

## Final-policy outcome
score=252.555500, len=314.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[225.137799, 287.028697]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_reward | 155.208197 | 93.7% | 93.7% | 100.0% |
| stability_penalty | -6.379674 | -3.9% | 3.9% | 100.0% |
| progress_delta | 3.895362 | 2.4% | 2.4% | 95.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
