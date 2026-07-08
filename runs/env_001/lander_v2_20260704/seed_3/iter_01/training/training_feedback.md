# Training Feedback

## Final-policy outcome
score=-95.246174, len=68.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.703384, -73.443454]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 1.780495 | 42.9% | 42.9% | 1.9% |
| stability_penalty | -1.215746 | -29.3% | 29.3% | 100.0% |
| approach_reward | 1.118138 | 26.9% | 27.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
