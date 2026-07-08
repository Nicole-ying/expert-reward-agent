# Training Feedback

## Final-policy outcome
score=313.460724, len=1158.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[310.931298, 314.849186]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.640006 | 76.8% | 76.8% | 100.0% |
| stability_penalty | -97.675690 | -14.9% | 14.9% | 100.0% |
| energy_penalty | -54.566253 | -8.3% | 8.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
