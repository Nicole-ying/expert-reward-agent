# Training Feedback

## Final-policy outcome
score=549.738928, len=170.200000, terminated=5/5, truncated=0/5, reward_errors=0
score_range=[545.044122, 553.672529]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 764.903983 | 98.1% | 98.2% | 100.0% |
| upright_penalty | -14.382514 | -1.8% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10004
- early_terminal (<150 steps and score<-50): 0/5
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
