# Training Feedback

## Final-policy outcome
score=91.871937, len=978.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[44.437008, 250.320034]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 639.353716 | 70.3% | 70.3% | 13.7% |
| landing_progress | 4.609947 | 0.5% | 29.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
