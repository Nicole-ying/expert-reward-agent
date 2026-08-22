# Training Feedback

## Final-policy outcome
score=255.399284, len=332.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[227.124994, 285.771283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 78.196610 | 47.7% | 47.7% | 100.0% |
| descent_bonus | 30.323321 | 18.5% | 18.5% | 92.1% |
| contact_reward | 24.030000 | 14.7% | 14.7% | 12.5% |
| progress_reward | 13.664675 | 8.3% | 8.5% | 95.5% |
| action_reward | -9.287000 | -5.7% | 5.7% | 89.3% |
| vel_damping | -7.740864 | -4.7% | 4.7% | 41.0% |
| orientation_penalty | -0.522724 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
