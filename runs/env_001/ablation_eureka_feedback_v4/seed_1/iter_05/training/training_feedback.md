# Training Feedback

## Final-policy outcome
score=33.646458, len=850.950000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-66.450193, 292.091603]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -12.499653 | -47.3% | 47.3% | 100.0% |
| contact_proximity | 8.030072 | 30.4% | 30.4% | 1.9% |
| orientation_penalty | -4.318368 | -16.3% | 16.3% | 100.0% |
| velocity_penalty | -1.591719 | -6.0% | 6.0% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
