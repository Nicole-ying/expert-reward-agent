# Training Feedback

## Final-policy outcome
score=-35.347845, len=817.550000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-72.738672, 53.111565]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 52.682058 | 73.2% | 73.2% | 100.0% |
| progress_reward | 7.926148 | 11.0% | 18.0% | 100.0% |
| stability_penalty | -6.309090 | -8.8% | 8.8% | 100.0% |
| landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
