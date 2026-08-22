# Training Feedback

## Final-policy outcome
score=186.537439, len=260.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[14.384539, 288.210746]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 19.171905 | 53.2% | 53.2% | 19.4% |
| descent_bonus | 6.855863 | 19.0% | 19.0% | 84.5% |
| distance_progress | 6.257078 | 17.4% | 18.6% | 96.8% |
| stability | -1.922777 | -5.3% | 5.3% | 100.0% |
| fuel_penalty | -1.383500 | -3.8% | 3.8% | 53.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
