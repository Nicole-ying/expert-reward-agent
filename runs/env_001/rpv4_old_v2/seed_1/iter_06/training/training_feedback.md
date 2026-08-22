# Training Feedback

## Final-policy outcome
score=-20.437082, len=611.100000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-261.299908, 212.847442]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 3244.915744 | 98.9% | 98.9% | 13.7% |
| fuel_penalty | -23.950000 | -0.7% | 0.7% | 78.4% |
| progress | 4.203633 | 0.1% | 0.3% | 100.0% |
| orientation_penalty | -2.134559 | -0.1% | 0.1% | 67.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
