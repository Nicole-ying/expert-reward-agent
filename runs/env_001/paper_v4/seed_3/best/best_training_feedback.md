# Training Feedback

## Final-policy outcome
score=253.706833, len=362.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[223.583934, 288.190027]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 35.086909 | 73.9% | 73.9% | 11.3% |
| progress | 6.948097 | 14.6% | 14.9% | 95.9% |
| velocity_penalty | -3.418937 | -7.2% | 7.2% | 96.6% |
| orientation_penalty | -1.901877 | -4.0% | 4.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
