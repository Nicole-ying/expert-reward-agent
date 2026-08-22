# Training Feedback

## Final-policy outcome
score=-102.307853, len=68.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.531851, -73.251336]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 39.184449 | 88.8% | 92.0% | 100.0% |
| soft_landing_bonus | 2.308037 | 5.2% | 5.2% | 3.4% |
| attitude_penalty | -1.245961 | -2.8% | 2.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
