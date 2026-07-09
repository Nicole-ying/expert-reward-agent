# Training Feedback

## Final-policy outcome
score=-111.070097, len=69.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.255058, -90.576126]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_tracking_reward | -43.020149 | -65.3% | 65.3% | 100.0% |
| landing_improvement | 21.697975 | 32.9% | 32.9% | 47.8% |
| progress_reward | 1.116579 | 1.7% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
