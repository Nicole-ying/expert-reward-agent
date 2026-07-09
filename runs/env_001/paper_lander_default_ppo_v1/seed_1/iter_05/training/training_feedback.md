# Training Feedback

## Final-policy outcome
score=162.940332, len=463.700000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-33.334812, 302.039261]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 1186.385838 | 90.5% | 90.5% | 96.1% |
| speed_tracking_reward | -123.661962 | -9.4% | 9.4% | 100.0% |
| progress_reward | 1.235939 | 0.1% | 0.1% | 98.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
