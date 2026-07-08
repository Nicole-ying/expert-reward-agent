# Training Feedback

## Final-policy outcome
score=-85.750932, len=88.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-120.929149, -2.598324]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | 6.148047 | 77.2% | 80.2% | 100.0% |
| stability_penalty | -1.074611 | -13.5% | 13.5% | 100.0% |
| soft_landing_proxy | 0.500000 | 6.3% | 6.3% | 0.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
