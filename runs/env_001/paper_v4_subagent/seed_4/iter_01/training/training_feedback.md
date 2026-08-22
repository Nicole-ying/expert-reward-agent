# Training Feedback

## Final-policy outcome
score=-166.964935, len=997.950000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-242.400897, -126.702705]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 865.369103 | 99.7% | 99.7% | 100.0% |
| stability_penalty | -1.060243 | -0.1% | 0.1% | 100.0% |
| progress | -0.673900 | -0.1% | 0.1% | 100.0% |
| failure_penalty | -0.500000 | -0.1% | 0.1% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
