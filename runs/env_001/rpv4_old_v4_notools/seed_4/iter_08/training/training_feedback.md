# Training Feedback

## Final-policy outcome
score=-653.720258, len=120.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-2382.833590, -292.239678]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity | -41.215924 | -52.7% | 52.7% | 19.9% |
| fuel | -20.900000 | -26.7% | 26.7% | 88.1% |
| attitude | -9.775370 | -12.5% | 12.5% | 100.0% |
| distance | -6.040630 | -7.7% | 8.1% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 16/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
