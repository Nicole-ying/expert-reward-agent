# Training Feedback

## Final-policy outcome
score=-471.595097, len=968.100000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-536.351823, -144.072933]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance | 2749.543247 | 71.9% | 71.9% | 100.0% |
| velocity | -916.312093 | -24.0% | 24.0% | 100.0% |
| fuel | -153.015000 | -4.0% | 4.0% | 100.0% |
| attitude | -4.137697 | -0.1% | 0.1% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
