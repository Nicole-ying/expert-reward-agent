# Training Feedback

## Final-policy outcome
score=-30.778019, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-57.297976, 6.039358]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 4403.989123 | 88.2% | 88.2% | 100.0% |
| distance_penalty | -580.923112 | -11.6% | 11.6% | 100.0% |
| orientation_penalty | -6.370182 | -0.1% | 0.1% | 100.0% |
| velocity_penalty | -2.611327 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
