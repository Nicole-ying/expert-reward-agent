# Training Feedback

## Final-policy outcome
score=-116.652889, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.061344, -97.217550]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -14.658990 | -51.2% | 51.2% | 99.7% |
| goal_proximity | -7.530878 | -26.3% | 26.3% | 100.0% |
| landing_contact | 4.958485 | 17.3% | 17.3% | 1.2% |
| attitude_constraint | -1.356701 | -4.7% | 4.7% | 100.0% |
| fuel_efficiency | -0.129000 | -0.5% | 0.5% | 6.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
