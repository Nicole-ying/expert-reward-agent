# Training Feedback

## Final-policy outcome
score=-42.359864, len=775.500000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-193.533817, 176.299875]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 331.808395 | 75.7% | 75.7% | 100.0% |
| fuel_efficiency | -37.137500 | -8.5% | 8.5% | 95.8% |
| landing_contact | 30.300000 | 6.9% | 6.9% | 1.1% |
| settling_stability | -29.237245 | -6.7% | 6.7% | 100.0% |
| attitude_constraint | -10.120989 | -2.3% | 2.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
