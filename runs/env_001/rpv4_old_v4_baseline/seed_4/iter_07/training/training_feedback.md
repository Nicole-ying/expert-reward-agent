# Training Feedback

## Final-policy outcome
score=-122.805639, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-146.002977, -100.809739]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -20.666787 | -61.3% | 61.3% | 100.0% |
| goal_proximity | 8.021252 | 23.8% | 25.5% | 100.0% |
| landing_contact | 2.400000 | 7.1% | 7.1% | 1.0% |
| attitude_constraint | -1.923012 | -5.7% | 5.7% | 100.0% |
| fuel_efficiency | -0.142500 | -0.4% | 0.4% | 4.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
