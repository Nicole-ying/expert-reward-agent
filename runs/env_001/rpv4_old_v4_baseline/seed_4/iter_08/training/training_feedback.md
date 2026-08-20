# Training Feedback

## Final-policy outcome
score=-114.629125, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 16.740101 | 43.1% | 45.6% | 100.0% |
| settling_stability | -17.262093 | -44.5% | 44.5% | 100.0% |
| landing_contact | 2.800000 | 7.2% | 7.2% | 1.2% |
| attitude_constraint | -0.902184 | -2.3% | 2.3% | 100.0% |
| fuel_efficiency | -0.137500 | -0.4% | 0.4% | 4.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
