# Training Feedback

## Final-policy outcome
score=12.575348, len=692.200000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-122.873860, 182.980661]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 1535.325346 | 96.0% | 96.0% | 100.0% |
| fuel_efficiency | -32.687500 | -2.0% | 2.0% | 94.4% |
| settling_stability | -26.527825 | -1.7% | 1.7% | 99.9% |
| attitude_constraint | -5.310040 | -0.3% | 0.3% | 100.0% |
| landing_contact | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
