# Training Feedback

## Final-policy outcome
score=233.496804, len=446.100000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[153.933640, 277.797546]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 78.339239 | 70.0% | 70.0% | 20.1% |
| goal_proximity | -10.884943 | -9.7% | 9.7% | 100.0% |
| fuel_efficiency | -9.336000 | -8.3% | 8.3% | 69.8% |
| attitude_constraint | -8.039939 | -7.2% | 7.2% | 100.0% |
| settling_stability | -5.322101 | -4.8% | 4.8% | 97.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
