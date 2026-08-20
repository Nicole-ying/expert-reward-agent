# Training Feedback

## Final-policy outcome
score=155.346028, len=514.800000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-63.064670, 224.611230]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 35.235666 | 41.0% | 41.0% | 17.5% |
| attitude_constraint | -16.951810 | -19.7% | 19.7% | 100.0% |
| goal_proximity | -16.000613 | -18.6% | 18.6% | 100.0% |
| fuel_efficiency | -11.658000 | -13.6% | 13.6% | 75.5% |
| settling_stability | -6.014708 | -7.0% | 7.0% | 98.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
