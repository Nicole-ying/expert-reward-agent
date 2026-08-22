# Training Feedback

## Final-policy outcome
score=172.869308, len=663.000000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[38.140480, 250.575294]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 391.203639 | 65.7% | 65.7% | 100.0% |
| landing_contact | 115.350000 | 19.4% | 19.4% | 18.3% |
| settling_stability | -41.362472 | -6.9% | 6.9% | 99.9% |
| fuel_efficiency | -27.587500 | -4.6% | 4.6% | 83.2% |
| attitude_constraint | -19.804275 | -3.3% | 3.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
