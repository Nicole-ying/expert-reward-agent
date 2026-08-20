# Training Feedback

## Final-policy outcome
score=-115.554859, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.753007, -96.731262]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -23.002279 | -65.9% | 65.9% | 100.0% |
| goal_proximity | 7.431531 | 21.3% | 21.3% | 100.0% |
| landing_contact | 3.100000 | 8.9% | 8.9% | 1.3% |
| attitude_constraint | -1.208334 | -3.5% | 3.5% | 100.0% |
| fuel_efficiency | -0.140000 | -0.4% | 0.4% | 4.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
