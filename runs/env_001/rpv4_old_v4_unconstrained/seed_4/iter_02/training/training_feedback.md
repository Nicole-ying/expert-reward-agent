# Training Feedback

## Final-policy outcome
score=-119.337430, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.445027, -98.057243]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -121.498283 | -51.4% | 52.9% | 100.0% |
| goal_proximity | -90.806575 | -38.4% | 38.4% | 100.0% |
| attitude_constraint | -17.107631 | -7.2% | 7.3% | 100.0% |
| landing_contact | 3.100000 | 1.3% | 1.3% | 0.5% |
| fuel_efficiency | -0.135000 | -0.1% | 0.1% | 4.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
