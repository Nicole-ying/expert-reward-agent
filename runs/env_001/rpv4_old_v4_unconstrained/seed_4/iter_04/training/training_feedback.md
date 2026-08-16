# Training Feedback

## Final-policy outcome
score=-109.733781, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -82.193259]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -28.486119 | -41.2% | 59.0% | 100.0% |
| landing_contact | 10.500000 | 15.2% | 15.2% | 0.8% |
| attitude_constraint | -9.176481 | -13.3% | 13.4% | 100.0% |
| goal_proximity | 8.156559 | 11.8% | 12.2% | 100.0% |
| fuel_efficiency | -0.192500 | -0.3% | 0.3% | 5.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
