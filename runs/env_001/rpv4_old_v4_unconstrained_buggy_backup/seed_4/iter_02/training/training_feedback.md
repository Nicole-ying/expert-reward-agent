# Training Feedback

## Final-policy outcome
score=-113.598118, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.587355, -96.181860]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -33.249548 | -61.5% | 61.5% | 100.0% |
| settling_stability | -18.853121 | -34.9% | 34.9% | 100.0% |
| landing_contact | 1.125000 | 2.1% | 2.1% | 0.7% |
| attitude_constraint | -0.752751 | -1.4% | 1.4% | 100.0% |
| fuel_efficiency | -0.091500 | -0.2% | 0.2% | 4.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
