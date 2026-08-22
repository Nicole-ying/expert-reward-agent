# Training Feedback

## Final-policy outcome
score=-5.856053, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-38.577245, 29.816082]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 642.588794 | 85.1% | 85.1% | 100.0% |
| settling_stability | -53.825902 | -7.1% | 7.1% | 100.0% |
| fuel_efficiency | -40.435000 | -5.4% | 5.4% | 80.9% |
| attitude_constraint | -18.657512 | -2.5% | 2.5% | 100.0% |
| landing_contact | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
