# Training Feedback

## Final-policy outcome
score=-23.641136, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-51.845382, 21.329442]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 647.239433 | 91.2% | 91.2% | 88.0% |
| fuel_efficiency | -28.876500 | -4.1% | 4.1% | 96.3% |
| goal_proximity | -16.396123 | -2.3% | 2.3% | 100.0% |
| attitude_constraint | -12.157995 | -1.7% | 1.7% | 100.0% |
| settling_stability | -4.985995 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
