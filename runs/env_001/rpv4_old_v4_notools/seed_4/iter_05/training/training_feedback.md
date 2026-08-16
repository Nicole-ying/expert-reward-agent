# Training Feedback

## Final-policy outcome
score=-107.730994, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.205223, -92.541813]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 10.854215 | 59.4% | 66.5% | 100.0% |
| speed_penalty | -4.147132 | -22.7% | 22.7% | 100.0% |
| attitude_constraint | -0.998233 | -5.5% | 5.5% | 100.0% |
| landing_contact | 0.761456 | 4.2% | 4.2% | 3.1% |
| fuel_efficiency | -0.220000 | -1.2% | 1.2% | 6.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
