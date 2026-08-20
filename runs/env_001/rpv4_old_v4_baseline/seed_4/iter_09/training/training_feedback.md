# Training Feedback

## Final-policy outcome
score=63.660194, len=955.500000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-36.988538, 150.857897]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 897.904747 | 64.4% | 64.4% | 100.0% |
| landing_contact | 399.600000 | 28.6% | 28.6% | 11.9% |
| fuel_efficiency | -46.075000 | -3.3% | 3.3% | 96.4% |
| settling_stability | -36.821348 | -2.6% | 2.6% | 100.0% |
| attitude_constraint | -14.506255 | -1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
