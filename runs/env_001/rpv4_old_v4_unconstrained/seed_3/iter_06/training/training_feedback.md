# Training Feedback

## Final-policy outcome
score=57.644809, len=845.800000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-44.608090, 252.584578]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pre_landing_bonus | 186.823219 | 61.9% | 61.9% | 96.8% |
| goal_proximity | -63.820861 | -21.1% | 21.1% | 100.0% |
| settled_reward | 27.320883 | 9.1% | 9.1% | 3.2% |
| landing_approach | 9.305849 | 3.1% | 3.1% | 100.0% |
| velocity_penalty | -4.531410 | -1.5% | 1.5% | 99.7% |
| fuel_penalty | -4.098750 | -1.4% | 1.4% | 96.9% |
| orientation_penalty | -2.414153 | -0.8% | 0.8% | 100.0% |
| contact_bonus | 2.190262 | 0.7% | 0.7% | 0.1% |
| contact_break_penalty | -1.200000 | -0.4% | 0.4% | 0.0% |
| contact_speed_penalty | -0.121308 | -0.0% | 0.0% | 2.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
