# Training Feedback

## Final-policy outcome
score=-143.494333, len=590.650000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-189.979339, -13.311123]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| altitude_proximity | 61.638530 | 46.1% | 46.1% | 100.0% |
| horizontal_centering | 25.132068 | 18.8% | 18.8% | 100.0% |
| engine_cost | -21.977000 | -16.5% | 16.5% | 99.3% |
| step_cost | -17.719500 | -13.3% | 13.3% | 100.0% |
| speed_penalty | -4.113149 | -3.1% | 3.1% | 100.0% |
| descent_reward | 2.864747 | 2.1% | 2.1% | 86.6% |
| attitude_penalty | -0.139886 | -0.1% | 0.1% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
