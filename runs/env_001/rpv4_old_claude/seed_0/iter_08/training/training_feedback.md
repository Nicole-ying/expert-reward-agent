# Training Feedback

## Final-policy outcome
score=8.805457, len=848.000000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-101.251629, 176.744185]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 111.096459 | 69.3% | 69.3% | 3.4% |
| velocity_brake_near_target | -29.310318 | -18.3% | 18.3% | 100.0% |
| goal_approach_progress | 9.582941 | 6.0% | 8.8% | 100.0% |
| orientation_stability | -5.792886 | -3.6% | 3.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
