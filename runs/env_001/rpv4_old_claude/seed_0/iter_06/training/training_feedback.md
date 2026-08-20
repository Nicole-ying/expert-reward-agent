# Training Feedback

## Final-policy outcome
score=17.867905, len=784.350000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-169.804912, 227.300020]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 509.239245 | 90.1% | 90.1% | 15.9% |
| velocity_brake_near_target | -32.704433 | -5.8% | 5.8% | 100.0% |
| goal_approach_progress | 8.423406 | 1.5% | 2.5% | 100.0% |
| orientation_stability | -9.012481 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
