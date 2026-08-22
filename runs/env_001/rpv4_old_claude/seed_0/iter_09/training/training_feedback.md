# Training Feedback

## Final-policy outcome
score=-1.589902, len=755.050000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-132.826009, 181.432183]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 140.038225 | 72.4% | 72.4% | 6.9% |
| velocity_brake_near_target | -34.264108 | -17.7% | 17.7% | 99.8% |
| goal_approach_progress | 8.646793 | 4.5% | 7.2% | 99.7% |
| orientation_stability | -5.325976 | -2.8% | 2.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
