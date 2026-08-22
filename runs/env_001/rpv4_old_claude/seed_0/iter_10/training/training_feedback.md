# Training Feedback

## Final-policy outcome
score=-118.066581, len=935.900000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-230.440602, -36.795420]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_brake_near_target | -23.587882 | -60.9% | 60.9% | 100.0% |
| goal_approach_progress | 5.937139 | 15.3% | 24.2% | 100.0% |
| orientation_stability | -5.754714 | -14.9% | 14.9% | 100.0% |
| contact_landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
