# Training Feedback

## Final-policy outcome
score=36.972140, len=930.150000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-70.300939, 196.774203]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 242.529974 | 65.1% | 65.1% | 6.3% |
| goal_approach_progress | 57.111455 | 15.3% | 21.2% | 99.6% |
| velocity_brake_near_target | -44.465018 | -11.9% | 11.9% | 99.7% |
| orientation_stability | -6.695241 | -1.8% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
