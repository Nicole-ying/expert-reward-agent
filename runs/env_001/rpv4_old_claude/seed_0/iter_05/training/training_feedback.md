# Training Feedback

## Final-policy outcome
score=-38.359016, len=951.800000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-164.496471, 10.675340]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_approach_progress | 92.032730 | 58.7% | 70.5% | 100.0% |
| velocity_brake_near_target | -31.682452 | -20.2% | 20.2% | 100.0% |
| contact_landing_reward | 9.553585 | 6.1% | 6.1% | 0.3% |
| orientation_stability | -5.053642 | -3.2% | 3.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
