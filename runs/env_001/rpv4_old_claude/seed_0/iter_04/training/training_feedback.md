# Training Feedback

## Final-policy outcome
score=103.628718, len=736.950000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-48.804643, 219.162094]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 433.442598 | 90.7% | 90.7% | 13.8% |
| velocity_brake_near_target | -25.202204 | -5.3% | 5.3% | 99.5% |
| goal_approach_progress | 11.665704 | 2.4% | 2.7% | 99.3% |
| orientation_stability | -6.340977 | -1.3% | 1.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
