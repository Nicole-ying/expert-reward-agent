# Training Feedback

## Final-policy outcome
score=-63.974403, len=963.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-95.513496, -20.657539]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_distance_progress | 3.779845 | 27.5% | 55.3% | 100.0% |
| soft_landing | -4.930737 | -35.9% | 35.9% | 100.0% |
| orientation_penalty | -1.218466 | -8.9% | 8.9% | 100.0% |
| landing_contact_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
