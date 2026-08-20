# Training Feedback

## Final-policy outcome
score=-37.583039, len=947.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-107.997234, 127.780807]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_distance_progress | 4.980184 | 34.3% | 49.2% | 100.0% |
| soft_landing | -2.922441 | -20.1% | 20.1% | 100.0% |
| landing_contact_success | 2.312616 | 15.9% | 15.9% | 0.1% |
| orientation_penalty | -2.135488 | -14.7% | 14.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
