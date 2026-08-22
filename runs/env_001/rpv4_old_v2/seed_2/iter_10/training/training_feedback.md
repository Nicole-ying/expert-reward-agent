# Training Feedback

## Final-policy outcome
score=-36.066300, len=961.850000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-137.566803, 13.429050]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| thrust_penalty | -48.015000 | -80.2% | 80.2% | 99.8% |
| goal_distance_progress | 5.549577 | 9.3% | 11.9% | 100.0% |
| soft_landing | -2.855812 | -4.8% | 4.8% | 100.0% |
| orientation_penalty | -1.838055 | -3.1% | 3.1% | 100.0% |
| landing_contact_success | 0.063978 | 0.1% | 0.1% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
