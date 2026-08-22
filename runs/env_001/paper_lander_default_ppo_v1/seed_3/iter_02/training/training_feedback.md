# Training Feedback

## Final-policy outcome
score=150.837305, len=220.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-328.943458, 310.556873]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 328.416914 | 83.2% | 83.2% | 100.0% |
| contact_bonus | 59.446282 | 15.1% | 15.1% | 14.0% |
| velocity_penalty | -4.531404 | -1.1% | 1.1% | 95.2% |
| angle_penalty | -1.680596 | -0.4% | 0.4% | 100.0% |
| angular_velocity_penalty | -0.627682 | -0.2% | 0.2% | 94.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
