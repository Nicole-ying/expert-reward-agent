# Training Feedback

## Final-policy outcome
score=144.172033, len=965.450000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[112.265517, 280.371299]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 312.464401 | 97.7% | 97.7% | 65.2% |
| orientation_stability_reward | -3.951835 | -1.2% | 1.2% | 100.0% |
| progress_reward | 1.985933 | 0.6% | 0.6% | 100.0% |
| velocity_damping_reward | -1.290861 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
