# Training Feedback

## Final-policy outcome
score=117.154078, len=942.100000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[79.555945, 225.709054]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 396.861483 | 97.9% | 97.9% | 42.6% |
| progress_reward | 3.951547 | 1.0% | 1.0% | 100.0% |
| orientation_stability_reward | -3.478453 | -0.9% | 0.9% | 100.0% |
| velocity_damping_reward | -0.970393 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
