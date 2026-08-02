# Training Feedback

## Final-policy outcome
score=-115.486496, len=131.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-209.105623, -1.353507]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_success_reward | 81.535503 | 66.2% | 66.2% | 0.6% |
| landing_gate | 21.118656 | 17.2% | 17.2% | 99.7% |
| progress | 11.947335 | 9.7% | 16.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
