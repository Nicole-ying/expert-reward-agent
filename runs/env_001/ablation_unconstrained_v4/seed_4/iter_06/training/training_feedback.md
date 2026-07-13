# Training Feedback

## Final-policy outcome
score=-78.744871, len=98.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.844196, -14.849896]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 20.919243 | 92.7% | 92.7% | 1.7% |
| progress | 1.338484 | 5.9% | 7.3% | 72.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
