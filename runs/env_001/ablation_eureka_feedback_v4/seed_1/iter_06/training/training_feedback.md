# Training Feedback

## Final-policy outcome
score=34.671779, len=321.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.052726, 257.646824]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 398.000000 | 95.2% | 95.2% | 6.2% |
| proximity_reward | -16.973396 | -4.1% | 4.1% | 100.0% |
| orientation_penalty | -2.141405 | -0.5% | 0.5% | 100.0% |
| velocity_penalty | -0.962080 | -0.2% | 0.2% | 98.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
