# Training Feedback

## Final-policy outcome
score=248.710617, len=343.050000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[122.259898, 300.447991]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 11.975929 | 69.1% | 69.1% | 28.0% |
| progress | 3.275544 | 18.9% | 20.4% | 96.1% |
| orientation_penalty | -1.586832 | -9.2% | 9.2% | 100.0% |
| velocity_penalty | -0.245900 | -1.4% | 1.4% | 95.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
