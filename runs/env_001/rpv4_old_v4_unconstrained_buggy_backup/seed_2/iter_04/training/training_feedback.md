# Training Feedback

## Final-policy outcome
score=-40.246879, len=74.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-83.913030, -17.708881]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 5.500235 | 85.2% | 89.3% | 100.0% |
| velocity_penalty | -0.444089 | -6.9% | 6.9% | 99.9% |
| orientation_penalty | -0.212755 | -3.3% | 3.3% | 100.0% |
| contact_guidance | 0.036393 | 0.6% | 0.6% | 0.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
