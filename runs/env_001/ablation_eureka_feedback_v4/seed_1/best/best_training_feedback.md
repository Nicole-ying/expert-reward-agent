# Training Feedback

## Final-policy outcome
score=170.398522, len=471.500000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-108.519734, 270.509772]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 341.000000 | 94.6% | 94.6% | 7.2% |
| proximity_reward | -15.149586 | -4.2% | 4.2% | 100.0% |
| orientation_penalty | -3.603562 | -1.0% | 1.0% | 100.0% |
| velocity_penalty | -0.860967 | -0.2% | 0.2% | 98.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
