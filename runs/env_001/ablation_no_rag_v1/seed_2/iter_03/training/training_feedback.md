# Training Feedback

## Final-policy outcome
score=-82.728679, len=858.200000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-214.630059, 153.094025]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_reward | 109.465266 | 98.7% | 98.7% | 6.9% |
| progress_reward | 0.708042 | 0.6% | 1.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
