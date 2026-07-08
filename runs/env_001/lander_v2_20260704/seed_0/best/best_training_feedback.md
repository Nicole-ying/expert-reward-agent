# Training Feedback

## Final-policy outcome
score=215.296113, len=577.650000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[112.547737, 286.800709]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 193.935000 | 78.7% | 78.7% | 57.6% |
| proximity_improvement | 39.325676 | 16.0% | 16.7% | 98.9% |
| velocity_penalty | -10.250978 | -4.2% | 4.2% | 99.9% |
| angle_penalty | -1.219389 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
