# Training Feedback

## Final-policy outcome
score=177.407477, len=652.250000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[59.355933, 286.482064]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -70.943143 | -59.3% | 59.3% | 10.9% |
| landing_quality | 40.616199 | 33.9% | 33.9% | 100.0% |
| angle_stability | -5.351721 | -4.5% | 4.5% | 100.0% |
| approach_progress | 2.740639 | 2.3% | 2.3% | 98.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
