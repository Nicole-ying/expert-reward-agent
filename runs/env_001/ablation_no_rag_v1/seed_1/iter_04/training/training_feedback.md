# Training Feedback

## Final-policy outcome
score=18.389246, len=321.600000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-110.310731, 237.775266]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_support | 116.100000 | 97.5% | 97.5% | 18.1% |
| tilt_penalty | -1.790227 | -1.5% | 1.5% | 100.0% |
| distance_progress | 0.772128 | 0.6% | 1.0% | 99.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
