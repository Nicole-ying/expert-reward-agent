# Training Feedback

## Final-policy outcome
score=210.366096, len=655.550000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[110.960134, 290.931897]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 42.970001 | 48.7% | 48.7% | 65.5% |
| landing_approach | 28.015063 | 31.7% | 31.7% | 100.0% |
| progress | 13.322153 | 15.1% | 16.0% | 99.0% |
| stability_penalty | -1.874574 | -2.1% | 2.1% | 100.0% |
| lateral_drift_penalty | -1.297326 | -1.5% | 1.5% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
