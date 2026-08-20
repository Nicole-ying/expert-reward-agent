# Training Feedback

## Final-policy outcome
score=235.523615, len=260.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[41.812479, 305.685722]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 26.526759 | 52.3% | 54.9% | 95.6% |
| landing_quality | 15.148330 | 29.9% | 29.9% | 21.8% |
| attitude_penalty | -4.514541 | -8.9% | 8.9% | 100.0% |
| engine_cost | -3.213000 | -6.3% | 6.3% | 61.7% |
| landing_velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
