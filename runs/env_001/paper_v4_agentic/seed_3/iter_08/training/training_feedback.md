# Training Feedback

## Final-policy outcome
score=263.999108, len=314.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[235.129129, 295.250355]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 27.780855 | 56.7% | 57.8% | 95.4% |
| landing_quality | 12.470364 | 25.5% | 25.5% | 13.7% |
| engine_cost | -4.522000 | -9.2% | 9.2% | 72.0% |
| attitude_penalty | -3.645440 | -7.4% | 7.4% | 100.0% |
| landing_velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
