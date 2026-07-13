# Training Feedback

## Final-policy outcome
score=24.613390, len=884.350000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-163.148255, 241.385488]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_success | 755.000000 | 88.7% | 88.7% | 8.5% |
| landing_quality | 64.434014 | 7.6% | 7.6% | 100.0% |
| velocity_penalty | -22.653619 | -2.7% | 2.7% | 20.9% |
| angle_stability | -6.138820 | -0.7% | 0.7% | 100.0% |
| approach_progress | 2.190013 | 0.3% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
