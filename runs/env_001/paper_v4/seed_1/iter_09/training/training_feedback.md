# Training Feedback

## Final-policy outcome
score=168.339342, len=657.000000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[22.862446, 255.489586]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 892.811968 | 99.4% | 99.4% | 100.0% |
| shaped_progress | 4.446378 | 0.5% | 0.6% | 99.9% |
| angular_vel_penalty | -0.297539 | -0.0% | 0.0% | 79.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
