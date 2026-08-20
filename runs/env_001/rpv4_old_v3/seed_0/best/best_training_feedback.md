# Training Feedback

## Final-policy outcome
score=217.212007, len=484.100000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[61.623931, 265.546415]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 36.450000 | 90.6% | 90.6% | 7.5% |
| goal_proximity | 2.567816 | 6.4% | 7.0% | 97.6% |
| living_penalty | -0.968200 | -2.4% | 2.4% | 100.0% |
| crash_prevention_x | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_prevention_y | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
