# Training Feedback

## Final-policy outcome
score=-80.853856, len=103.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.190127, 16.648258]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_soft_reward | 4.059274 | 54.9% | 54.9% | 1.1% |
| safety_penalty | -1.475376 | -19.9% | 19.9% | 9.5% |
| progress | 0.936338 | 12.7% | 15.6% | 100.0% |
| action_cost | -0.483500 | -6.5% | 6.5% | 46.7% |
| angle_penalty | -0.228760 | -3.1% | 3.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
