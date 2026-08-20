# Training Feedback

## Final-policy outcome
score=-117.497950, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.339809, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 35.000000 | 59.6% | 59.6% | 0.5% |
| dist_reward | 11.169318 | 19.0% | 19.7% | 100.0% |
| vel_penalty | -9.183137 | -15.6% | 15.6% | 100.0% |
| step_penalty | -1.367000 | -2.3% | 2.3% | 100.0% |
| touch_bonus | 1.220000 | 2.1% | 2.1% | 3.0% |
| angle_penalty | -0.398954 | -0.7% | 0.7% | 100.0% |
| fuel_penalty | -0.033000 | -0.1% | 0.1% | 4.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
