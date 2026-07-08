# Training Feedback

## Final-policy outcome
score=-33.418021, len=999.100000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-196.726569, 17.646217]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| engine_penalty | -49.742500 | -83.6% | 83.6% | 99.6% |
| progress_reward | 5.674868 | 9.5% | 11.5% | 100.0% |
| velocity_penalty | -2.425436 | -4.1% | 4.1% | 100.0% |
| attitude_penalty | -0.382036 | -0.6% | 0.6% | 100.0% |
| landing_proxy | 0.086142 | 0.1% | 0.1% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
