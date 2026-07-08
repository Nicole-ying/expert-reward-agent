# Training Feedback

## Final-policy outcome
score=233.821081, len=453.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[198.465418, 269.234751]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 335.792706 | 88.6% | 88.6% | 100.0% |
| fuel_penalty | -16.672500 | -4.4% | 4.4% | 73.5% |
| tilt_penalty | -15.672364 | -4.1% | 4.1% | 100.0% |
| velocity_penalty | -9.340148 | -2.5% | 2.5% | 99.8% |
| rotation_penalty | -1.471558 | -0.4% | 0.4% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
