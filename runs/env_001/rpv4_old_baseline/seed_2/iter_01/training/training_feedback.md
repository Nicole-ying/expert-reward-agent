# Training Feedback

## Final-policy outcome
score=-108.861604, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-122.877739, -92.920684]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -66.013367 | -59.6% | 59.6% | 98.1% |
| proximity_reward | -37.639882 | -34.0% | 34.0% | 100.0% |
| landing_bonus | 7.070155 | 6.4% | 6.4% | 1.8% |
| angle_penalty | -0.032399 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
