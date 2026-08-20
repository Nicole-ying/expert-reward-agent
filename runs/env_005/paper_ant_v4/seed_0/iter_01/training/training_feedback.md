# Training Feedback

## Final-policy outcome
score=-27.217357, len=409.800000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-802.747355, 216.903026]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_reward | 104.500965 | 75.9% | 82.3% | 37.9% |
| angular_penalty | 17.247833 | 12.5% | 12.5% | 26.7% |
| lateral_penalty | 6.380383 | 4.6% | 4.6% | 26.8% |
| height_penalty | 0.591622 | 0.4% | 0.4% | 100.0% |
| upright_penalty | 0.117791 | 0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
