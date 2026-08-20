# Training Feedback

## Final-policy outcome
score=-66.969786, len=70.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-102.244138, -36.190638]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -67.062514 | -81.6% | 81.6% | 100.0% |
| velocity_penalty | -10.801944 | -13.1% | 13.1% | 99.9% |
| landing_proxy | 1.111407 | 1.4% | 4.7% | 100.0% |
| orientation_penalty | -0.494246 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
