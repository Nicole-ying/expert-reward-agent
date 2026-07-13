# Training Feedback

## Final-policy outcome
score=-110.844022, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.542282, -95.157694]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -66.440824 | -84.0% | 84.0% | 100.0% |
| velocity_penalty | -11.395084 | -14.4% | 14.4% | 99.6% |
| landing_proxy | 0.744629 | 0.9% | 0.9% | 1.2% |
| orientation_penalty | -0.522102 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
