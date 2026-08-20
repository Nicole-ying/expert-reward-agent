# Training Feedback

## Final-policy outcome
score=283.868350, len=906.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-49.733471, 319.687114]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 830.791090 | 42.6% | 42.6% | 100.0% |
| forward_reward | 567.634816 | 29.1% | 29.1% | 100.0% |
| gated_forward_reward | 521.232098 | 26.7% | 26.7% | 100.0% |
| action_penalty | -28.967650 | -1.5% | 1.5% | 100.0% |
| balance_penalty | -0.253752 | -0.0% | 0.0% | 100.0% |
| angle_penalty | -0.205921 | -0.0% | 0.0% | 0.2% |
| angular_vel_penalty | -0.047831 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
