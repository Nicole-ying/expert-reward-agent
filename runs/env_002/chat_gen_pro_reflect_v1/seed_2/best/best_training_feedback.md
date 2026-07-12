# Training Feedback

## Final-policy outcome
score=304.498984, len=885.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[301.215542, 307.036430]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 837.271620 | 52.4% | 52.4% | 100.0% |
| forward_progress_reward | 390.622652 | 24.5% | 24.5% | 100.0% |
| gated_forward_reward | 369.146573 | 23.1% | 23.1% | 100.0% |
| balance_angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| balance_angular_vel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
