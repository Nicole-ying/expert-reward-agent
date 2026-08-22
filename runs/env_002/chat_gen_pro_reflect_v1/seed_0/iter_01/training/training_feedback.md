# Training Feedback

## Final-policy outcome
score=231.235900, len=794.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-8.630315, 304.936297]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 739.805773 | 46.9% | 46.9% | 100.0% |
| forward_reward | 433.975669 | 27.5% | 27.5% | 100.0% |
| gated_forward_reward | 403.783431 | 25.6% | 25.6% | 100.0% |
| balance_penalty | -0.059792 | -0.0% | 0.0% | 99.9% |
| angular_vel_penalty | -0.034990 | -0.0% | 0.0% | 99.9% |
| angle_penalty | -0.024802 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
