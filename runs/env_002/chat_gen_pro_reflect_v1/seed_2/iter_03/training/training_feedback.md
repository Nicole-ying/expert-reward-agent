# Training Feedback

## Final-policy outcome
score=214.488907, len=626.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[21.816173, 332.575032]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 585.289287 | 44.1% | 44.1% | 100.0% |
| forward_progress_reward | 368.999919 | 27.8% | 27.8% | 100.0% |
| gated_forward_reward | 344.982755 | 26.0% | 26.0% | 100.0% |
| energy_penalty | -27.730234 | -2.1% | 2.1% | 100.0% |
| balance_angle_penalty | -0.006485 | -0.0% | 0.0% | 0.2% |
| balance_angular_vel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
