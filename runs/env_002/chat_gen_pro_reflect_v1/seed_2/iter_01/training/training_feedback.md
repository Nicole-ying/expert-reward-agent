# Training Feedback

## Final-policy outcome
score=286.889201, len=1024.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[141.281789, 301.972733]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 956.371091 | 65.7% | 65.7% | 100.0% |
| forward_progress_reward | 257.650163 | 17.7% | 17.7% | 100.0% |
| gated_forward_reward | 242.564517 | 16.7% | 16.7% | 100.0% |
| balance_angle_penalty | -0.025355 | -0.0% | 0.0% | 0.1% |
| balance_angular_vel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
