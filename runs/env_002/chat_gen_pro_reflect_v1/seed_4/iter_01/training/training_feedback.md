# Training Feedback

## Final-policy outcome
score=166.539950, len=820.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-43.184163, 295.075255]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 791.718914 | 51.3% | 51.3% | 100.0% |
| forward_reward | 381.440275 | 24.7% | 24.7% | 100.0% |
| gated_forward | 367.604345 | 23.8% | 23.8% | 100.0% |
| balance_penalty | -1.686407 | -0.1% | 0.1% | 100.0% |
| angle_penalty | -1.655801 | -0.1% | 0.1% | 100.0% |
| angular_vel_penalty | -0.030605 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
