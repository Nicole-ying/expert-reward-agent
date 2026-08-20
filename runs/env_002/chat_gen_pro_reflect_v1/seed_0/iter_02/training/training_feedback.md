# Training Feedback

## Final-policy outcome
score=306.040534, len=818.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[304.506127, 307.301880]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 762.867135 | 43.9% | 43.9% | 100.0% |
| forward_reward | 503.510380 | 29.0% | 29.0% | 100.0% |
| gated_forward_reward | 469.005862 | 27.0% | 27.0% | 100.0% |
| angular_vel_penalty | -0.047401 | -0.0% | 0.0% | 100.0% |
| balance_penalty | -0.047401 | -0.0% | 0.0% | 100.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
