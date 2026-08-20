# Training Feedback

## Final-policy outcome
score=329.813461, len=865.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[329.348038, 330.230989]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 820.972003 | 45.0% | 45.0% | 100.0% |
| forward_reward | 503.343168 | 27.6% | 27.6% | 100.0% |
| gated_forward_reward | 477.644939 | 26.2% | 26.2% | 100.0% |
| action_penalty | -20.804661 | -1.1% | 1.1% | 100.0% |
| angular_vel_penalty | -0.017378 | -0.0% | 0.0% | 99.9% |
| balance_penalty | -0.017378 | -0.0% | 0.0% | 99.9% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
