# Training Feedback

## Final-policy outcome
score=294.256016, len=1111.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[129.026273, 304.785476]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 442.901384 | 99.0% | 99.0% | 99.4% |
| posture_penalty | -2.951836 | -0.7% | 0.7% | 100.0% |
| action_penalty | -1.490680 | -0.3% | 0.3% | 100.0% |
| angular_vel_penalty | -0.012714 | -0.0% | 0.0% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
