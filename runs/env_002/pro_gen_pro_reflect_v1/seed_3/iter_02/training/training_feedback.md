# Training Feedback

## Final-policy outcome
score=284.141664, len=885.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[24.020388, 311.154853]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 471.509274 | 95.3% | 95.3% | 100.0% |
| action_cost | -15.938158 | -3.2% | 3.2% | 100.0% |
| tilt_penalty | -6.837885 | -1.4% | 1.4% | 100.0% |
| angular_velocity_penalty | -0.333029 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
