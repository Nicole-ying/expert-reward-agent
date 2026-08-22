# Training Feedback

## Final-policy outcome
score=-1313.789807, len=757.450000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-2496.075443, -14.595903]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_gate | 592.025357 | 45.5% | 45.5% | 100.0% |
| uprightness_gate | 466.814019 | 35.9% | 35.9% | 63.3% |
| forward_reward | 156.960863 | 12.1% | 17.6% | 63.2% |
| joint_vel_penalty | -4.380601 | -0.3% | 0.3% | 58.3% |
| roll_vel_penalty | -2.233884 | -0.2% | 0.2% | 58.0% |
| action_penalty | -2.152889 | -0.2% | 0.2% | 100.0% |
| lateral_penalty | -2.079944 | -0.2% | 0.2% | 8.7% |
| pitch_vel_penalty | -1.685992 | -0.1% | 0.1% | 58.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
