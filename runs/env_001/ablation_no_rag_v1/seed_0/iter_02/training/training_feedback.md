# Training Feedback

## Final-policy outcome
score=255.536242, len=345.550000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[172.224289, 292.453207]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 256.183906 | 92.8% | 92.8% | 100.0% |
| tilt_penalty | -10.056788 | -3.6% | 3.6% | 100.0% |
| velocity_penalty | -8.253758 | -3.0% | 3.0% | 99.7% |
| rotation_penalty | -1.450491 | -0.5% | 0.5% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
