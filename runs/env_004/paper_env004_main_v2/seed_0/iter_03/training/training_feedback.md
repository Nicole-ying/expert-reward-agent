# Training Feedback

## Final-policy outcome
score=2509.866253, len=828.050000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[1804.090297, 3098.838425]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_posture_reward | 1635.404043 | 99.0% | 99.0% | 100.0% |
| ang_vel_penalty | 15.779193 | 1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
