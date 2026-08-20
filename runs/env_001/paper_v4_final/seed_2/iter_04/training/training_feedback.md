# Training Feedback

## Final-policy outcome
score=21.667461, len=974.100000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-52.728246, 166.524437]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 741.667998 | 94.9% | 94.9% | 100.0% |
| pose_penalty | -29.803183 | -3.8% | 3.8% | 100.0% |
| progress | -10.458987 | -1.3% | 1.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
