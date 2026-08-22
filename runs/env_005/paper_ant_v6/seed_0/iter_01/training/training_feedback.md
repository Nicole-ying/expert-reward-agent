# Training Feedback

## Final-policy outcome
score=-2.503153, len=15.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-31.279649, 39.998365]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 27.621533 | 68.6% | 68.6% | 72.9% |
| lateral_penalty | -6.520053 | -16.2% | 16.2% | 100.0% |
| posture_penalty | -5.644006 | -14.0% | 14.0% | 100.0% |
| height_penalty | -0.497614 | -1.2% | 1.2% | 6.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
