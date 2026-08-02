# Training Feedback

## Final-policy outcome
score=-178.948706, len=915.450000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-212.525185, -114.986480]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 1126.281054 | 97.3% | 97.3% | 100.0% |
| lateral_pos_penalty | -26.089827 | -2.3% | 2.3% | 100.0% |
| progress_gated | -4.212283 | -0.4% | 0.5% | 100.0% |
| angvel_penalty | -0.026130 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
