# Training Feedback

## Final-policy outcome
score=30.056993, len=668.350000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-220.599147, 235.983745]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 280.196212 | 89.9% | 89.9% | 12.8% |
| lateral_pos_penalty | -18.143868 | -5.8% | 5.8% | 100.0% |
| progress_gated | 7.726084 | 2.5% | 4.3% | 99.3% |
| angvel_penalty | -0.127459 | -0.0% | 0.0% | 98.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
