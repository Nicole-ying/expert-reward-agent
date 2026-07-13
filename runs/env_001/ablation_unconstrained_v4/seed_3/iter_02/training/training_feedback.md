# Training Feedback

## Final-policy outcome
score=-63.805763, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-97.520066, -25.163108]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_reward | 283.190755 | 54.6% | 54.6% | 100.0% |
| gated_goal | 185.916275 | 35.9% | 35.9% | 100.0% |
| engine_penalty | -49.310000 | -9.5% | 9.5% | 98.6% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
