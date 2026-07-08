# Training Feedback

## Final-policy outcome
score=-47.114451, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.096485, -7.464344]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_reward | 4379.958540 | 100.0% | 100.0% | 100.0% |
| progress_reward | 0.930532 | 0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
