# Training Feedback

## Final-policy outcome
score=98.815158, len=372.450000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-68.669016, 254.618698]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_stability | 2211.304918 | 100.0% | 100.0% | 74.4% |
| progress_gated | 0.417642 | 0.0% | 0.0% | 70.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
