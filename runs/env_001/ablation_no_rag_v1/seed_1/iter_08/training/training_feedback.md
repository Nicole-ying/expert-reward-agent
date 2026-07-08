# Training Feedback

## Final-policy outcome
score=78.233789, len=328.100000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-84.803943, 243.019391]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_guidance | 216.183075 | 96.2% | 97.9% | 100.0% |
| tilt_penalty | -2.388357 | -1.1% | 1.1% | 100.0% |
| distance_progress | 1.596376 | 0.7% | 1.0% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
