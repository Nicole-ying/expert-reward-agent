# Training Feedback

## Final-policy outcome
score=32.829837, len=704.200000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-39.170786, 235.957396]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_guidance | 20.137015 | 67.8% | 82.4% | 100.0% |
| tilt_penalty | -3.812883 | -12.8% | 12.8% | 100.0% |
| distance_progress | 1.169573 | 3.9% | 4.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
