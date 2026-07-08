# Training Feedback

## Final-policy outcome
score=20.707070, len=913.050000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-330.786399, 103.105350]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality_reward | 702.837310 | 96.9% | 96.9% | 41.3% |
| progress_reward | 13.160338 | 1.8% | 2.2% | 100.0% |
| stability_penalty | -6.646973 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
