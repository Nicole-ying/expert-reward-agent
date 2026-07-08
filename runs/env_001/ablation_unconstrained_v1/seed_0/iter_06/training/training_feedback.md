# Training Feedback

## Final-policy outcome
score=-64.501673, len=951.000000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-146.576931, -4.212043]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | 649.250327 | 66.3% | 66.3% | 100.0% |
| landing_proxy | 323.740209 | 33.1% | 33.1% | 90.0% |
| stability_penalty | -5.867505 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
