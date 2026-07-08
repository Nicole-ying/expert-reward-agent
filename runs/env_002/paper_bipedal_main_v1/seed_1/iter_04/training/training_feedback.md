# Training Feedback

## Final-policy outcome
score=294.435303, len=1084.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[87.006624, 307.543152]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 494.682016 | 80.2% | 80.2% | 100.0% |
| energy_penalty | -72.428542 | -11.7% | 11.7% | 100.0% |
| stability_penalty | -31.085154 | -5.0% | 5.0% | 100.0% |
| gait_reward | 18.745500 | 3.0% | 3.0% | 57.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
