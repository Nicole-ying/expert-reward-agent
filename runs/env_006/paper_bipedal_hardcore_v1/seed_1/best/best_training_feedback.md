# Training Feedback

## Final-policy outcome
score=-18.010346, len=411.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-77.428639, 146.486930]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 145.031629 | 94.3% | 94.8% | 100.0% |
| action_efficiency_penalty | -4.809344 | -3.1% | 3.1% | 100.0% |
| stability_penalty | -3.116211 | -2.0% | 2.0% | 19.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
