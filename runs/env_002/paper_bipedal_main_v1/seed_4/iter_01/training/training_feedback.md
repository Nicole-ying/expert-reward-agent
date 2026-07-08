# Training Feedback

## Final-policy outcome
score=103.027232, len=787.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-80.357059, 287.319037]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 319.959350 | 94.0% | 94.4% | 100.0% |
| stability_penalty | -19.094838 | -5.6% | 5.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
