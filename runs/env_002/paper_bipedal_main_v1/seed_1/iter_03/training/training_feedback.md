# Training Feedback

## Final-policy outcome
score=313.730594, len=1087.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[312.671026, 314.763825]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.368648 | 80.2% | 80.3% | 100.0% |
| energy_penalty | -57.686055 | -9.2% | 9.2% | 100.0% |
| stability_penalty | -43.861611 | -7.0% | 7.0% | 100.0% |
| gait_reward | 22.476000 | 3.6% | 3.6% | 68.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
