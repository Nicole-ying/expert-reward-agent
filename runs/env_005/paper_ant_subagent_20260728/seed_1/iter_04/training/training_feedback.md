# Training Feedback

## Final-policy outcome
score=69.243129, len=989.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-1067.235042, 393.773006]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 1458.913070 | 58.2% | 59.0% | 100.0% |
| height_factor | 933.736552 | 37.2% | 37.2% | 100.0% |
| upright_bonus | 68.045612 | 2.7% | 3.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
