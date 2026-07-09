# Training Feedback

## Final-policy outcome
score=-107.992451, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.761502, -84.828353]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 2.279767 | 56.4% | 56.4% | 92.0% |
| stability_penalty | -0.948808 | -23.5% | 23.5% | 100.0% |
| soft_landing_bonus | 0.816565 | 20.2% | 20.2% | 0.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
