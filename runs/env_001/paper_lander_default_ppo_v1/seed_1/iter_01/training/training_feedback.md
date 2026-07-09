# Training Feedback

## Final-policy outcome
score=-108.316585, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.883170, -78.636102]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -9.457480 | -78.9% | 78.9% | 100.0% |
| progress_reward | 2.280888 | 19.0% | 19.0% | 91.9% |
| soft_landing_bonus | 0.250000 | 2.1% | 2.1% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
