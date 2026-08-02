# Training Feedback

## Final-policy outcome
score=-111.811408, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-127.801424, -92.011505]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.121431 | 93.1% | 96.3% | 100.0% |
| soft_landing | 0.043550 | 3.6% | 3.6% | 0.7% |
| angle_penalty | -0.000729 | -0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
