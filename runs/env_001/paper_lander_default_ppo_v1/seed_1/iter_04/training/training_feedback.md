# Training Feedback

## Final-policy outcome
score=-92.002425, len=68.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.040647, -62.208767]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 2.274556 | 55.2% | 55.2% | 91.5% |
| stability_penalty | -0.978505 | -23.7% | 23.7% | 100.0% |
| soft_landing_bonus | 0.867577 | 21.1% | 21.1% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
