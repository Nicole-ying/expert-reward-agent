# Training Feedback

## Final-policy outcome
score=-10.506030, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-45.136736, 21.242936]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 32.621106 | 84.4% | 98.3% | 100.0% |
| stability_penalty | -0.648516 | -1.7% | 1.7% | 100.0% |
| soft_landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
