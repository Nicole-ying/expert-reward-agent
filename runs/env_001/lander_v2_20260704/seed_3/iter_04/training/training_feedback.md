# Training Feedback

## Final-policy outcome
score=136.731955, len=237.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-77.201837, 271.577244]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 114.790378 | 96.5% | 96.5% | 11.4% |
| approach_reward | 3.768758 | 3.2% | 3.3% | 96.2% |
| stability_penalty | -0.276755 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
