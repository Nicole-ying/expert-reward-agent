# Training Feedback

## Final-policy outcome
score=-373.711267, len=150.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-455.869985, -206.162474]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | -0.102557 | -12.2% | 82.9% | 100.0% |
| stability_penalty | -0.143540 | -17.1% | 17.1% | 18.3% |
| soft_landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
