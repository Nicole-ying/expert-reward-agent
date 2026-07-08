# Training Feedback

## Final-policy outcome
score=-108.904317, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.435133, -90.378464]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -9.362558 | -87.7% | 87.7% | 100.0% |
| progress_delta | 1.120462 | 10.5% | 10.9% | 100.0% |
| soft_landing_proxy | 0.150000 | 1.4% | 1.4% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
