# Training Feedback

## Final-policy outcome
score=-112.963689, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746612, -95.059093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -66.484716 | -87.9% | 87.9% | 100.0% |
| stability_penalty | -8.521020 | -11.3% | 11.3% | 100.0% |
| landing_quality | 0.636913 | 0.8% | 0.8% | 42.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
