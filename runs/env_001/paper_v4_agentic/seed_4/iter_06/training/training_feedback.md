# Training Feedback

## Final-policy outcome
score=88.063850, len=982.050000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[34.000325, 246.430802]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_prep | 890.144194 | 57.5% | 57.5% | 100.0% |
| progress_gated | 462.426357 | 29.9% | 29.9% | 73.4% |
| fuel_penalty | -196.410000 | -12.7% | 12.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
