# Training Feedback

## Final-policy outcome
score=-115.516815, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.329319, -96.201424]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -9.284369 | -75.7% | 75.7% | 100.0% |
| shaping_reward | 2.151998 | 17.6% | 18.9% | 100.0% |
| contact_reward | 0.523461 | 4.3% | 4.3% | 0.7% |
| angle_penalty | -0.083709 | -0.7% | 0.7% | 100.0% |
| fuel_penalty | -0.055000 | -0.4% | 0.4% | 4.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
