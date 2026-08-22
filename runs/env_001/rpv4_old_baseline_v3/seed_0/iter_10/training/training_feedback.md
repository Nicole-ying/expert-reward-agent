# Training Feedback

## Final-policy outcome
score=176.138547, len=424.600000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-22.541096, 269.760935]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 111.176726 | 47.1% | 47.1% | 15.3% |
| contact_bonus | 69.875000 | 29.6% | 29.6% | 17.6% |
| proximity | 15.657828 | 6.6% | 6.6% | 100.0% |
| approach | 12.140287 | 5.1% | 5.5% | 97.9% |
| vel_cost | -8.786674 | -3.7% | 3.7% | 99.8% |
| fuel_cost | -7.588500 | -3.2% | 3.2% | 59.6% |
| aligned_descent | 7.310724 | 3.1% | 3.1% | 86.2% |
| orientation_cost | -2.565920 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
