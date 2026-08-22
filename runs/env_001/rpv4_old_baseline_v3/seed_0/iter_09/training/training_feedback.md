# Training Feedback

## Final-policy outcome
score=35.633589, len=574.250000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-223.136820, 243.361624]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 142.827770 | 43.4% | 43.4% | 5.5% |
| contact_bonus | 131.000000 | 39.8% | 39.8% | 5.9% |
| proximity | 19.754417 | 6.0% | 6.0% | 100.0% |
| fuel_cost | -13.909500 | -4.2% | 4.2% | 80.7% |
| vel_cost | -10.891025 | -3.3% | 3.3% | 99.9% |
| approach_reward | 4.179364 | 1.3% | 2.0% | 98.8% |
| descent_reward | 2.974024 | 0.9% | 0.9% | 75.8% |
| orientation_cost | -1.078823 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
