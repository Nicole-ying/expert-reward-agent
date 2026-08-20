# Training Feedback

## Final-policy outcome
score=40.020879, len=850.350000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-42.012992, 237.852795]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 77.367006 | 36.1% | 36.1% | 5.4% |
| contact_bonus | 58.500000 | 27.3% | 27.3% | 8.4% |
| proximity | 32.951288 | 15.4% | 15.4% | 100.0% |
| fuel_cost | -25.110000 | -11.7% | 11.7% | 98.4% |
| vel_cost | -10.518180 | -4.9% | 4.9% | 100.0% |
| approach_reward | 5.951207 | 2.8% | 3.3% | 100.0% |
| orientation_cost | -2.621655 | -1.2% | 1.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
