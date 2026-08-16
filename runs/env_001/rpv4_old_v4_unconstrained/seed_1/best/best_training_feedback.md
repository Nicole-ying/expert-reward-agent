# Training Feedback

## Final-policy outcome
score=234.887724, len=453.700000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[150.809826, 282.347116]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| slow_bonus | 259.723753 | 82.9% | 82.9% | 100.0% |
| approach_reward | 16.160866 | 5.2% | 5.3% | 97.8% |
| landing_reward | 15.589665 | 5.0% | 5.0% | 0.4% |
| speed_penalty | -9.792360 | -3.1% | 3.1% | 99.8% |
| fuel_penalty | -6.480000 | -2.1% | 2.1% | 71.4% |
| contact_loss_penalty | -2.800000 | -0.9% | 0.9% | 0.1% |
| attitude_penalty | -1.150950 | -0.4% | 0.4% | 100.0% |
| descent_incentive | 0.712589 | 0.2% | 0.2% | 61.6% |
| centering_penalty | -0.420942 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
