# Training Feedback

## Final-policy outcome
score=88.619535, len=908.950000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[16.011625, 267.138425]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 922.079418 | 85.6% | 85.6% | 14.1% |
| slow_bonus | 113.704663 | 10.6% | 10.6% | 100.0% |
| approach_reward | 16.397651 | 1.5% | 1.8% | 99.8% |
| fuel_penalty | -16.073000 | -1.5% | 1.5% | 88.4% |
| speed_penalty | -3.566617 | -0.3% | 0.3% | 100.0% |
| attitude_penalty | -1.502817 | -0.1% | 0.1% | 100.0% |
| descent_incentive | 0.828957 | 0.1% | 0.1% | 55.9% |
| centering_penalty | -0.539452 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
