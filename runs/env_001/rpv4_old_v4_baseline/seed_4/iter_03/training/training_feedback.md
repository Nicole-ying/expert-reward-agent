# Training Feedback

## Final-policy outcome
score=88.521717, len=892.550000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[14.625518, 267.349190]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 3090.000000 | 80.0% | 80.0% | 11.9% |
| goal_proximity | 675.197383 | 17.5% | 17.5% | 100.0% |
| settling_stability | -46.442001 | -1.2% | 1.2% | 100.0% |
| fuel_efficiency | -27.457500 | -0.7% | 0.7% | 61.5% |
| attitude_constraint | -21.251541 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
