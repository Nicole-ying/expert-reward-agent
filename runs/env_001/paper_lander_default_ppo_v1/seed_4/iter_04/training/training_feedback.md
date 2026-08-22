# Training Feedback

## Final-policy outcome
score=168.769480, len=517.550000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-32.608815, 284.052030]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| event_bonus | 411.500000 | 66.1% | 66.1% | 8.0% |
| soft_landing_proxy | 159.842110 | 25.7% | 25.7% | 49.3% |
| direction_reward | 27.402744 | 4.4% | 6.4% | 100.0% |
| attitude_penalty | -7.901572 | -1.3% | 1.3% | 100.0% |
| fuel_penalty | -3.597500 | -0.6% | 0.6% | 69.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
