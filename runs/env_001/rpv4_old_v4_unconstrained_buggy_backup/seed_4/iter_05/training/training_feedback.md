# Training Feedback

## Final-policy outcome
score=82.657106, len=961.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[38.817958, 254.357664]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 478.885087 | 91.4% | 91.4% | 58.4% |
| fuel_efficiency | -16.294500 | -3.1% | 3.1% | 56.5% |
| attitude_constraint | -12.265481 | -2.3% | 2.3% | 100.0% |
| goal_proximity | -10.262364 | -2.0% | 2.0% | 100.0% |
| settling_stability | -6.315288 | -1.2% | 1.2% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
