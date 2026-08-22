# Training Feedback

## Final-policy outcome
score=178.696817, len=404.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-97.916855, 256.588882]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 285.600000 | 54.0% | 54.0% | 21.9% |
| goal_proximity | 193.003749 | 36.5% | 36.5% | 100.0% |
| settling_stability | -23.303586 | -4.4% | 4.4% | 99.8% |
| fuel_efficiency | -16.750000 | -3.2% | 3.2% | 82.8% |
| attitude_constraint | -10.678825 | -2.0% | 2.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
