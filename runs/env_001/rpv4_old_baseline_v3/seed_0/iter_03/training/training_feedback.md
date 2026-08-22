# Training Feedback

## Final-policy outcome
score=-45.143561, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.943309, -0.360620]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 1175.046021 | 89.0% | 89.0% | 100.0% |
| vel_penalty | -86.206726 | -6.5% | 6.5% | 100.0% |
| fuel_cost | -29.739000 | -2.3% | 2.3% | 99.1% |
| approach_reward | 18.935215 | 1.4% | 1.8% | 100.0% |
| landing_quality | 2.535736 | 0.2% | 0.2% | 0.2% |
| contact_bonus | 2.025000 | 0.2% | 0.2% | 0.2% |
| orientation_cost | -0.943085 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
