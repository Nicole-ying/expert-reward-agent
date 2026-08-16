# Training Feedback

## Final-policy outcome
score=-39.065460, len=940.050000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-167.581926, 118.393827]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_potential | 149.379634 | 60.3% | 60.3% | 99.8% |
| contact_reward | 42.788019 | 17.3% | 17.3% | 1.2% |
| approach_reward | 21.748194 | 8.8% | 11.6% | 99.9% |
| fuel_penalty | -23.527500 | -9.5% | 9.5% | 83.4% |
| velocity_penalty | -2.411060 | -1.0% | 1.0% | 99.9% |
| angle_penalty | -1.055468 | -0.4% | 0.4% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| perfect_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
