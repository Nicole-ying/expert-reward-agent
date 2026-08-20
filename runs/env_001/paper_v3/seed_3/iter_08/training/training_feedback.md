# Training Feedback

## Final-policy outcome
score=198.234770, len=617.050000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[84.519905, 241.370215]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_proximity | 463.902777 | 79.4% | 79.4% | 100.0% |
| contact_reward | 117.450000 | 20.1% | 20.1% | 6.3% |
| soft_landing_velocity | -2.252747 | -0.4% | 0.4% | 97.8% |
| upright_stabilization | -0.420633 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
