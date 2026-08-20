# Training Feedback

## Final-policy outcome
score=-21.815132, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-54.105859, 12.974534]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_and_soft_landing | 340.062689 | 99.9% | 99.9% | 100.0% |
| upright_stabilization | -0.342332 | -0.1% | 0.1% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
