# Training Feedback

## Final-policy outcome
score=-6.632975, len=616.500000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-1457.285948, 685.688881]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1888.595534 | 85.5% | 87.1% | 94.6% |
| upright_reward | 173.616354 | 7.9% | 12.7% | 100.0% |
| height_hinge_penalty | -2.570377 | -0.1% | 0.1% | 0.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
