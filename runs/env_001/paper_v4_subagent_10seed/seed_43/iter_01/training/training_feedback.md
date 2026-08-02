# Training Feedback

## Final-policy outcome
score=-115.296813, len=775.100000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-226.584766, 8.489686]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_penalty | 64.076064 | 84.7% | 84.7% | 100.0% |
| progress | 5.199023 | 6.9% | 15.3% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
