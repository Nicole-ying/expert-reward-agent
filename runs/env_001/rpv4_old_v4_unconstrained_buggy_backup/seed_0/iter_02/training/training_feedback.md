# Training Feedback

## Final-policy outcome
score=-99.872139, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-131.358588, -57.127609]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 3792.866119 | 99.3% | 99.3% | 100.0% |
| vel_damping | -25.982364 | -0.7% | 0.7% | 100.0% |
| progress_reward | 0.398749 | 0.0% | 0.0% | 100.0% |
| orientation_penalty | -0.409954 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
