# Training Feedback

## Final-policy outcome
score=-39.647305, len=74.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-68.627101, -15.426479]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 30.000000 | 50.2% | 50.2% | 0.2% |
| progress | 22.006115 | 36.8% | 38.4% | 100.0% |
| survival | -5.940000 | -9.9% | 9.9% | 100.0% |
| stability | -0.363858 | -0.6% | 0.6% | 100.0% |
| fuel | -0.267000 | -0.4% | 0.4% | 36.0% |
| failure_penalty | -0.261144 | -0.4% | 0.4% | 3.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
