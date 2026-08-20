# Training Feedback

## Final-policy outcome
score=-85.932872, len=71.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-136.391192, -28.364158]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 4.253808 | 50.9% | 50.9% | 2.2% |
| orientation_penalty | -2.748167 | -32.9% | 32.9% | 23.9% |
| fuel_penalty | -0.785000 | -9.4% | 9.4% | 22.1% |
| progress | 0.539318 | 6.5% | 6.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 16/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
