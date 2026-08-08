# Training Feedback

## Final-policy outcome
score=67.712045, len=201.550000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-29.956133, 213.348711]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 288.609528 | 89.4% | 95.7% | 100.0% |
| upright_reward | -11.210514 | -3.5% | 3.5% | 100.0% |
| height_reward | -2.733672 | -0.8% | 0.8% | 8.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
