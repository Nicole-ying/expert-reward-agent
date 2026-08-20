# Training Feedback

## Final-policy outcome
score=-32.402309, len=89.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-194.638565, 23.862791]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 85.431982 | 77.3% | 97.6% | 100.0% |
| upright | -1.950329 | -1.8% | 1.8% | 100.0% |
| height | -0.672680 | -0.6% | 0.6% | 3.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
