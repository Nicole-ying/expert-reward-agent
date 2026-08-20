# Training Feedback

## Final-policy outcome
score=-24.032875, len=148.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-118.838826, 261.583534]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settlement_bonus | 25.240861 | 41.7% | 80.9% | 37.4% |
| speed_penalty_gated | -8.926664 | -14.8% | 14.8% | 99.6% |
| proximity_reward | 2.081760 | 3.4% | 3.7% | 99.6% |
| orientation_penalty | -0.353664 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
