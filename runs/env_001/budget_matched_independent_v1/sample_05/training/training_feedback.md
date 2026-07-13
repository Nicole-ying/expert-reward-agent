# Training Feedback

## Final-policy outcome
score=-166.959934, len=351.650000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-225.944711, -116.690073]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.147697 | 13.3% | 78.8% | 100.0% |
| orientation_penalty | -0.236183 | -21.2% | 21.2% | 100.0% |
| soft_land_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
