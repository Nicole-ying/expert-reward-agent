# Training Feedback

## Final-policy outcome
score=165.623437, len=979.000000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[126.588596, 241.536025]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 325.849044 | 98.5% | 98.5% | 70.8% |
| progress_reward | 2.791908 | 0.8% | 0.9% | 99.8% |
| velocity_penalty | 1.642290 | 0.5% | 0.5% | 99.8% |
| angle_penalty | 0.428463 | 0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
