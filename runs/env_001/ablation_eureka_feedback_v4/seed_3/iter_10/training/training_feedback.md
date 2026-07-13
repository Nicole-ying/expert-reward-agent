# Training Feedback

## Final-policy outcome
score=115.192166, len=845.400000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[41.592413, 278.595329]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_event | 803.734209 | 91.6% | 91.6% | 9.5% |
| soft_landing | 66.266770 | 7.5% | 7.5% | 87.0% |
| energy_penalty | -6.284000 | -0.7% | 0.7% | 74.3% |
| proximity | 1.425887 | 0.2% | 0.2% | 58.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
