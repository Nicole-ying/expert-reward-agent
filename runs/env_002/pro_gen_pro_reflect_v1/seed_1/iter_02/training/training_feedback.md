# Training Feedback

## Final-policy outcome
score=243.227157, len=803.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-25.790293, 312.533733]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 431.420090 | 99.8% | 99.8% | 99.8% |
| stability_penalty | -0.851873 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
