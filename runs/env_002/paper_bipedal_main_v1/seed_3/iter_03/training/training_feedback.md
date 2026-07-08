# Training Feedback

## Final-policy outcome
score=268.707813, len=1438.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[265.017844, 271.774768]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.356688 | 82.5% | 82.5% | 100.0% |
| energy_penalty | -75.856378 | -12.4% | 12.4% | 100.0% |
| angle_penalty | -23.509941 | -3.8% | 3.8% | 100.0% |
| vert_penalty | -7.367474 | -1.2% | 1.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
