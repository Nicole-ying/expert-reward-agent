# Training Feedback

## Final-policy outcome
score=212.370151, len=632.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-30.337260, 318.220843]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 404.485881 | 99.8% | 99.8% | 99.5% |
| angle_penalty | -0.841930 | -0.2% | 0.2% | 1.3% |
| angvel_penalty | -0.000814 | -0.0% | 0.0% | 99.8% |
| vert_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
