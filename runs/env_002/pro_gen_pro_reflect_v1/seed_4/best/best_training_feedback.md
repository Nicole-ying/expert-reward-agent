# Training Feedback

## Final-policy outcome
score=316.493351, len=892.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[314.552502, 317.895769]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_gate | 879.151491 | 46.8% | 46.8% | 100.0% |
| forward_reward | 503.443935 | 26.8% | 26.8% | 99.7% |
| gated_forward_reward | 495.486809 | 26.4% | 26.4% | 99.7% |
| angvel_penalty | -0.000652 | -0.0% | 0.0% | 99.8% |
| vert_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
