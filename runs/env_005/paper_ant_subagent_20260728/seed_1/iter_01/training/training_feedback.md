# Training Feedback

## Final-policy outcome
score=-11.317827, len=59.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-65.458824, 19.185297]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 72.310818 | 92.8% | 92.8% | 84.1% |
| upright_penalty | -5.412590 | -6.9% | 6.9% | 100.0% |
| height_penalty | -0.170182 | -0.2% | 0.2% | 2.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
