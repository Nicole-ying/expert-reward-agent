# Training Feedback

## Final-policy outcome
score=0.719501, len=11.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-4.237267, 10.044042]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 13.114699 | 75.4% | 90.0% | 99.6% |
| lateral_penalty | -1.562115 | -9.0% | 9.0% | 100.0% |
| height_penalty | -0.133082 | -0.8% | 0.8% | 16.5% |
| action_penalty | -0.034932 | -0.2% | 0.2% | 100.0% |
| upright_penalty | -0.014110 | -0.1% | 0.1% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
