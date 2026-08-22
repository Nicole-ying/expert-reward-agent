# Training Feedback

## Final-policy outcome
score=-55.524057, len=292.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-105.105425, 258.968333]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress_reward | 80.078792 | 91.1% | 95.0% | 100.0% |
| upright_stability_penalty | -4.408739 | -5.0% | 5.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
