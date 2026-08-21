# Training Feedback

## Final-policy outcome
score=190.749948, len=1052.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-21.208159, 299.292479]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress_reward | 369.462443 | 99.9% | 99.9% | 100.0% |
| upright_stability_penalty | -0.272956 | -0.1% | 0.1% | 80.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
