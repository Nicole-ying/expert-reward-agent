# Training Feedback

## Final-policy outcome
score=-86.300185, len=105.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.688014, -84.316982]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 29.558881 | 70.5% | 70.6% | 100.0% |
| air_stability_penalty | -10.686779 | -25.5% | 25.5% | 68.0% |
| balance_penalty | -1.615357 | -3.9% | 3.9% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
